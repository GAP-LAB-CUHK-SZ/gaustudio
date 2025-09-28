import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torchvision
from PIL import Image, PngImagePlugin
from tqdm import tqdm

from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.base import BaseInitializer
from gaustudio.utils.colmap_utils import (
    COLMAPDatabase,
    create_cameras_and_points_bin,
    create_images_bin,
)

@initializers.register('colmap')
class ColmapInitializer(BaseInitializer):
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        self.ws_dir = self.initializer_config.get('workspace_dir')
        if self.ws_dir is None:
            self.ws_dir = tempfile.mkdtemp()
            print(f"No workspace directory provided. Using temporary directory: {self.ws_dir}")

        os.makedirs(self.ws_dir, exist_ok=True)

        self.db_path = os.path.join(self.ws_dir, "database.db")
        self.images_dir = os.path.join(self.ws_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.pose_dict = {}
        self.filename_lookup = {}
    
    def __call__(self, model, dataset, overwrite=False):
        # Skip processing if sparse results exists
        if not os.path.exists(f'{self.ws_dir}/sparse') or overwrite:
            self.cache_dataset(dataset)
            self.process_dataset()

        model = self.build_model(model)
        return model

    def cache_dataset(self, dataset):
        print("Collecting dataset for batch processing...")

        images_to_encode = []
        masks_data = []
        depths_data = []
        pose_data = {}
        filename_lookup = {}
        intrinsics = None

        for img_id, camera in enumerate(tqdm(dataset, desc="Collecting data")):
            img_name = str(img_id).zfill(8)

            # Track pose information (stored as camera-to-world)
            pose_data[img_name] = camera.extrinsics.inverse()

            # Cache intrinsics once; assume identical across the capture
            if intrinsics is None:
                cam_intr = camera.intrinsics
                intrinsics = {
                    'width': camera.image_width,
                    'height': camera.image_height,
                    'fx': float(cam_intr[0, 0]),
                    'fy': float(cam_intr[1, 1]),
                    'cx': float(cam_intr[0, 2]),
                    'cy': float(cam_intr[1, 2]),
                }

            # Always encode from the in-memory tensor
            target_name = f"{img_name}.jpg"
            images_to_encode.append((img_name, camera.image))

            filename_lookup[img_name] = target_name

            if camera.mask is not None:
                masks_data.append((img_name, camera.mask))

            if camera.depth is not None:
                depths_data.append((img_name, camera.depth))

        self.pose_dict = pose_data
        self.filename_lookup = filename_lookup

        images_dir_path = Path(self.images_dir)

        if images_to_encode:
            print("Encoding images from tensors...")
            for img_name, image_tensor in tqdm(images_to_encode, desc="Saving images"):
                target_name = filename_lookup[img_name]
                dst_path = images_dir_path / target_name
                if dst_path.exists():
                    dst_path.unlink()

                tensor = image_tensor
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.detach()
                    if tensor.device.type != 'cpu':
                        tensor = tensor.cpu()
                    if tensor.dtype != torch.uint8:
                        tensor = (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
                    tensor = tensor.contiguous()
                    img_np = tensor.numpy()
                elif isinstance(tensor, np.ndarray):
                    if tensor.dtype != np.uint8:
                        img_np = np.clip(np.rint(tensor * 255.0), 0, 255).astype(np.uint8)
                    else:
                        img_np = tensor
                else:
                    raise TypeError("Unsupported image type for encoding")

                if img_np.ndim == 3 and img_np.shape[2] > 3:
                    img_np = img_np[..., :3]
                if img_np.ndim == 2:
                    pil_image = Image.fromarray(img_np, mode='L')
                else:
                    pil_image = Image.fromarray(img_np)

                save_kwargs = {}
                if Path(target_name).suffix.lower() in {'.jpg', '.jpeg'}:
                    save_kwargs['quality'] = 95
                pil_image.save(dst_path, **save_kwargs)

        if masks_data:
            self.masks_dir = os.path.join(self.ws_dir, 'masks')
            os.makedirs(self.masks_dir, exist_ok=True)
            print("Batch saving masks...")
            for img_name, mask in tqdm(masks_data, desc="Saving masks"):
                mask_tensor = mask.detach() if isinstance(mask, torch.Tensor) else torch.from_numpy(mask)
                if mask_tensor.device.type != 'cpu':
                    mask_tensor = mask_tensor.cpu()
                mask_tensor = mask_tensor.float()
                if mask_tensor.dim() == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                elif mask_tensor.dim() == 3 and mask_tensor.shape[-1] == 1:
                    mask_tensor = mask_tensor.permute(2, 0, 1)
                elif mask_tensor.dim() == 3 and mask_tensor.shape[0] not in {1, 3}:
                    mask_tensor = mask_tensor.permute(2, 0, 1)
                mask_tensor = mask_tensor.contiguous()
                torchvision.utils.save_image(mask_tensor, os.path.join(self.masks_dir, f'{img_name}.png'))

        if depths_data:
            self.depths_dir = os.path.join(self.ws_dir, 'depths')
            os.makedirs(self.depths_dir, exist_ok=True)
            print("Batch saving depths...")
            for img_name, depth in tqdm(depths_data, desc="Saving depths"):
                depth_tensor = depth.detach() if isinstance(depth, torch.Tensor) else torch.from_numpy(depth)
                if depth_tensor.device.type != 'cpu':
                    depth_tensor = depth_tensor.cpu()
                depth_tensor = depth_tensor.float()
                depth_max = float(depth_tensor.max().item() + 1e-6)
                normalized = (depth_tensor / depth_max).clamp(min=0.0, max=1.0)
                depth_map_16bit = (normalized * 65535).numpy().astype(np.uint16)
                depth_img = Image.fromarray(depth_map_16bit)
                meta = PngImagePlugin.PngInfo()
                meta.add_text("depth_max", str(depth_max))
                depth_img.save(os.path.join(self.depths_dir, f'{img_name}.png'), "PNG", pnginfo=meta)

        if intrinsics is None:
            raise RuntimeError("Failed to compute camera intrinsics from dataset")

        print("Creating camera and points model data...")
        create_cameras_and_points_bin(self.ws_dir, intrinsics)

    def process_dataset(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        pycolmap.extract_features(image_path=self.images_dir, database_path=self.db_path)
        pycolmap.match_exhaustive(self.db_path)
        
        db = COLMAPDatabase.connect(self.db_path)
        images = list(db.execute('select * from images'))
        create_images_bin(self.ws_dir, self.pose_dict, images, self.filename_lookup)

        sparse_reconstruction_folder = os.path.join(self.ws_dir, 'sparse', '0')
        os.makedirs(sparse_reconstruction_folder, exist_ok=True)

        reference = pycolmap.Reconstruction(os.path.join(self.ws_dir, 'model'))
        pycolmap.triangulate_points(reference, self.db_path, self.images_dir, sparse_reconstruction_folder)
        shutil.rmtree(os.path.join(self.ws_dir, 'model'))

    def build_model(self, model):
        """
        Build the 3D model point cloud by reading the COLMAP points3D binary file.
        :param model: model instance to update
        :return: Updated model instance
        """
        candidates = ['sparse/0/points3D.bin', 'sparse/points3D.bin']
        from gaustudio.utils.colmap_utils import read_points3D_binary
        
        for candidate in candidates:
            candidate_path = os.path.join(self.ws_dir, candidate)
            if os.path.exists(candidate_path):
                try:
                    pts3d = read_points3D_binary(candidate_path)
                except Exception as e:
                    print(f"Failed to read points3D binary file: {e}")
                    raise

                try:
                    xyz = np.array([pts3d[k].xyz for k in pts3d])
                    rgb = np.array([pts3d[k].rgb / 255 for k in pts3d])
                    model.create_from_attribute(xyz=xyz, rgb=rgb)
                except Exception as e:
                    print(f"Failed to update point cloud: {e}")
                    raise
                break
        else:
            print(f"No points3D binary file found in {self.ws_dir} with candidates {candidates}")
            raise
        return model
