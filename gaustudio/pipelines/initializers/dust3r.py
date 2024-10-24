import os
import numpy as np
import tempfile
import torch
import open3d as o3d
from typing import List, Tuple
from pathlib import Path
import PIL.Image
from tqdm import tqdm

from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.pcd import PcdInitializer
from gaustudio.datasets import Camera
from gaustudio.datasets.utils import focal2fov

try:
    from mini_dust3r.model import AsymmetricCroCo3DStereo
    from mini_dust3r.inference import inference
    from mini_dust3r.image_pairs import make_pairs
    from mini_dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from mini_dust3r.cloud_opt.base_opt import BasePCOptimizer
    from mini_dust3r.viz import pts3d_to_trimesh, cat_meshes
    from mini_dust3r.utils.image import load_images, ImgNorm
    DUST3R_AVAILABLE = True
except:
    DUST3R_AVAILABLE = False

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def remove_normal_outliers(pcd, nb_neighbors=20, angle_threshold=np.pi/4):
    normals = np.asarray(pcd.normals)
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    inlier_indices = []
    
    for i in range(len(pcd.points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], nb_neighbors)
        neighbor_normals = normals[idx[1:]]  # Exclude the point itself
        angles = np.arccos(np.abs(np.dot(neighbor_normals, normals[i])))
        if np.mean(angles) < angle_threshold:
            inlier_indices.append(i)
    
    return pcd.select_by_index(inlier_indices)

def clean_point_cloud(pcd, nb_neighbors=50, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_clean = pcd.select_by_index(ind)
    
    pcd_clean = remove_normal_outliers(pcd_clean)
    
    return pcd_clean

def combine_and_clean_point_clouds(pcds, max_points=500000):
    pcd_combined = o3d.geometry.PointCloud()
    for p3d in pcds:
        pcd_combined += p3d
    total_points = len(pcd_combined.points)
    if total_points > max_points:
        every_k = total_points // max_points
    else:
        every_k = 1
    pcd_combined = pcd_combined.uniform_down_sample(every_k)

    pcd_clean = clean_point_cloud(pcd_combined)
    return pcd_clean

def pts2normal(pts):
    h, w, _ = pts.shape
    
    # Compute differences in x and y directions
    dx = pts[2:, 1:-1] - pts[:-2, 1:-1]
    dy = pts[1:-1, 2:] - pts[1:-1, :-2]
    
    # Compute normal vectors using cross product
    normals = np.cross(dx, dy, axis=-1)

    # Normalize the normal vectors
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normal_map = normals / np.where(norms == 0, 1, norms)  # Avoid division by zero
    
    # Create padded normal map
    padded_normal_map = np.zeros_like(pts)
    padded_normal_map[1:-1, 1:-1, :] = normal_map

    # Pad the borders
    padded_normal_map[0, 1:-1, :] = normal_map[0, :, :]  # Top edge
    padded_normal_map[-1, 1:-1, :] = normal_map[-1, :, :]  # Bottom edge
    padded_normal_map[1:-1, 0, :] = normal_map[:, 0, :]  # Left edge
    padded_normal_map[1:-1, -1, :] = normal_map[:, -1, :]  # Right edge
    
    # Pad the corners
    padded_normal_map[0, 0, :] = normal_map[0, 0, :]  # Top-left corner
    padded_normal_map[0, -1, :] = normal_map[0, -1, :]  # Top-right corner
    padded_normal_map[-1, 0, :] = normal_map[-1, 0, :]  # Bottom-left corner
    padded_normal_map[-1, -1, :] = normal_map[-1, -1, :]  # Bottom-right corner
    
    return padded_normal_map

@initializers.register('dust3r')
class Dust3rInitializer(PcdInitializer):
    def __init__(self, initializer_config):
        assert DUST3R_AVAILABLE, "mini_dust3r is not installed"
        super().__init__(initializer_config)
        self.ws_dir = self.initializer_config.get('workspace_dir')
        self.prune_background = self.initializer_config.get('prune_bg', False)
        if self.ws_dir is None:
            self.ws_dir = Path(tempfile.mkdtemp())
            print(f"No workspace directory provided. Using temporary directory: {self.ws_dir}")
        else:
            self.ws_dir = Path(self.ws_dir)

        os.makedirs(self.ws_dir, exist_ok=True)
        self.model_path = str(self.ws_dir / 'fused.ply')
        self.dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(
            "https://huggingface.co/camenduru/dust3r/resolve/main/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        ).to("cuda")
        self.imgs = []
        self.poses = []
        self.intrinsics = []
        self.image_size = 512
        self.max_images = 20

    def __call__(self, model, dataset, overwrite=False):
        if not os.path.exists(self.model_path) or overwrite:
            self.cache_dataset(dataset)
            self.process_dataset()
        model = self.build_model(model)
        return model
    
    def cache_dataset(self, dataset, square_ok=False):
        if len(dataset) > self.max_images:
            print(f"Downsampling dataset to {self.max_images} images using interval-based selection.")
            interval = len(dataset) // self.max_images
            dataset = dataset[::interval][:self.max_images]  # Ensure we don't exceed max_images
        
        for i, camera in enumerate(dataset[:self.max_images]):
            img = PIL.Image.fromarray((camera.image.numpy() * 255).astype(np.uint8))
            mask = PIL.Image.fromarray((camera.mask.numpy() * 255).astype(np.uint8))  # Assuming mask is available in camera object
            original_W, original_H = img.size
            
            _intrinsics = camera.intrinsics.cpu().numpy()
            fx, fy = _intrinsics[0, 0], _intrinsics[1, 1]
            cx, cy = _intrinsics[0, 2], _intrinsics[1, 2]

            # Calculate margins to crop
            min_margin_x = min(cx, original_W - cx)
            min_margin_y = min(cy, original_H - cy)

            # Calculate crop box
            left = max(0, cx - min_margin_x)
            top = max(0, cy - min_margin_y)
            right = min(original_W, cx + min_margin_x)
            bottom = min(original_H, cy + min_margin_y)

            # Crop image and mask
            img = img.crop((left, top, right, bottom))
            mask = mask.crop((left, top, right, bottom))
            crop_W, crop_H = img.size

            # Adjust intrinsics after cropping
            cx -= left
            cy -= top

            if self.image_size == 224:
                # resize short side to 224 (then crop)
                scale = self.image_size / min(original_W, original_H)
            else:
                # resize long side to 512
                scale = self.image_size / max(original_W, original_H)

            # Calculate new size ensuring width and height are multiples of 16
            new_W = round(original_W * scale / 16) * 16
            new_H = round(original_H * scale / 16) * 16

            # Adjust size if not square_ok and W == H
            if not square_ok and new_W == new_H:
                if new_W > new_H:
                    new_H = round(new_H * 0.75 / 16) * 16
                else:
                    new_W = round(new_W * 0.75 / 16) * 16

            # Calculate actual scale factors
            scale_W = new_W / original_W
            scale_H = new_H / original_H

            # Resize image and mask
            img = img.resize((new_W, new_H), PIL.Image.LANCZOS)
            mask = mask.resize((new_W, new_H), PIL.Image.NEAREST)  # Use NEAREST for mask to preserve binary values

            # Adjust intrinsics
            fx *= scale_W
            fy *= scale_H
            cx *= scale_W
            cy *= scale_H

            img_tensor = torch.tensor(np.asarray(img))
            mask_tensor = torch.tensor(np.asarray(mask))
            self.imgs.append(dict(
                img=ImgNorm(img)[None],
                unnorm_img=img_tensor[None],
                mask=mask_tensor[None],  # Add mask to the dictionary
                true_shape=np.int32([img.size[::-1]]),
                idx=len(self.imgs),
                instance=str(len(self.imgs)),
            ))

            pose = torch.linalg.inv(camera.extrinsics)
            self.poses.append(pose)

            intrinsic = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            self.intrinsics.append(intrinsic)

        # Convert poses and intrinsics to tensors
        self.poses = torch.stack(self.poses)
        self.intrinsics = torch.stack(self.intrinsics)

    def process_dataset(self):
        pairs: List[Tuple[dict, dict]] = make_pairs(
            self.imgs, scene_graph="complete", prefilter=None, symmetrize=True
        )
        output = inference(pairs, self.dust3r_model, "cuda", batch_size=16)
        self.dust3r_model = None
        images_list = [img_dict['unnorm_img'].squeeze() for img_dict in self.imgs]
        fg_masks_list = [img_list['mask'].squeeze() for img_list in self.imgs]
        del pairs, self.imgs
        scene: BasePCOptimizer = global_aligner(
            dust3r_output=output, device="cuda", mode=GlobalAlignerMode.PointCloudOptimizer
        )

        # Preset poses and intrinsics
        scene.preset_pose(self.poses)
        scene.preset_focal([K.diagonal()[:2].mean() for K in self.intrinsics])
        scene.preset_principal_point([K[:2, 2] for K in self.intrinsics])
        
        scene.compute_global_alignment(
            init="known_poses", niter=500, schedule="cosine", lr=0.01
        )

        pts3d_list = [pt3d.numpy(force=True) for pt3d in scene.get_pts3d()]
        masks_list = [mask.numpy(force=True) for mask in scene.get_masks()]
        pcds = []
        for pts, img, mask, fg_mask in tqdm(zip(pts3d_list, images_list, masks_list, fg_masks_list), desc="Fusing Point Cloud"):
            if mask.mean() == 0:
                continue
            if self.prune_background:
                # combine fg_mask and confidance mask
                mask = np.logical_and(mask, fg_mask.cpu().numpy())
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[mask].reshape(-1, 3))
            pts_normal = pts2normal(pts)
            pcd.normals = o3d.utility.Vector3dVector(pts_normal[mask].reshape(-1, 3))

            colors = (img[mask] / 255.0).cpu().numpy()
            pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
            pcds.append(pcd)
        combined_pcd = combine_and_clean_point_clouds(pcds)   
        o3d.io.write_point_cloud(self.model_path, combined_pcd)