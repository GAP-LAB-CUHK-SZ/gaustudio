import os
import numpy as np
import tempfile
import torch
import trimesh
from tqdm import tqdm
from typing import List, Tuple
from pathlib import Path
import PIL.Image
import torchvision.transforms as tvf

from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.pcd import PcdInitializer
from gaustudio.datasets import Camera
from gaustudio.datasets.utils import focal2fov


from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.inference import inference
from mini_dust3r.image_pairs import make_pairs
from mini_dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from mini_dust3r.cloud_opt.base_opt import BasePCOptimizer
from mini_dust3r.viz import pts3d_to_trimesh, cat_meshes
from mini_dust3r.utils.image import load_images, ImgNorm

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)

@initializers.register('dust3r')
class Dust3rInitializer(PcdInitializer):
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        self.ws_dir = self.initializer_config.get('workspace_dir')
        if self.ws_dir is None:
            self.ws_dir = Path(tempfile.mkdtemp())
            print(f"No workspace directory provided. Using temporary directory: {self.ws_dir}")
        else:
            self.ws_dir = Path(self.ws_dir)

        os.makedirs(self.ws_dir, exist_ok=True)
        self.model_path = str(self.ws_dir / 'fused.ply')
        self.dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(
            "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to("cuda")
        self.imgs = []
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
            print(f"Due to memory limitations, only the first {self.max_images} images will be processed.")
        
        for i, camera in enumerate(dataset[:self.max_images]):
            img_path = self.ws_dir / f"{i:08d}.jpg"
            img = PIL.Image.fromarray((camera.image.numpy() * 255).astype(np.uint8))
            W1, H1 = img.size
            if self.image_size == 224:
                # resize short side to 224 (then crop)
                img = _resize_pil_image(img, round(self.image_size * max(W1 / H1, H1 / W1)))
            else:
                # resize long side to 512
                img = _resize_pil_image(img, self.image_size)
            W, H = img.size
            cx, cy = W // 2, H // 2
            if self.image_size == 224:
                half = min(cx, cy)
                img = img.crop((cx - half, cy - half, cx + half, cy + half))
            else:
                halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
                if not (square_ok) and W == H:
                    halfh = 3 * halfw / 4
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
                
            img_tensor = torch.tensor(np.asarray(img))
            self.imgs.append(dict(
                img=ImgNorm(img)[None],
                unnorm_img=img_tensor[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(self.imgs),
                instance=str(len(self.imgs)),
            ))

    def process_dataset(self):
        pairs: List[Tuple[dict, dict]] = make_pairs(
            self.imgs, scene_graph="swin", prefilter=None, symmetrize=True
        )
        output = inference(pairs, self.dust3r_model, "cuda", batch_size=1)

        mode = (
            GlobalAlignerMode.PointCloudOptimizer
            if len(self.imgs) > 2
            else GlobalAlignerMode.PairViewer
        )
        scene: BasePCOptimizer = global_aligner(
            dust3r_output=output, device="cuda", mode=mode
        )

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            scene.compute_global_alignment(
                init="mst", niter=100, schedule="linear", lr=0.01
            )

        K_b33 = scene.get_intrinsics().numpy(force=True)
        world_T_cam_b44 = scene.get_im_poses().numpy(force=True)
        depth_hw_list = [depth.numpy(force=True) for depth in scene.get_depthmaps()]
        pts3d_list = [pt3d.numpy(force=True) for pt3d in scene.get_pts3d()]
        masks_list = [mask.numpy(force=True) for mask in scene.get_masks()]
        images_list = [img_dict['unnorm_img'].squeeze() for img_dict in self.imgs]
        # for i, (camera, K, world_T_cam, depth_hw, pts3d, mask, img_dict) in enumerate(zip(dataset[:self.max_images], K_b33, world_T_cam_b44, depth_hw_list, pts3d_list, masks_list, self.imgs)):
        #     R = np.transpose(np.linalg.inv(world_T_cam[:3, :3]))
        #     T = world_T_cam[:3, 3] * 100
            
        #     fx, fy = K[0, 0], K[1, 1]
        #     image_height, image_width = self.image_size, self.image_size
            
        #     FoVy = focal2fov(fy, image_height)
        #     FoVx = focal2fov(fx, image_width)
            
        #     new_camera = Camera(R=R, T=T, 
        #                         FoVy=FoVy, FoVx=FoVx, 
        #                         image=img_dict['img'].squeeze().permute(1, 2, 0), 
        #                         image_name=camera.image_name,
        #                         image_width=image_width, 
        #                         image_height=image_height)
            
        #     new_camera.depth = torch.from_numpy(depth_hw)
        #     new_camera.pts3d = torch.from_numpy(pts3d)
        #     new_camera.mask = torch.from_numpy(mask)
            
        #     self.cameras.append(new_camera)

        point_cloud = np.concatenate([p[m] for p, m in zip(pts3d_list, masks_list)])
        colors = np.concatenate([c[m] for c, m in zip(images_list, masks_list)])
        self.point_cloud = trimesh.PointCloud(point_cloud.reshape(-1, 3), colors=colors.reshape(-1, 3))
        
        self.point_cloud.export(self.model_path)
        print(f"Fused point cloud saved to {self.model_path}")