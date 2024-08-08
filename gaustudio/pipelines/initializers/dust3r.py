import os
import pycolmap
import numpy as np
import torchvision
import torch

from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.base import BaseInitializer
from gaustudio.datasets import Camera
from gaustudio.datasets.utils import focal2fov

import cv2
from tqdm import tqdm
import numpy as np
import trimesh
import tempfile
from copy import deepcopy

try:
    from mini_dust3r.api import OptimizedResult, inferece_dust3r, log_optimized_result
    from mini_dust3r.model import AsymmetricCroCo3DStereo
    dust3r_installed = True
except:
    dust3r_installed = False

@initializers.register('dust3r')
class Dust3rInitializer(BaseInitializer):
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        if not dust3r_installed:
            raise ImportError("Please install mini_dust3r to use Dust3rInitializer.")
        self.ws_dir = self.initializer_config.get('workspace_dir')
        if self.ws_dir is None:
            self.ws_dir = tempfile.mkdtemp()
            print(f"No workspace directory provided. Using temporary directory: {self.ws_dir}")

        os.makedirs(self.ws_dir, exist_ok=True)

        self.model = AsymmetricCroCo3DStereo.from_pretrained(
            "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to("cuda")
        self.cameras = []

    def __call__(self, model, dataset, overwrite=False):
        # Skip processing if sparse results exists
        if not os.path.exists(f'{self.ws_dir}/fused_mesh.ply') or overwrite:
            self.cache_dataset(dataset)
            self.process_dataset()
        model = self.build_model(model)
        return model
    
    def cache_dataset(self, dataset, max_images = 20):
        self.cache_image_paths = []
        if len(dataset) > max_images:
            print(f"Due to memory limitations, only the first {max_images} images will be cached.")
        for img_id, camera in enumerate(dataset[:max_images]):
            img_name = str(img_id).zfill(8)
            img_path = os.path.join(self.ws_dir, f'{img_name}.jpg')
            torchvision.utils.save_image(camera.image.permute(2, 0, 1), img_path)
        
            self.cache_image_paths.append(img_path)

    def process_dataset(self):
        optimized_results: OptimizedResult = inferece_dust3r(
            image_dir_or_list=self.cache_image_paths,
            model=self.model,
            device="cuda",
        )
        
        for _i, _img_path in enumerate(self.cache_image_paths):
            _ori_img = cv2.imread(_img_path)
            _ori_img = cv2.cvtColor(_ori_img, cv2.COLOR_BGR2RGB)
            _ori_img = torch.from_numpy(_ori_img) / 255.0
            _ori_height, _ori_width, _ = _ori_img.shape
            
            _intrinsic = optimized_results.K_b33[_i]
            _extrinsic = np.linalg.inv(optimized_results.world_T_cam_b44[_i])
            _image_tensor = torch.from_numpy(optimized_results.rgb_hw3_list[_i])
            image_height, image_width, _ = _image_tensor.shape

            R = np.transpose(_extrinsic[:3, :3])
            T = _extrinsic[:3, 3] * 100
            fx, fy, cx, cy = _intrinsic[0, 0], _intrinsic[1, 1], \
                            _intrinsic[0, 2], _intrinsic[1, 2]                
            
            FoVy = focal2fov(fy, image_height)
            FoVx = focal2fov(fx, image_width)
            _camera = Camera(R=R, T=T, 
                             FoVy=FoVy, FoVx=FoVx, 
                            image=_image_tensor, 
                            image_name=os.path.basename(_img_path),
                            image_width=image_width, 
                            image_height=image_height)

            _camera = _camera.downsample((_ori_width, _ori_height))
            _camera.image = _ori_img
            self.cameras.append(_camera)

        # Save the fused point cloud
        optimized_results.point_cloud.vertices *= 100
        optimized_results.point_cloud.export(os.path.join(self.ws_dir, 'fused.ply'))
        print(f"Fused point cloud saved to {self.ws_dir}/fused.ply")

    def build_model(self, model):
        print("Building model...")
        
        _pcd = trimesh.load(os.path.join(self.ws_dir, 'fused.ply'))
        xyz = torch.from_numpy(_pcd.vertices).float().cuda()
        rgb = torch.from_numpy(_pcd.visual.vertex_colors[:, :3]).float().cuda() / 255.0
        model.create_from_attribute(xyz=xyz, rgb=rgb)
        return model