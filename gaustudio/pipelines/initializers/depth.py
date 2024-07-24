import os
import pycolmap
import numpy as np
import torchvision
import torch
from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.base import BaseInitializer
import math
from tqdm import tqdm
import numpy as np
import trimesh
import tempfile
from copy import deepcopy

def inverse_sigmoid(x):
    return np.log(x / (1 - x))

@initializers.register('unproject')
class UnprojectInitializer(BaseInitializer):
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        self.ws_dir = self.initializer_config.get('workspace_dir')
        if self.ws_dir is None:
            self.ws_dir = tempfile.mkdtemp()
            print(f"No workspace directory provided. Using temporary directory: {self.ws_dir}")

        os.makedirs(self.ws_dir, exist_ok=True)

    def __call__(self, model, dataset, overwrite=False):
        # Skip processing if sparse results exists
        if not os.path.exists(f'{self.ws_dir}/fused.ply') or overwrite:
            self.cache_dataset(dataset)
            self.process_dataset()
        model = self.build_model(model)
        return model
    
    def cache_dataset(self, dataset):
        print("Caching point cloud...")
        os.makedirs(self.ws_dir, exist_ok=True)  # Ensure the directory exists

        self.cache_pcd_paths = []
        for _id, camera in tqdm(enumerate(dataset), total=len(dataset)):  # Add total for tqdm progress bar
            camera = deepcopy(camera)
            camera = camera.downsample_scale(4).to("cuda")
            world_xyz = camera.depth2point(coordinate='world').cpu().numpy()  # Ensure conversion to numpy
            world_rgb = camera.image.cpu().numpy()  # Ensure conversion to numpy
            
            # Adopted from https://github.com/spla-tam/SplaTAM/blob/main/scripts/splatam.py#L100
            world_scale = camera.depth / ((camera.fx + camera.fy) / 2)
            pcd = np.hstack((world_xyz.reshape(-1, 3), world_rgb.reshape(-1, 3), 
                             world_scale.cpu().numpy().reshape(-1, 1)))
            pcd_path = os.path.join(self.ws_dir, f"point_cloud_{_id}.bin")
            pcd.astype('float16').tofile(pcd_path)
            self.cache_pcd_paths.append(pcd_path)

    def process_dataset(self):
        pcds = []
        scales = []
        for _cache_pcd_path in self.cache_pcd_paths:
            try:
                pcd = np.fromfile(_cache_pcd_path, dtype='float16').reshape(-1, 7)  # Specify dtype
                pcds.append(pcd[:, :6])
                scales.append(pcd[:, 6:])
            except Exception as e:
                print(f"Error reading file {_cache_pcd_path}: {e}")
        
        if pcds:
            scales = np.concatenate(scales, axis=0)
            pcds = np.concatenate(pcds, axis=0)
            point_cloud = trimesh.PointCloud(vertices=pcds[:, :3], colors=(pcds[:, 3:6] * 255).astype('uint8'))
            scales.astype('float16').tofile(os.path.join(self.ws_dir, 'scales.bin'))
            point_cloud.export(os.path.join(self.ws_dir, 'fused.ply'))
            print(f"Fused point cloud saved to {self.ws_dir}/fused.ply")
        else:
            print("No point clouds to process.")
        
    def build_model(self, model):
        print("Building point cloud...")
        
        point_cloud = trimesh.load(os.path.join(self.ws_dir, 'fused.ply'))
        if os.path.exists(os.path.join(self.ws_dir, 'scales.bin')):
            scales = np.fromfile(os.path.join(self.ws_dir, 'scales.bin'), dtype='float16').reshape(-1, 1)
            log_scales = torch.from_numpy(np.log(scales)).repeat(1, 3).float().cuda()
        else:
            log_scales = None
        xyz = torch.from_numpy(point_cloud.vertices).float().cuda()
        opacity = inverse_sigmoid(0.5 * np.ones((xyz.shape[0], 1)))
        rgb = torch.from_numpy(point_cloud.visual.vertex_colors[:, :3]).float().cuda() / 255.0
        model.create_from_attribute(xyz=xyz, rgb=rgb, opacity=opacity, scale=log_scales)
        return model