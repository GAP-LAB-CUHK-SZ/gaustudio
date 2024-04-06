import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from gaustudio import datasets
from gaustudio.datasets.utils import focal2fov, getNerfppNorm
from typing import List, Dict 
from pathlib import Path

class MobileBrickDatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        
        self.image_dir = self.source_path / "image"
        self.pose_dir = self.source_path /  "pose"
        self.intrinsic_dir = self.source_path / 'intrinsic'
        self.image_filenames = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)],
                                      key=lambda fn: int(os.path.splitext(os.path.basename(fn))[0]))
        
        self._initialize()
        self.ply_path = None
        
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def _initialize(self):
        all_cameras_unsorted = []
        
        for image_path in self.image_filenames:
            
            _id = os.path.splitext(os.path.basename(image_path))[0]
            _image = cv2.imread(image_path)
            height, width, _ = _image.shape

            intr = np.loadtxt(self.intrinsic_dir /  f'{_id}.txt')
            fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]                
            c2w = np.loadtxt(self.pose_dir / f'{_id}.txt')

            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            FoVy = focal2fov(fy, height)
            FoVx = focal2fov(fx, width)
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image_path=image_path, image_width=width, image_height=height)
            all_cameras_unsorted.append(_camera)
        self.all_cameras = sorted(all_cameras_unsorted, key=lambda x: x.image_name) 
        self.nerf_normalization = getNerfppNorm(self.all_cameras)
        self.cameras_extent = self.nerf_normalization["radius"]
    
@datasets.register('mobilebrick')
class MobileBrickDataset(Dataset, MobileBrickDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]