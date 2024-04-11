import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from gaustudio import datasets
from gaustudio.datasets.utils import focal2fov, getNerfppNorm
from typing import List, Dict 
from pathlib import Path

class NSVFDatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        
        self.image_dir = self.source_path / "rgb"
        self.pose_dir = self.source_path /  "pose"
        self.intrinsic_path = self.source_path / 'intrinsics.txt'
        
        self.image_filenames = []
        for f in os.listdir(self.image_dir):
            if os.path.basename(f).startswith('0_train'):
                self.image_filenames.append(os.path.join(self.image_dir, f))
        self.image_filenames = sorted(self.image_filenames, key=lambda fn: os.path.splitext(os.path.basename(fn))[0].split('_')[-1])
        self._initialize()
        self.ply_path = None
        
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def _initialize(self):
        all_cameras_unsorted = []
        
        with open(self.intrinsic_path) as f:
            intrs = f.readline().split()
            fx = fy = float(intrs[0])
            cx = float(intrs[1])
            cy = float(intrs[2])
            
        for image_path in self.image_filenames:
            _id = os.path.splitext(os.path.basename(image_path))[0]
            _image = cv2.imread(image_path)
            height, width, _ = _image.shape

            c2w = np.loadtxt(self.pose_dir / f'{_id}.txt' )

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
    
@datasets.register('nsvf')
class NSVFDataset(Dataset, NSVFDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]