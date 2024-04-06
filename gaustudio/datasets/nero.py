import os
import json
import cv2
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from gaustudio import datasets
from gaustudio.datasets.utils import focal2fov, getNerfppNorm
from typing import List, Dict 
from pathlib import Path
import math
import pickle

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


class NeRODatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        self.image_path = Path(config['source_path'])
        
        self.image_ids = sorted([int(f.stem.split('-')[0]) for f in self.source_path.glob("*-camera.pkl")])


        self._initialize()
        self.ply_path = None
        
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def _initialize(self):
        all_cameras_unsorted = []
    
        for _id in self.image_ids:
            cam = read_pickle(self.source_path / f"{_id}-camera.pkl")
            
            image_path = self.image_path / f"{_id}.png"
            _image = cv2.imread(str(image_path))
            height, width, _ = _image.shape
            extrinsics = np.eye(4)
            
            extrinsics[:3, :] = cam[0].astype(np.float32)
            intr = cam[1].astype(np.float32)
            fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]                
            
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            FoVy = focal2fov(fy, height)
            FoVx = focal2fov(fx, width)
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image_path=image_path, image_width=width, image_height=height)
            all_cameras_unsorted.append(_camera)
        self.all_cameras = sorted(all_cameras_unsorted, key=lambda x: x.image_name) 
        self.nerf_normalization = getNerfppNorm(self.all_cameras)
        self.cameras_extent = self.nerf_normalization["radius"]
    
@datasets.register('nero')
class NeRODataset(Dataset, NeRODatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]