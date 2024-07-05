import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from gaustudio import datasets
from gaustudio.datasets.utils import focal2fov, getNerfppNorm, camera_to_JSON
from typing import List, Dict 
from pathlib import Path
import math

class NerfStudioDatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        self.image_path = Path(config['source_path'])
        
        self._initialize()
        self.ply_path = None
        
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def _initialize(self):
        all_cameras_unsorted = []
        
        with open(self.source_path / f"transforms.json", 'r') as f:
            meta = json.load(f)
    
        
        for _frame in meta['frames']:
            width, height = int(_frame['w']), int(_frame['h'])
            fx, fy, cx, cy = int(_frame['fl_x']), int(_frame['fl_y']), int(_frame['cx']), int(_frame['cy'])                
            
            FoVy = focal2fov(fy, height)
            FoVx = focal2fov(fx, width)
        
            image_name = f"{_frame['file_path']}"
            image_path = self.image_path / image_name
            
            c2w = np.array(_frame['transform_matrix'])
            c2w[:,1:3] *= -1
            
            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image_path=image_path, image_width=width, image_height=height)
            all_cameras_unsorted.append(_camera)
        self.all_cameras = sorted(all_cameras_unsorted, key=lambda x: x.image_name) 
        self.nerf_normalization = getNerfppNorm(self.all_cameras)
        self.cameras_extent = self.nerf_normalization["radius"]
    
    def export(self, save_path):
        json_cams = []
        camlist = []
        camlist.extend(self.all_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(save_path, 'w') as file:
            json.dump(json_cams, file)
            
@datasets.register('nerfstudio')
class NerfStudioDataset(Dataset, NerfStudioDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]