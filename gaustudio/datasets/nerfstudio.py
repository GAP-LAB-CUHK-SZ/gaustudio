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
import torch

class NerfStudioDatasetBase:
    transform_path = 'transforms.json'
    
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        self.image_path = Path(config['source_path'])
        self.masks_dir = Path(config['source_path'])
        self.white_background = config.get('white_background', False)
        self.w_mask = config.get('w_mask', False)
        
        self._initialize()
        
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def _initialize(self):
        all_cameras_unsorted = []
        
        with open(self.source_path / self.transform_path, 'r') as f:
            meta = json.load(f)
    
        for _frame in meta['frames']:
            width, height = int(_frame['w']), int(_frame['h'])
            fx, fy, cx, cy = int(_frame['fl_x']), int(_frame['fl_y']), int(_frame['cx']), int(_frame['cy'])                
            
            FoVy = focal2fov(fy, height)
            FoVx = focal2fov(fx, width)
        
            image_name = f"{_frame['file_path']}"
            image_path = self.image_path / image_name
            
            # Load image
            _image = cv2.imread(str(image_path))
            _image_tensor = torch.from_numpy(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)).float() / 255
            
            # Load or create mask
            mask_path = self.masks_dir / _frame['mask_path']
            if self.w_mask and mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                mask = cv2.resize(mask, (width, height))
                bg_mask = cv2.bitwise_not(mask)
                bg_image = cv2.bitwise_and(_image, _image, mask=bg_mask)
                bg_image_tensor = torch.from_numpy(cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)).float() / 255
                _mask_tensor = torch.from_numpy(mask) / 255
            else:
                if self.white_background:
                    bg_image_tensor = torch.ones((height, width, 3))
                else:
                    bg_image_tensor = torch.zeros((height, width, 3))
                _mask_tensor = None
            
            if 'depth_file_path' in _frame:
                depth_path = self.image_path / f"{_frame['depth_file_path']}"
                depth_tensor = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) / 1000
                depth_tensor = torch.from_numpy(depth_tensor).float()
            else:
                depth_tensor = None
            c2w = np.array(_frame['transform_matrix'])
            c2w[:,1:3] *= -1
            
            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, 
                                      image_name=image_name,
                                      image_path=image_path, 
                                      image_width=width, 
                                      image_height=height, 
                                      image=_image_tensor,
                                      bg_image=bg_image_tensor,
                                      mask=_mask_tensor,
                                      depth=depth_tensor,
                                      principal_point_ndc=np.array([cx / width, cy /height]))
            all_cameras_unsorted.append(_camera)
        self.all_cameras = sorted(all_cameras_unsorted, key=lambda x: x.image_name) 
        self.nerf_normalization = getNerfppNorm(self.all_cameras)
        self.cameras_extent = self.nerf_normalization["radius"]
        self.cameras_center = self.nerf_normalization["translate"]
        self.cameras_min_extent = self.nerf_normalization["min_radius"]

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

@datasets.register('mushroom')
class MuSHRoomDataset(Dataset, NerfStudioDatasetBase):
    transform_path = 'transformations_colmap.json'
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]