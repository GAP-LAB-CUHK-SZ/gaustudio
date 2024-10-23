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
from tqdm import tqdm

class NerfDatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        self.image_path = Path(config['source_path'])
        
        self.split = config.get('split', 'train')
        self._initialize()
        self.ply_path = None
        
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def _initialize(self):
        all_cameras_unsorted = []
        
        with open(self.source_path / f"transforms_{self.split}.json", 'r') as f:
            meta = json.load(f)
        
        if 'w' in meta and 'h' in meta:
            width, height = int(meta['w']), int(meta['h'])
        else:
            width, height = 800, 800
        
        focal = 0.5 * width / math.tan(0.5 * meta['camera_angle_x'])
        FoVy = focal2fov(focal, height)
        FoVx = focal2fov(focal, width) 
        
        for _frame in meta['frames']:
            image_name = f"{_frame['file_path']}.png"
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
            
@datasets.register('nerf')
class NerfDataset(Dataset, NerfDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]
    
def load_camera_parameters(data: Dict):
    camera_data = data['camera_data']
    
    # Get camera-to-world matrix
    c2w = np.array(camera_data['cam2world'])
    
    # Get focal lengths
    fx = camera_data['intrinsics']['fx']
    fy = camera_data['intrinsics']['fy']
    cx = camera_data['intrinsics']['cx']
    cy = camera_data['intrinsics']['cy']
    
    return c2w, fx, fy

@datasets.register('rtmv')
class RTMVDataset(NerfDataset):
    transform_path = 'nerf_train.json'
    def _initialize(self):
        import os
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        all_cameras_unsorted = []
        
        with open(self.source_path / f"nerf_{self.split}.json", 'r') as f:
            meta = json.load(f)
        
        for _frame in tqdm(meta['frames']):
            image_name = f"{_frame['file_path']}.exr"
            image_path = self.image_path / image_name
            json_path = self.image_path / f"{_frame['file_path']}.json"
            mask_path = self.image_path / f"{_frame['file_path']}.seg.exr"
            
            # Load image
            _image = cv2.imread(str(image_path), -1)
            _image_tensor = torch.from_numpy(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)).float()
            _image_tensor = _image_tensor.clip(0,1)**(1/2.2)
            
            # Load mask
            _mask = cv2.imread(str(mask_path), -1)
            _mask = (_mask[..., 0] < 1e6).astype(float)
            _mask_tensor = torch.from_numpy(_mask)

            _camera_data = json.load(open(json_path, 'r'))['camera_data']
            _camera_intrinsics = _camera_data['intrinsics']
            width, height = _camera_data['width'], _camera_data['height']
            FoVy = focal2fov(_camera_intrinsics['fy'], height)
            FoVx = focal2fov(_camera_intrinsics['fx'], width) 

            c2w = np.array(_frame['transform_matrix'])
            c2w[:,1:3] *= -1
            
            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            _camera = datasets.Camera(image_name=image_name, image_path=image_path, image=_image_tensor, 
                                      mask=_mask_tensor,
                                      R=R, T=T, 
                                      FoVy=FoVy, FoVx=FoVx, image_width=width, image_height=height)
            all_cameras_unsorted.append(_camera)
        self.all_cameras = sorted(all_cameras_unsorted, key=lambda x: x.image_name) 
        self.nerf_normalization = getNerfppNorm(self.all_cameras)
        self.cameras_extent = self.nerf_normalization["radius"]