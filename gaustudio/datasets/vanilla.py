import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from gaustudio import datasets
from gaustudio.datasets.utils import focal2fov, getNerfppNorm, camera_to_JSON, JSON_to_camera
from typing import List, Dict 
from pathlib import Path
import math

class VanillaDatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        self.image_path = Path(config['source_path']) / "images"
        
        self._initialize()
        self.ply_path = None
        
    def _initialize(self):        
        with open(self.source_path / f"cameras.json", 'r') as f:
            camera_data = json.load(f)
            
        all_cameras = []
        for _camera_dict in camera_data:
            _camera = JSON_to_camera(_camera_dict, "cuda")
            _image_path = self.image_path / _camera_dict['img_name']
            _camera.load_image(_image_path)
            all_cameras.append(_camera)
        
        self.all_cameras = sorted(all_cameras, key=lambda x: x.image_name) 
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
            
@datasets.register('vanilla')
class VanillaDataset(Dataset, VanillaDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]