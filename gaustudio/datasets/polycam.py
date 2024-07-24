import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from gaustudio import datasets
from gaustudio.datasets.utils import focal2fov, getNerfppNorm
from typing import List, Dict 
from pathlib import Path

class PolycamDatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        
        self.image_dir = self.source_path / "keyframes" / "images"
        self.cameras_dir = self.source_path /  "keyframes" / "cameras"
        self.image_filenames = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)],
                                      key=lambda fn: int(os.path.splitext(os.path.basename(fn))[0]))

        self._initialize()
        
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def _initialize(self):
        all_cameras_unsorted = []
        
        for image_path in self.image_filenames:
            ori_frame_id = int(os.path.splitext(os.path.basename(image_path))[0])
            
            json_filename = os.path.join(self.cameras_dir, "{}.json".format(ori_frame_id))
            frame_json = json.load(open(json_filename))
            width, height = frame_json["width"], frame_json["height"]
            
            intr = np.array([[frame_json['fx'], 0, frame_json['cx']],
                                [0, frame_json['fy'], frame_json['cy']],
                                [0, 0, 1]])
            fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]                
            
            c2w = np.array([
                [frame_json["t_20"], frame_json["t_21"], frame_json["t_22"], frame_json["t_23"]],
                [frame_json["t_00"], frame_json["t_01"], frame_json["t_02"], frame_json["t_03"]],
                [frame_json["t_10"], frame_json["t_11"], frame_json["t_12"], frame_json["t_13"]],
                [0, 0, 0, 1],
            ], dtype=np.float32)
            c2w[..., 2] *= -1
            c2w[..., 1] *= -1
            
            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            FoVy = focal2fov(fy, height)
            FoVx = focal2fov(fx, width)
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image_path=image_path, 
                                      image_width=width, image_height=height,
                                      principal_point_ndc=np.array([cx / width, cy /height]))
            all_cameras_unsorted.append(_camera)
        self.all_cameras = sorted(all_cameras_unsorted, key=lambda x: x.image_name) 
        self.nerf_normalization = getNerfppNorm(self.all_cameras)
        self.cameras_extent = self.nerf_normalization["radius"]
    
@datasets.register('polycam')
class PolycamDataset(Dataset, PolycamDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]