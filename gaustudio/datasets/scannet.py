import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from gaustudio import datasets
from gaustudio.datasets.utils import focal2fov, getNerfppNorm, camera_to_JSON
from typing import List, Dict 
from pathlib import Path
from tqdm import tqdm
import torch

class ScannetDatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        
        self.image_dir = self.source_path / "color"
        self.pose_dir = self.source_path /  "pose"
        self.depth_dir = self.source_path /  "depth"
        self.intrinsic_path = self.source_path / 'intrinsic' /  'intrinsic_color.txt'
        self.image_filenames = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)],
                                      key=lambda fn: int(os.path.splitext(os.path.basename(fn))[0]))
        self.image_filenames = self.image_filenames
        
        self._initialize()
        
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def _initialize(self):
        all_cameras_unsorted = []
        
        intr = np.loadtxt(self.intrinsic_path)
        for image_path in tqdm(self.image_filenames, total=len(self.image_filenames), desc="Reading cameras"):
            _id = int(os.path.splitext(os.path.basename(image_path))[0])
            _image = cv2.imread(image_path)
            _depth = cv2.imread(os.path.join(self.depth_dir, '%d.png' % _id), cv2.IMREAD_UNCHANGED) / 1000
            height, width, _ = _image.shape
            
            _image_tensor = torch.from_numpy(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)).float() / 255
            _depth_tensor = torch.from_numpy(_depth).float()

            fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]                
            c2w = np.loadtxt(self.pose_dir / ('%d.txt' % _id))

            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            FoVy = focal2fov(fy, height)
            FoVx = focal2fov(fx, width)
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, 
                                      image=_image_tensor, depth=_depth_tensor,
                                      image_name=os.path.basename(image_path),
                                      image_width=width, image_height=height,
                                      principal_point_ndc=np.array([cx / width, cy /height]))
            all_cameras_unsorted.append(_camera)
        self.all_cameras = sorted(all_cameras_unsorted, key=lambda x: int(
            os.path.splitext(os.path.basename(x.image_name))[0]))
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

@datasets.register('scannet')
class ScannetDataset(Dataset, ScannetDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]
