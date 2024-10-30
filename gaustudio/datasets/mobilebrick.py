import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from gaustudio import datasets
from gaustudio.datasets.utils import focal2fov, getNerfppNorm, camera_to_JSON
from typing import List, Dict 
from pathlib import Path
import torch

class MobileBrickDatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        
        self.image_dir = self.source_path / "image"
        self.mask_dir = self.source_path / "mask"
        self.pose_dir = self.source_path /  "pose"
        self.intrinsic_dir = self.source_path / 'intrinsic'
        self.image_filenames = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)],
                                      key=lambda fn: int(os.path.splitext(os.path.basename(fn))[0]))
        self.w_mask = config.get('w_mask', False)
        
        self._initialize()
        
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def _initialize(self):
        all_cameras_unsorted = []
        
        for image_path in self.image_filenames:
            image_name = os.path.basename(image_path)
            _id = os.path.splitext(image_name)[0]
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
            
            mask_path = self.mask_dir / f'{_id}.png'
            if self.w_mask and os.path.exists(mask_path):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY) # Ensure mask is binary so multiply operation works as expected
                mask = cv2.resize(mask, (width, height)) # Resize the mask to match the size of the image
                background_mask = cv2.bitwise_not(mask) # Invert the mask to get the background
                black_background = np.full(_image.shape, 0, dtype=np.uint8) # Make the background black

                # Combine the white background and the mask
                background = cv2.bitwise_and(black_background, black_background, mask=background_mask)
                masked_image = cv2.bitwise_and(_image, _image, mask=mask)
                _image = cv2.addWeighted(masked_image, 1, background, 1, 0)
                _mask_tensor = torch.from_numpy(mask)
            else:
                _mask_tensor = torch.ones((height, width), dtype=torch.uint8)
            _image_tensor = torch.from_numpy(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)).float() / 255
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image=_image_tensor, image_name=image_name, 
                                      image_width=width, image_height=height, mask=_mask_tensor,
                                      principal_point_ndc=np.array([cx / width, cy /height]))
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
            
@datasets.register('mobilebrick')
class MobileBrickDataset(Dataset, MobileBrickDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]