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

def load_from_log(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    image_ids = []
    intrinsics = []
    extrinsics = []

    for line in range(0, len(content), 7):
        image_ids.append(int(content[line]))

        intrinsics.append([float(value) for value in content[line + 1].split()])
        extrinsics.append([[float(value) for value in content[line + 3].split()], \
                           [float(value) for value in content[line + 4].split()], \
                           [float(value) for value in content[line + 5].split()], \
                           [float(value) for value in content[line + 6].split()]])
    return image_ids, intrinsics, extrinsics

class NisrDatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        self.image_path = Path(config['source_path']) / "images"
        self.mask_path = Path(config['source_path']) / "mask"
        self.cams_path = Path(config['source_path']) / "camera.log"
        self.w_mask = config.get('w_mask', False)
        self._initialize()
        self.ply_path = None
        
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def _initialize(self):
        all_cameras = []

        image_ids, intrinsics, extrinsics = load_from_log(self.cams_path)
        for _id, K, c2w in zip(image_ids, intrinsics, extrinsics):
            image_name = f'{_id}.png'
            image_path = self.image_path / image_name
            _image = cv2.imread(str(image_path))
            height, width, _ = _image.shape
            mask_path = self.mask_path / f'{_id}.png'
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
            else:
                mask = None

            fx, fy, cx, cy = K[0], K[1], K[2], K[3]

            c2w = np.array(c2w)
            # c2w[:,1:3] *= -1            
            
            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            FoVy = focal2fov(fy, height)
            FoVx = focal2fov(fx, width)

            _image_tensor = torch.from_numpy(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)).float() / 255
            _mask_tensor = torch.from_numpy(mask) if mask is not None else None
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image=_image_tensor, image_name=image_name, 
                                      image_width=width, image_height=height, mask=_mask_tensor,
                                      principal_point_ndc=np.array([cx / width, cy /height]))
            all_cameras.append(_camera)
            if _id == 20:
                break
        self.all_cameras = all_cameras
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

@datasets.register('nisr')
class NisrDataset(Dataset, NisrDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]