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

def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class NeusDatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        self.image_path = Path(config['source_path']) / "image"
        self.mask_path = Path(config['source_path']) / "mask"
        self.cams_path = Path(config['source_path']) / "cameras_sphere.npz"
        self.w_mask = config.get('w_mask', False)
        self._initialize()
        
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def _initialize(self):
        all_cameras_unsorted = []
        
        cams = np.load(self.cams_path)
        id_list = list(cams.keys())
        n_images = max([int(k.split('_')[-1]) for k in id_list]) + 1
        self.image_ids = range(n_images)
        
        for _id in self.image_ids:
            image_name = f'{_id:06d}.png'
            image_path = self.image_path / image_name
            _image = cv2.imread(str(image_path))
            height, width, _ = _image.shape
            mask_path = self.mask_path / f'{_id:03d}.png'
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

            world_mat, scale_mat = cams[f'world_mat_{_id}'], cams[f'scale_mat_{_id}']
            P = (world_mat @ scale_mat)[:3,:4]
            K, c2w = load_K_Rt_from_P(P)
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            FoVy = focal2fov(fy, height)
            FoVx = focal2fov(fx, width)
            _image_tensor = torch.from_numpy(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)).float() / 255
            _mask_tensor = torch.from_numpy(mask) if mask is not None else None
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image=_image_tensor, 
                                      image_name=image_name, image_width=width, image_height=height, 
                                      mask=_mask_tensor, principal_point_ndc=np.array([cx / width, cy /height]))
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

@datasets.register('neus')
class NeusDataset(Dataset, NeusDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]