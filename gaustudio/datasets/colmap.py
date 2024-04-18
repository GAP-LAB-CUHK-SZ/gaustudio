from gaustudio.datasets.utils import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
                                     read_extrinsics_binary, read_intrinsics_binary, \
                                     focal2fov, getNerfppNorm, camera_to_JSON, storePly
from gaustudio import datasets
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import sys
import json

import numpy as np
from PIL import Image
from typing import List, Dict 
from pathlib import Path
from tqdm import tqdm
import torch

class ColmapDatasetBase:
    # the data only has to be processed once
    def __init__(self, config: Dict):
        self._validate_config(config)
        self.path = Path(config['source_path'])
        self.images_dir = self.path / config.get('images', 'images')
        self.masks_dir = self.path / config.get('masks', 'masks')
        self.w_mask = config.get('w_mask', False)
        self.eval = config.get('eval', False)
        self.ply_path = config.get('ply_path', os.path.join(self.path, "sparse/0/points3D.ply"))
        self._initialize()
    
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def downsample_scale(self, resolution_scale):
        self.all_cameras = [c.downsample_scale(resolution_scale) for c in self.all_cameras]

    def _initialize(self):
        try:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        all_cameras_unsorted = []
        for key in tqdm(cam_extrinsics, total=len(cam_extrinsics), desc="Reading cameras"):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width

            uid = intr.id
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FoVy = focal2fov(focal_length_x, height)
                FoVx = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FoVy = focal2fov(focal_length_y, height)
                FoVx = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            image_path = self.images_dir / os.path.basename(extr.name)
            if not image_path.exists():
                continue
            
            mask_path_png = self.masks_dir / (os.path.basename(extr.name)[:-4] + '.png')
            mask_path_jpg = self.masks_dir / (os.path.basename(extr.name)[:-4] + '.jpg')
            if mask_path_png.exists():
                mask_path = mask_path_png
            elif mask_path_jpg.exists():
                mask_path = mask_path_jpg
            else:
                mask_path = None
                
            if self.w_mask and mask_path is not None:
                _image = cv2.imread(str(image_path))
                height, width, _ = _image.shape
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY) # Ensure mask is binary so multiply operation works as expected
                mask = cv2.resize(mask, (width, height)) # Resize the mask to match the size of the image
                background_mask = cv2.bitwise_not(mask) # Invert the mask to get the background
                black_background = np.full(_image.shape, 0, dtype=np.uint8) # Make the background black
                 # Combine the white background and the mask
                background = cv2.bitwise_and(black_background, black_background, mask=background_mask)
                masked_image = cv2.bitwise_and(_image, _image, mask=mask)
                _image = cv2.addWeighted(masked_image, 1, background, 1, 0)
                _image_tensor = torch.from_numpy(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)).float() / 255
                _mask_tensor = torch.from_numpy(mask)
                _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image_name=os.path.basename(extr.name), 
                                          image_width=width, image_height=height, image=_image_tensor, mask=_mask_tensor)
            else:
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

@datasets.register('colmap')
class ColmapDataset(Dataset, ColmapDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]
