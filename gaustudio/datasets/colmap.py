from gaustudio.datasets.utils import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
                                     read_extrinsics_binary, read_intrinsics_binary, \
                                     focal2fov, getNerfppNorm, camera_to_JSON, storePly
from gaustudio import datasets
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import json
import warnings
import numpy as np
from PIL import Image
from typing import List, Dict 
from pathlib import Path
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor

class ColmapDatasetBase:
    # the data only has to be processed once
    def __init__(self, config: Dict):
        self._validate_config(config)
        self.path = Path(config['source_path'])
        self.white_background = config.get('white_background', False)
        self.images_dir = self.path / config.get('images', 'images')
        self.sparse_dir = self.path / config.get('sparse', 'sparse')
        self.depths_dir = self.path / config.get('depths', 'depths')
        self.resolution = config.get('resolution', 1)
        
        if config.get('masks') is not None:
            self.masks_dir = self.path / config.get('masks')
            self.w_mask = True
        else:
            self.masks_dir = None
            self.w_mask = config.get('w_mask', False)
        self.eval = config.get('eval', False)
        self._initialize()
    
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def downsample_scale(self, resolution_scale):
        self.all_cameras = [c.downsample_scale(resolution_scale) for c in self.all_cameras]

    def _initialize(self):
        scene_dir = os.path.join(self.path, "sparse", "0")
        if not os.path.exists(scene_dir):
            scene_dir = os.path.join(self.path, self.sparse_dir)
        try:
            cameras_extrinsic_file = os.path.join(scene_dir, "images.bin")
            cameras_intrinsic_file = os.path.join(scene_dir, "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(scene_dir, "images.txt")
            cameras_intrinsic_file = os.path.join(scene_dir, "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        def process_camera(key):
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
                cx = intr.params[1]
                cy = intr.params[2]
            elif intr.model=="PINHOLE" or intr.model=="OPENCV":
                if intr.model == "OPENCV":
                    warnings.warn(
                        "OpenCV camera model detected. Distortion parameters will be discarded, which may degrade image quality. "
                        "It is recommended to run undistortion on your images before proceeding.",
                        UserWarning
                    )
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FoVy = focal2fov(focal_length_y, height)
                FoVx = focal2fov(focal_length_x, width)
                cx = intr.params[2]
                cy = intr.params[3]
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            image_path = self.images_dir / os.path.basename(extr.name)
            if not image_path.exists():
                return None

            # Optimized image loading - check format first
            _image = cv2.imread(str(image_path))
            if _image is None:
                return None
            height, width, _ = _image.shape
            
            # Optimized depth loading - use specific flags for better performance
            depth_path = self.depths_dir / (os.path.basename(extr.name)[:-4] + '.png')
            depth_tensor = None
            if depth_path.exists():
                depth_data = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
                if depth_data is not None:
                    depth_tensor = torch.from_numpy(depth_data.astype(np.float32) / 1000.0)
            
            if self.w_mask:
                mask_path_png = self.masks_dir / (os.path.basename(extr.name).split('.')[0] + '.png')
                mask_path_jpg = self.masks_dir / (os.path.basename(extr.name).split('.')[0] + '.jpg')
                if mask_path_png.exists():
                    mask_path = mask_path_png
                elif mask_path_jpg.exists():
                    mask_path = mask_path_jpg
                else:
                    mask_path = None
            else:
                mask_path = None
                
            # Optimized image and mask processing
            _image_rgb = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
            _image_tensor = torch.from_numpy(_image_rgb.astype(np.float32) / 255.0)

            if mask_path is not None:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Vectorized thresholding and resizing
                    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
                    if mask.shape[:2] != (height, width):
                        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

                    # Optimized mask operations
                    bg_mask = 255 - mask  # Faster than cv2.bitwise_not
                    bg_image = cv2.bitwise_and(_image_rgb, _image_rgb, mask=bg_mask)
                    _bg_image_tensor = torch.from_numpy(bg_image.astype(np.float32) / 255.0)
                    _mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0)
                else:
                    _mask_tensor = torch.ones((height, width), dtype=torch.float32)
                    _bg_image_tensor = torch.ones((height, width, 3), dtype=torch.float32) if self.white_background else torch.zeros((height, width, 3), dtype=torch.float32)
            else:
                _mask_tensor = torch.ones((height, width), dtype=torch.float32)
                _bg_image_tensor = torch.ones((height, width, 3), dtype=torch.float32) if self.white_background else torch.zeros((height, width, 3), dtype=torch.float32)
            
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image_name=os.path.basename(extr.name), image_path=image_path, 
                                          image_width=width, image_height=height, principal_point_ndc=np.array([cx / width, cy /height]), 
                                          image=_image_tensor, bg_image=_bg_image_tensor, mask=_mask_tensor, depth=depth_tensor)
            if self.resolution > 1:
                _camera.downsample_scale(self.resolution)
            return _camera

        # Optimize thread count for I/O bound operations
        max_workers = min(8, os.cpu_count())  # Limit to 8 threads for optimal I/O performance
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for key in cam_extrinsics:
                future = executor.submit(process_camera, key)
                futures.append(future)
            
            all_cameras_unsorted = []
            for future in tqdm(futures, desc="Reading cameras"):
                camera = future.result()
                if camera is not None:
                    all_cameras_unsorted.append(camera)

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

@datasets.register('colmap')
class ColmapDataset(Dataset, ColmapDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]
