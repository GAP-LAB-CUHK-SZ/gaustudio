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
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

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
        print("Loading NerfStudio transforms...")
        with open(self.source_path / self.transform_path, 'r') as f:
            meta = json.load(f)

        frames = meta['frames']
        print(f"Processing {len(frames)} NerfStudio frames...")

        def process_frame(frame_data):
            """Process a single frame with optimizations"""
            try:
                _frame = frame_data
                width, height = int(_frame['w']), int(_frame['h'])
                fx, fy, cx, cy = int(_frame['fl_x']), int(_frame['fl_y']), int(_frame['cx']), int(_frame['cy'])

                # Pre-compute FoV values
                FoVy = focal2fov(fy, height)
                FoVx = focal2fov(fx, width)

                image_name = f"{_frame['file_path']}"
                image_path = self.image_path / image_name

                # Optimized image loading with error checking
                _image = cv2.imread(str(image_path))
                if _image is None:
                    print(f"Warning: Could not load image {image_path}")
                    return None

                # Optimized color conversion and tensor creation
                _image_rgb = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
                _image_tensor = torch.from_numpy(_image_rgb.astype(np.float32) / 255.0)

                # Optimized mask processing
                mask_path = None
                if self.w_mask and 'mask_path' in _frame:
                    mask_path = self.masks_dir / _frame['mask_path']

                if mask_path is not None and mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                        if mask.shape[:2] != (height, width):
                            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

                        # Vectorized mask operations
                        bg_mask = 255 - mask
                        bg_image = cv2.bitwise_and(_image_rgb, _image_rgb, mask=bg_mask)
                        bg_image_tensor = torch.from_numpy(bg_image.astype(np.float32) / 255.0)
                        _mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0)
                    else:
                        bg_image_tensor = torch.ones((height, width, 3), dtype=torch.float32) if self.white_background else torch.zeros((height, width, 3), dtype=torch.float32)
                        _mask_tensor = None
                else:
                    bg_image_tensor = torch.ones((height, width, 3), dtype=torch.float32) if self.white_background else torch.zeros((height, width, 3), dtype=torch.float32)
                    _mask_tensor = None

                # Optimized depth loading
                depth_tensor = None
                if 'depth_file_path' in _frame:
                    depth_path = self.image_path / f"{_frame['depth_file_path']}"
                    if depth_path.exists():
                        depth_data = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
                        if depth_data is not None:
                            depth_tensor = torch.from_numpy(depth_data.astype(np.float32) / 1000.0)

                # Optimized transformation matrix processing
                c2w = np.array(_frame['transform_matrix'], dtype=np.float32)
                c2w[:, 1:3] *= -1  # Vectorized operation

                extrinsics = np.linalg.inv(c2w)
                R = extrinsics[:3, :3].T  # More efficient transpose
                T = extrinsics[:3, 3]

                return datasets.Camera(
                    R=R, T=T, FoVy=FoVy, FoVx=FoVx,
                    image_name=image_name, image_path=image_path,
                    image_width=width, image_height=height,
                    image=_image_tensor, bg_image=bg_image_tensor,
                    mask=_mask_tensor, depth=depth_tensor,
                    principal_point_ndc=np.array([cx / width, cy / height], dtype=np.float32)
                )
            except Exception as e:
                print(f"Error processing frame {_frame.get('file_path', 'unknown')}: {e}")
                return None

        # Parallel processing for better performance
        max_workers = min(4, len(frames))  # Limit workers for I/O operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_frame, frame) for frame in frames]
            all_cameras_unsorted = []

            for future in tqdm(futures, desc="Processing NerfStudio cameras"):
                camera = future.result()
                if camera is not None:
                    all_cameras_unsorted.append(camera)

        print(f"Successfully processed {len(all_cameras_unsorted)} cameras")
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
