import os
import json
import cv2
import numpy as np
from gaustudio import datasets
from gaustudio.datasets.base import BaseDataset
from gaustudio.datasets.utils import focal2fov
from typing import Dict
import math
import torch

class NerfStudioDatasetBase(BaseDataset):
    transform_path = 'transforms.json'
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.image_path = self.source_path
        self.masks_dir = self.source_path
        self.w_mask = config.get('w_mask', False)
        
        self._initialize()
    
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

        all_cameras_unsorted = self.process_in_parallel(
            frames,
            process_frame,
            desc="Processing NerfStudio cameras",
        )

        print(f"Successfully processed {len(all_cameras_unsorted)} cameras")
        self.finalize_cameras(all_cameras_unsorted)
            
@datasets.register('nerfstudio')
class NerfStudioDataset(NerfStudioDatasetBase):
    pass

@datasets.register('mushroom')
class MuSHRoomDataset(NerfStudioDatasetBase):
    transform_path = 'transformations_colmap.json'
    pass
