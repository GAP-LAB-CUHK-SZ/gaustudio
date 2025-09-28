import os
import json
import cv2
import numpy as np
from gaustudio import datasets
from gaustudio.datasets.base import BaseDataset
from gaustudio.datasets.utils import focal2fov
from gaustudio.datasets.optimization_utils import OptimizedImageLoader
from typing import Dict
import torch

class ScannetDatasetBase(BaseDataset):
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.image_dir = self.source_path / "color"
        self.pose_dir = self.source_path /  "pose"
        self.depth_dir = self.source_path /  "depth"
        self.intrinsic_path = self.source_path / 'intrinsic' /  'intrinsic_color.txt'
        self.image_filenames = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)],
                                      key=lambda fn: int(os.path.splitext(os.path.basename(fn))[0]))
        self.image_filenames = self.image_filenames
        
        self._initialize()
    
    def _initialize(self):
        print(f"Loading ScanNet dataset with {len(self.image_filenames)} images...")
        intr = np.loadtxt(self.intrinsic_path)
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]

        def process_scannet_frame(image_path):
            """Process a single ScanNet frame with optimizations"""
            try:
                _id = int(os.path.splitext(os.path.basename(image_path))[0])

                # Optimized image loading
                _image_tensor = OptimizedImageLoader.load_image_optimized(image_path)
                if _image_tensor is None:
                    return None

                height, width = _image_tensor.shape[:2]

                # Optimized depth loading
                depth_path = os.path.join(self.depth_dir, f'{_id}.png')
                _depth_tensor = OptimizedImageLoader.load_depth_optimized(depth_path, scale_factor=1000.0)

                # Load pose with error handling
                pose_path = self.pose_dir / f'{_id}.txt'
                if not pose_path.exists():
                    return None

                c2w = np.loadtxt(pose_path)
                extrinsics = np.linalg.inv(c2w)
                R = extrinsics[:3, :3].T  # More efficient transpose
                T = extrinsics[:3, 3]

                # Pre-compute FoV values
                FoVy = focal2fov(fy, height)
                FoVx = focal2fov(fx, width)

                return datasets.Camera(
                    R=R, T=T, FoVy=FoVy, FoVx=FoVx,
                    image=_image_tensor, depth=_depth_tensor,
                    image_name=os.path.basename(image_path),
                    image_width=width, image_height=height,
                    principal_point_ndc=np.array([cx / width, cy / height], dtype=np.float32)
                )
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                return None

        # Use parallel processing
        all_cameras_unsorted = self.process_in_parallel(
            self.image_filenames,
            process_scannet_frame,
            desc="Processing ScanNet cameras",
        )

        print(f"Successfully processed {len(all_cameras_unsorted)} ScanNet cameras")
        sort_key = lambda cam: int(os.path.splitext(os.path.basename(cam.image_name))[0])
        self.finalize_cameras(all_cameras_unsorted, sort_key=sort_key)

@datasets.register('scannet')
class ScannetDataset(ScannetDatasetBase):
    pass
