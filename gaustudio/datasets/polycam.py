import os
import json
import cv2
import numpy as np
from gaustudio import datasets
from gaustudio.datasets.base import BaseDataset
from gaustudio.datasets.utils import focal2fov
from typing import Dict

class PolycamDatasetBase(BaseDataset):
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.image_dir = self.source_path / "keyframes" / "corrected_images"
        self.cameras_dir = self.source_path /  "keyframes" / "corrected_cameras"
        self.image_filenames = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)],
                                      key=lambda fn: int(os.path.splitext(os.path.basename(fn))[0]))

        self._initialize()
    
    def _initialize(self):
        print(f"Processing {len(self.image_filenames)} Polycam images...")

        def process_image(image_path):
            """Process a single image and its camera parameters"""
            try:
                ori_frame_id = int(os.path.splitext(os.path.basename(image_path))[0])
                json_filename = os.path.join(self.cameras_dir, f"{ori_frame_id}.json")

                with open(json_filename, 'r') as f:
                    frame_json = json.load(f)

                width, height = frame_json["width"], frame_json["height"]
                fx, fy, cx, cy = frame_json['fx'], frame_json['fy'], frame_json['cx'], frame_json['cy']

                # Optimized transformation matrix construction
                c2w = np.array([
                    [frame_json["t_20"], frame_json["t_21"], frame_json["t_22"], frame_json["t_23"]],
                    [frame_json["t_00"], frame_json["t_01"], frame_json["t_02"], frame_json["t_03"]],
                    [frame_json["t_10"], frame_json["t_11"], frame_json["t_12"], frame_json["t_13"]],
                    [0, 0, 0, 1],
                ], dtype=np.float32)
                c2w[:, [1, 2]] *= -1  # Vectorized operation

                extrinsics = np.linalg.inv(c2w)
                R = extrinsics[:3, :3].T  # More efficient transpose
                T = extrinsics[:3, 3]

                # Pre-compute FoV values
                FoVy = focal2fov(fy, height)
                FoVx = focal2fov(fx, width)

                return datasets.Camera(
                    R=R, T=T, FoVy=FoVy, FoVx=FoVx, image_path=image_path,
                    image_width=width, image_height=height,
                    principal_point_ndc=np.array([cx / width, cy / height], dtype=np.float32)
                )
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                return None

        all_cameras_unsorted = self.process_in_parallel(
            self.image_filenames,
            process_image,
            desc="Processing Polycam cameras",
        )

        print(f"Successfully processed {len(all_cameras_unsorted)} cameras")
        self.finalize_cameras(all_cameras_unsorted)
    
@datasets.register('polycam')
class PolycamDataset(PolycamDatasetBase):
    pass
