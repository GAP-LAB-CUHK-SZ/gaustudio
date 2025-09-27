import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from gaustudio import datasets
from gaustudio.datasets.utils import focal2fov, getNerfppNorm, camera_to_JSON, JSON_to_camera
from typing import List, Dict
from pathlib import Path
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class VanillaDatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        self.image_path = Path(config['source_path']) / "images"
        
        self._initialize()
        self.ply_path = None
        
    def _initialize(self):
        print("Loading vanilla dataset cameras...")
        with open(self.source_path / f"cameras.json", 'r') as f:
            camera_data = json.load(f)

        print(f"Processing {len(camera_data)} vanilla cameras...")

        def process_camera(camera_dict):
            """Process a single camera with optimizations"""
            try:
                _camera = JSON_to_camera(camera_dict, "cuda")
                _image_path = self.image_path / camera_dict['img_name']

                # Check if image exists before processing
                if not _image_path.exists():
                    print(f"Warning: Image not found: {_image_path}")
                    return None

                _camera.load_image(_image_path)
                return _camera
            except Exception as e:
                print(f"Error processing camera {camera_dict.get('img_name', 'unknown')}: {e}")
                return None

        # Parallel processing for better performance
        max_workers = min(4, len(camera_data))  # Limit workers for I/O operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_camera, cam_dict) for cam_dict in camera_data]
            all_cameras = []

            for future in tqdm(futures, desc="Processing vanilla cameras"):
                camera = future.result()
                if camera is not None:
                    all_cameras.append(camera)

        print(f"Successfully processed {len(all_cameras)} cameras")
        self.all_cameras = sorted(all_cameras, key=lambda x: x.image_name)
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
            
@datasets.register('vanilla')
class VanillaDataset(Dataset, VanillaDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]