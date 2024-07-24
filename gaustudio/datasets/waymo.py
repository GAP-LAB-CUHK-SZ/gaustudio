from gaustudio import datasets
from gaustudio.datasets.utils import focal2fov, getNerfppNorm
from torch.utils.data import Dataset, DataLoader
import tempfile

import os
import cv2
import sys
import json

import numpy as np
from PIL import Image
from typing import List, Dict 
from pathlib import Path
import pickle

cameras = ['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT', 'camera_SIDE_LEFT', 'camera_SIDE_RIGHT']

def generate_sample(scenario_data, frame_idx, camera_id):
    print(scenario_data['observers'].keys())
    camera_data = scenario_data['observers'][camera_id]
    n_frames = camera_data['n_frames']
    if frame_idx >= n_frames:
        raise ValueError(f"Frame index {frame_idx} out of range for camera {camera_id} with {n_frames} frames.")

    # Load image
    image_path = f"images/{camera_id}/{frame_idx:08d}.jpg"
    image = np.array(Image.open(image_path))

    # Get frame metadata
    frame_json = camera_data['data']
    for key in ['hw', 'intr', 'c2w', 'distortion']:
        frame_json[key] = frame_json[key][frame_idx]

    width, height = frame_json['hw']
    fx, fy, cx, cy = frame_json['intr'][0, 0], frame_json['intr'][1, 1], frame_json['intr'][0, 2], frame_json['intr'][1, 2]

    intrinsic_4x4 = np.array([[fx, 0, cx, 0],
                              [0, fy, cy, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

    intrinsics = {'width': width, 'height': height, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

    c2w = frame_json['c2w']

    sample = {"color": image,
              "c2w": c2w,
              "intrinsic_dict": intrinsics,
              "intrinsic_4x4": intrinsic_4x4}

    return sample

class WaymoDatasetBase:
    # the data only has to be processed once
    def __init__(self, config: Dict):
        self._validate_config(config)
        self.path = Path(config['source_path'])
        scenario_path = self.path / "scenario.pt"
        self.camera_number = config.get('camera_number', 1)
        self.camera_ids = cameras[:self.camera_number]
        self.eval = config.get('eval', False)
        
        with open(scenario_path, 'rb') as f:
            scenario_data = pickle.load(f)
        self._initialize(scenario_data)

    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def downsample_scale(self, resolution_scale):
        self.all_cameras = [c.downsample_scale(resolution_scale) for c in self.all_cameras]

    def _initialize(self, scenario_data):
        all_cameras_unsorted = []
        for _camera_id in self.camera_ids:
            camera_data = scenario_data['observers'][_camera_id]
            n_frames = camera_data['n_frames']

            # Get frame metadata
            frame_json = camera_data['data']
            for frame_idx in range(n_frames):
                image_path = self.path / "images" / f"{_camera_id}/{frame_idx:08d}.jpg"
                height, width = frame_json['hw'][frame_idx]
                intr = frame_json['intr'][frame_idx]
                fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]                
                
                distortion_coeffs = frame_json['distortion'][frame_idx]
                temp_dir = Path(tempfile.gettempdir())
                temp_image_path = temp_dir / f"{frame_idx:08d}_{_camera_id}.jpg"
                img = cv2.imread(str(image_path))
                undistort_intr, roi = cv2.getOptimalNewCameraMatrix(intr, distortion_coeffs, (width, height), 0)
                undistorted_img = cv2.undistort(img, intr, distortion_coeffs, None, undistort_intr)
                cv2.imwrite(str(temp_image_path), undistorted_img)
                fx, fy, cx, cy = undistort_intr[0, 0], undistort_intr[1, 1], undistort_intr[0, 2], undistort_intr[1, 2]               

                c2w = frame_json['c2w'][frame_idx]
                extrinsics = np.linalg.inv(c2w)
                R = np.transpose(extrinsics[:3, :3])
                T = extrinsics[:3, 3]
                
                FoVy = focal2fov(fy, height)
                FoVx = focal2fov(fx, width)
                _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image_path=temp_image_path, 
                                          image_width=width, image_height=height,
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

@datasets.register('waymo')
class WaymoDataset(Dataset, WaymoDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]