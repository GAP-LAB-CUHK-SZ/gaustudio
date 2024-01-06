from gaustudio.datasets.utils import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
                                     read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, \
                                     focal2fov, getNerfppNorm, camera_to_JSON, storePly
from gaustudio import datasets
from torch.utils.data import Dataset, DataLoader, IterableDataset

import os
import cv2
import sys
import json

import numpy as np
from PIL import Image
from typing import List, Dict 
from pathlib import Path

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depths_folder=None):
    cameras = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

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

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image_path=image_path, image_width=width, image_height=height)
        cameras.append(_camera)
    sys.stdout.write('\n')
    return cameras

class ColmapDatasetBase:
    # the data only has to be processed once
    def __init__(self, config: Dict):
        self._validate_config(config)
        self.path = Path(config['source_path'])
        self.images_dir = config['images']
        self.eval = config.get('eval', False)

        self._initialize()
    
    def _validate_config(self, config: Dict):
        required_keys = ['source_path', 'images']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")

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

        reading_dir = self.images_dir
        depth_dir = "depths"
        all_cameras_unsorted = readColmapCameras(cam_extrinsics, cam_intrinsics,  
                                            os.path.join(self.path, reading_dir), 
                                            os.path.join(self.path, depth_dir))
                                            
        self.all_cameras = sorted(all_cameras_unsorted, key=lambda x: x.image_name) 

        ply_path = os.path.join(self.path, "sparse/0/points3D.ply")
        bin_path = os.path.join(self.path, "sparse/0/points3D.bin")
        txt_path = os.path.join(self.path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        
        self.ply_path = ply_path
        self.nerf_normalization = getNerfppNorm(self.all_cameras)
        self.cameras_extent = self.nerf_normalization["radius"]
        
class ColmapDataset(Dataset, ColmapDatasetBase):
    def __init__(self, config, resolution_scale):
        super().__init__(config)
        self.resolution_scale = resolution_scale
        if self.eval:
            train_cam_infos = [c.downsample(resolution_scale) for idx, c in enumerate(self.all_cameras) if idx % self.llffhold != 0]
            test_cam_infos = [c.downsample(resolution_scale) for idx, c in enumerate(self.all_cameras) if idx % self.llffhold == 0]
        else:
            train_cam_infos = self.all_cameras
            test_cam_infos = []
        self.train_cameras = train_cam_infos
        self.test_cameras = test_cam_infos
        
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }
    
    def export(self, save_path):
        json_cams = []
        camlist = []
        camlist.extend(self.test_cameras)
        camlist.extend(self.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(save_path, 'w') as file:
            json.dump(json_cams, file)

@datasets.register('colmap')
class ColmapDataloader:
    def __init__(self, 
                 config: Dict,
                 shuffle: bool = True,
                 resolution_scales: List[float] = [1.0]):
        
        self.datasets = {}
        for scale in resolution_scales:
            self.datasets[scale] = ColmapDataset(config, scale)
            
    def get_dataloader(self, dataset):
        return DataLoader(dataset, 
                          num_workers=os.cpu_count(), 
                          batch_size=1,
                          pin_memory=True)

    def get_train_dataloader(self, scale=1.0):
        return self.get_dataloader(self.datasets[scale].train_cameras)

    def get_test_dataloader(self, scale=1.0):
        return self.get_dataloader(self.datasets[scale].test_cameras)
