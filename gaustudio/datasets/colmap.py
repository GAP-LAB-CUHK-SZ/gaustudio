from gaustudio.datasets.utils import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
                                     read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, \
                                     focal2fov, getNerfppNorm, camera_to_JSON, storePly
from gaustudio import datasets

import os
import cv2
import sys
import json

import numpy as np
from PIL import Image


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depths_folder=None):
    cam_infos = []
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
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        
        if depths_folder is not None and os.path.exists(depths_folder):
            depth_path = os.path.join(depths_folder, os.path.basename(extr.name).split('.')[0]+'.png')
            
            depth = cv2.imread(depth_path, -1) / 1000.
            depth = cv2.resize(depth, (width, height))
            cam_info = datasets.CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height,
                                depth=depth)
        else:
            cam_info = datasets.CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

@datasets.register('colmap')
class ColmapDataset:
    def __init__(self, config):
        self.config = config        
        self.path = config.get('source_path', None)
        assert self.path is not None, "Please provide a source path"
        self.images = config.get('images', "images")
        self.eval = config.get('eval', False)
        self.llffhold = config.get('llffhold', 8)
        self.setup()
        
    def setup(self):
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

        reading_dir = self.images
        depth_dir = "depths"
        cam_infos_unsorted = readColmapCameras(cam_extrinsics, cam_intrinsics,  
                                               os.path.join(self.path, reading_dir), 
                                               os.path.join(self.path, depth_dir))
                                               
        cam_infos = sorted(cam_infos_unsorted, key=lambda x: x.image_name) 
        
        if self.eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % self.llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % self.llffhold == 0]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []
            
        nerf_normalization = getNerfppNorm(train_cam_infos)
        
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

        self.train_cameras = train_cam_infos
        self.test_cameras = test_cam_infos
        self.nerf_normalization = nerf_normalization
        self.cameras_extent = nerf_normalization["radius"]
        self.ply_path = ply_path
        
    def export(self, save_path):
        json_cams = []
        camlist = []
        camlist.extend(self.test_cameras)
        camlist.extend(self.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(save_path, 'w') as file:
            json.dump(json_cams, file)