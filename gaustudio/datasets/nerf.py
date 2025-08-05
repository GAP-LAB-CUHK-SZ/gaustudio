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
from tqdm import tqdm

class NerfDatasetBase:
    def __init__(self, config: Dict):
        self.source_path = Path(config['source_path'])
        self.image_path = Path(config['source_path'])
        
        self.split = config.get('split', 'train')
        self._initialize()
        self.ply_path = None
        
    def _validate_config(self, config: Dict):
        required_keys = ['source_path']
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")
    
    def _initialize(self):
        all_cameras_unsorted = []
        
        with open(self.source_path / f"transforms_{self.split}.json", 'r') as f:
            meta = json.load(f)
        
        if 'w' in meta and 'h' in meta:
            width, height = int(meta['w']), int(meta['h'])
        else:
            width, height = 800, 800
        
        focal = 0.5 * width / math.tan(0.5 * meta['camera_angle_x'])
        FoVy = focal2fov(focal, height)
        FoVx = focal2fov(focal, width) 
        
        for _frame in meta['frames']:
            image_name = f"{_frame['file_path']}.png"
            image_path = self.image_path / image_name
            
            c2w = np.array(_frame['transform_matrix'])
            c2w[:,1:3] *= -1
            
            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
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
            
@datasets.register('nerf')
class NerfDataset(Dataset, NerfDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)
    
    def __getitem__(self, index):
        return self.all_cameras[index]

def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img>limit, 1.055*img**(1/2.4)-0.055, 12.92*img)
    img[img>1] = 1 # "clamp" tonemapper
    return img

@datasets.register('rtmv')
class RTMVDataset(NerfDataset):
    transform_path = 'nerf_train.json'
    def _initialize(self):
        import os
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        all_cameras_unsorted = []
        
        split_json_path = self.source_path / f"transforms_{self.split}.json"
        if split_json_path.exists():
            with open(split_json_path, 'r') as f:
                meta = json.load(f)
        else:
            frames = []
            for _depth in self.source_path.glob('*.depth.exr'):
                _frame = {}
                _frame['file_path'] = str(_depth).split('.')[0]
                frames.append(_frame)
        meta = {'frames': frames}
        
        for _frame in tqdm(meta['frames']):
            image_name = f"{_frame['file_path']}.exr"
            image_path = self.image_path / image_name
            json_path = self.image_path / f"{_frame['file_path']}.json"
            mask_path = self.image_path / f"{_frame['file_path']}.seg.exr"
            depth_path = self.image_path / f"{_frame['file_path']}.depth.exr"
            
            # Load image
            _image = cv2.imread(str(image_path), -1)
            _image = linear_to_srgb(_image)
            _image_tensor = torch.from_numpy(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)).float()
            
            # Load mask
            _mask = cv2.imread(str(mask_path), -1)
            _mask = (_mask[..., 0] < 1e6).astype(float)
            _mask_tensor = torch.from_numpy(_mask)

            _camera_data = json.load(open(json_path, 'r'))['camera_data']
            _camera_intrinsics = _camera_data['intrinsics']
            width, height = _camera_data['width'], _camera_data['height']
            FoVy = focal2fov(_camera_intrinsics['fy'], height)
            FoVx = focal2fov(_camera_intrinsics['fx'], width) 
            cx, cy = _camera_intrinsics['cx'], _camera_intrinsics['cy']

            c2w = np.array(_camera_data['cam2world']).T            
            c2w[:,1:3] *= -1
            
            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            _camera = datasets.Camera(image_name=image_name, image_path=image_path, image=_image_tensor, 
                                      mask=_mask_tensor,
                                      R=R, T=T, principal_point_ndc=np.array([cx / width, cy /height]),
                                      FoVy=FoVy, FoVx=FoVx, image_width=width, image_height=height)
            
            # Load ndc depth
            _depth = cv2.imread(str(depth_path), -1)[..., 0]
            _depth_tensor = torch.from_numpy(_depth)
            mask_depth = torch.logical_and(_depth_tensor > -1000, _depth_tensor < 1000)
            _depth_tensor[~mask_depth] = 0
            _camera.depth = _camera.nerfdepth2depth(_depth_tensor)
            
            all_cameras_unsorted.append(_camera)
        self.all_cameras = sorted(all_cameras_unsorted, key=lambda x: x.image_name) 
        self.nerf_normalization = getNerfppNorm(self.all_cameras)
        self.cameras_extent = self.nerf_normalization["radius"]

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    return R
@datasets.register('navi')
class NAVIDataset(NerfDataset):
    def _initialize(self):
        all_cameras_unsorted = []
        self.image_dir = self.source_path / "images"
        
        self.annotations_path = self.source_path / 'annotations.json'
        
        # Load annotations
        with open(self.annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        for _id, anno in enumerate(tqdm(self.annotations)):            
            image_name = anno['filename']
            image_path = self.image_dir / image_name

            depth_path = self.source_path / "depth" / image_name.replace('.jpg', '.png')
            mask_path = self.source_path / "masks" / image_name.replace('.jpg', '.png')

            # Load mask
            _mask = cv2.imread(str(mask_path), -1) / 255
            _depth = cv2.imread(str(depth_path), -1) / 1000

            _mask_tensor = torch.from_numpy(_mask)
            _depth_tensor = torch.from_numpy(_depth)
            
            focal_length = anno['camera']['focal_length']
            width, height = anno['image_size']
            FoVy = focal2fov(focal_length, height)
            FoVx = focal2fov(focal_length, width)
            cx, cy = width / 2, height / 2
            
            q = np.array(anno['camera']['q'])
            T = np.array(anno['camera']['t'])
            R = quaternion_to_rotation_matrix(q)
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = T
            
            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]

            _camera = datasets.Camera(image_name=image_name, image_path=image_path,
                                      mask=_mask_tensor, depth=_depth_tensor, 
                                      R=R, T=T, principal_point_ndc=np.array([cx / width, cy /height]),
                                      FoVy=FoVy, FoVx=FoVx, image_width=width, image_height=height)
            
            all_cameras_unsorted.append(_camera)
        self.all_cameras = sorted(all_cameras_unsorted, key=lambda x: x.image_name) 
        self.nerf_normalization = getNerfppNorm(self.all_cameras)
        self.cameras_extent = self.nerf_normalization["radius"]
        
@datasets.register('kiri')
class KiriDataset(NerfDataset):
    def _initialize(self):
        all_cameras_unsorted = []
        
        with open(self.source_path / f"transforms.json", 'r') as f:
            meta = json.load(f)
        
        for _frame in tqdm(meta['frames']):
            image_name = _frame['file_path'].lstrip('./')
            image_path = self.source_path / image_name
            
            # Get intrinsics
            width, height = _frame['w'], _frame['h']
            fx, fy = _frame['fl_x'], _frame['fl_y']
            cx, cy = _frame['cx'], _frame['cy']
            
            FoVy = focal2fov(fy, height)
            FoVx = focal2fov(fx, width)
            
            # Get extrinsics
            c2w = np.array(_frame['transform_matrix'])
            c2w[:,1:3] *= -1
            
            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            # Load depth if available
            depth_tensor = None
            if 'depth_file_path' in _frame:
                depth_path = self.source_path / _frame['depth_file_path'].lstrip('./')
                if depth_path.exists():
                    _depth = cv2.imread(str(depth_path), -1) / 1000.0
                    depth_tensor = torch.from_numpy(_depth)
            
            _camera = datasets.Camera(
                image_name=image_name, 
                image_path=image_path,
                depth=depth_tensor,
                R=R, T=T, 
                principal_point_ndc=np.array([cx / width, cy / height]),
                FoVy=FoVy, FoVx=FoVx, 
                image_width=width, image_height=height
            )
            
            all_cameras_unsorted.append(_camera)
            
        self.all_cameras = sorted(all_cameras_unsorted, key=lambda x: x.image_name)
        self.nerf_normalization = getNerfppNorm(self.all_cameras)
        self.cameras_extent = self.nerf_normalization["radius"]