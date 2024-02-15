datasets = {}
import numpy as np
from PIL import Image
import dataclasses
import math
import os
import torch

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

@dataclasses.dataclass
class Camera:

    R: np.ndarray
    T: np.ndarray 
    FoVx: float
    FoVy: float
    
    image_width: int
    image_height: int
    
    znear: float = 0.1
    zfar: float = 100
    
    trans: np.array = np.array([0.0, 0.0, 0.0])
    scale: float = 1.0

    world_view_transform: torch.tensor = None 
    full_proj_transform: torch.tensor = None

    image_path: str = None
    image_name: str = None
    image: np.array = None

    data_device: torch.device = None
    def __post_init__(self):
        self._setup()
        if self.data_device is not None:
            self.to_device(self.data_device)

    def _setup(self):
        if self.world_view_transform is None:
            self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1)
        if self.full_proj_transform is None:
            projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
            self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        
        if self.image_path is not None:
            self.image = torch.from_numpy(np.array(Image.open(self.image_path))) / 255.0
            self.image_name = os.path.basename(self.image_path).split(".")[0]
            self.image_height, self.image_width, _ = self.image.shape
            
        # Compute camera center from inverse view matrix
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
    
    def to_device(self, device):
        self.world_view_transform = self.world_view_transform.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        self.data_device = device
        
    @property
    def extrinsics(self):
        return self.world_view_transform.transpose(0,1).contiguous() # cam2world
    
    @property
    def intrinsics(self):
        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        focal_y = self.image_height / (2.0 * tan_fovy)
        focal_x = self.image_width / (2.0 * tan_fovx)
        return torch.tensor([[focal_x, 0, self.image_width / 2], 
                             [0, focal_y, self.image_height / 2], 
                             [0, 0, 1]]).float()
     
    @extrinsics.setter
    def extrinsics(self, value):
        """Sets the extrinsic parameters of the camera"""

        # The rotation matrix R is the top left 3x3 block of the extrinsics
        R = extrinsics[:3, :3]
        
        # The translation vector T is the top right 3x1 block of the extrinsics
        T = extrinsics[:3, 3]

        self.R = R
        self.T = T

        # Update the world view transform and full projection transform after setting new extrinsics
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def downsample(self, resolution):
        if self.image is not None:
            resized_image_rgb = PILtoTorch(self.image, resolution)
            
            gt_image = resized_image_rgb[:3, ...]
            self.image = gt_image.clamp(0.0, 1.0)
            self.image_height, self.image_width = gt_image.shape[1:3]
        else:
            self.image_height, self.image_width = resolution
    
    def downsample_scale(self, scale):
        resolution = round(self.image_width/scale), round(self.image_height/scale)
        if self.image is not None:
            resized_image_rgb = PILtoTorch(self.image, resolution)
            
            gt_image = resized_image_rgb[:3, ...]
            self.image = gt_image.clamp(0.0, 1.0)
            self.image_height, self.image_width = gt_image.shape[1:3]
        else:
            self.image_width, self.image_height = resolution

def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, config):
    dataset = datasets[name](config)
    return dataset


from . import colmap