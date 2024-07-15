datasets = {}
import numpy as np
from PIL import Image
import dataclasses
import math
import os
import torch

def resizeTorch(tensor_image, resolution):
    if tensor_image.ndim != 3 or tensor_image.shape[2] != 3:
        raise ValueError("Input tensor must have shape [H, W, 3]")

    # Normalize to PIL range [0, 255] and convert to uint8 type if necessary
    if tensor_image.max() <= 1.0:
        tensor_image = tensor_image * 255.0
    tensor_image = tensor_image.byte()

    # Convert tensor to PIL image for resizing
    pil_image = Image.fromarray(tensor_image.cpu().numpy())

    # Resize using the target resolution
    resized_pil_image = pil_image.resize(resolution)

    # Convert back to tensor and normalize to [0, 1]
    resized_tensor_image = torch.from_numpy(np.array(resized_pil_image)).float() / 255.0
    return resized_tensor_image

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

# Adopted from https://github.com/hbb1/2d-gaussian-splatting/pull/74
def getProjectionMatrix(znear, zfar, fovX, fovY, width, height, principal_point_ndc=None):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    if principal_point_ndc is not None:
        # shift the frame window due to the non-zero principle point offsets
        cx = width * principal_point_ndc[0]
        cy = height * principal_point_ndc[1]
        tan_fovx = np.tan(fovX / 2.0)
        tan_fovy = np.tan(fovY / 2.0)
        focal_x = width / (2.0 * tan_fovx)
        focal_y = height / (2.0 * tan_fovy)
        offset_x = cx - (width / 2)
        offset_x = (offset_x / focal_x) * znear
        offset_y = cy - (height / 2)
        offset_y = (offset_y / focal_y) * znear

        top = top + offset_y
        left = left + offset_x
        right = right + offset_x
        bottom = bottom + offset_y

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
    camera_center: torch.tensor = None
    principal_point_ndc: np.array = None
    
    image_path: str = None
    image_name: str = None
    image: np.array = None
    bg_image: np.array = None
    mask: np.array = None
    normal: np.array = None
    depth: np.array = None
    
    def __post_init__(self):
        self._setup()

    def _setup(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, 
                                                     fovX=self.FoVx, fovY=self.FoVy,
                                                     width=self.image_width, height=self.image_height,
                                                     principal_point_ndc=self.principal_point_ndc).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        
        if self.image_path is not None:
            self.image = torch.from_numpy(np.array(Image.open(self.image_path).convert("RGB"))) / 255.0
            self.image_name = os.path.basename(self.image_path).split(".")[0]
            self.image_height, self.image_width, _ = self.image.shape

        # Compute camera center from inverse view matrix
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
    
    def __repr__(self):
        return f"Camera(FoVx={self.FoVx:.2f}, FoVy={self.FoVy:.2f}, image_width={self.image_width}, image_height={self.image_height}, znear={self.znear}, zfar={self.zfar})"
        
    def to(self, device: torch.device):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name) 
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.to(device))
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                setattr(self, field.name, [v.to(device) for v in value])

        return self
    
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
    def extrinsics(self, extrinsics):
        """Sets the extrinsic parameters of the camera"""
        self.R = np.transpose(extrinsics[:3, :3])
        self.T = extrinsics[:3, 3]
        self._setup()

    def downsample(self, resolution):
        if self.image is not None:
            resized_image_rgb = resizeTorch(self.image, resolution)
            
            gt_image = resized_image_rgb[..., :3]
            self.image = gt_image.clamp(0.0, 1.0)
            self.image_height, self.image_width, _ = gt_image.shape
            # TODO: Add mask, normal, depth resize, modify principle point
        else:
            self.image_height, self.image_width = resolution
        return self
    
    def downsample_scale(self, scale):
        resolution = round(self.image_width/scale), round(self.image_height/scale)
        if self.image is not None:
            resized_image_rgb = resizeTorch(self.image, resolution)
            
            gt_image = resized_image_rgb[..., :3]
            self.image = gt_image.clamp(0.0, 1.0)
            self.image_height, self.image_width, _ = gt_image.shape
        else:
            self.image_width, self.image_height = resolution
        return self

def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(config):
    if isinstance(config, str):
        name = config
        config = {}
    else:
        name = config.get('name')

    if name not in datasets:
        raise ValueError(f'Unknown dataset: {name}')

    dataset = datasets[name](config)
    return dataset


from . import colmap, waymo, polycam, scannet, mvsnet, nerf, \
                nsvf, deepvoxels, nero, mobilebrick, neus, \
                nerfstudio