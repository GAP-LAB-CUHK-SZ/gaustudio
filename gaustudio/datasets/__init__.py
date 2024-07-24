datasets = {}
import numpy as np
from PIL import Image
import dataclasses
import math
import os
import torch
import torch.nn.functional as F

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

def resizeDepthTorch(tensor_depth_image, resolution):
    if tensor_depth_image.ndim != 2:
        raise ValueError("Input tensor must have shape [H, W]")

    # Convert tensor to PIL image for resizing
    depth_image_np = tensor_depth_image.cpu().numpy()

    # Handle the conversion to a PIL image ensuring to maintain depth values as float32.
    pil_depth_image = Image.fromarray(depth_image_np.astype(np.float32))

    # Resize using the target resolution.
    resized_pil_depth_image = pil_depth_image.resize(resolution, Image.NEAREST)

    # Convert back to tensor, keeping depth values as they are.
    resized_depth_image_np = np.array(resized_pil_depth_image).astype(np.float32)
    resized_tensor_depth_image = torch.from_numpy(resized_depth_image_np)

    return resized_tensor_depth_image

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

def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic.t())
    return cam_xyz

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
    principal_point_ndc: np.array = np.array([0.5, 0.5])
    
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
    
    def update_intrinsics(self, intrinsics, image_width, image_height):
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        
        self.FoVx = 2.0 * np.arctan(image_width / (2.0 * fx))
        self.FoVy = 2.0 * np.arctan(image_height / (2.0 * fy))
        
        self.image_width = image_width
        self.image_height = image_height
    
    @property
    def fx(self):
        return self.intrinsics[0, 0]
    
    @property
    def fy(self):
        return self.intrinsics[1, 1]
    
    @property
    def extrinsics(self):
        return self.world_view_transform.transpose(0,1).contiguous() # cam2world
    
    @property
    def intrinsics(self):
        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        focal_y = self.image_height / (2.0 * tan_fovy)
        focal_x = self.image_width / (2.0 * tan_fovx)
        return torch.tensor([[focal_x, 0, self.image_width * self.principal_point_ndc[0]], 
                             [0, focal_y, self.image_height * self.principal_point_ndc[1]], 
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
    
    def depth2point(self, depth=None, coordinate='camera'):
        if depth is None:
            depth = self.depth
        if depth is None:
            raise ValueError("Depth is not available.")
        
        depth_height, depth_width = depth.shape

        valid_x = torch.arange(depth_width, dtype=torch.float32, device=depth.device) / (depth_width - 1)
        valid_y = torch.arange(depth_height, dtype=torch.float32, device=depth.device) / (depth_height - 1)
        valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
        ndc_xyz = torch.stack([valid_x, valid_y, depth], dim=-1)
        
        if coordinate == 'ndc':
            return ndc_xyz
        else:
            # cam_x = torch.arange(depth_width, dtype=torch.float32, device=depth.device)
            # cam_y = torch.arange(depth_height, dtype=torch.float32, device=depth.device)
            # cam_x, cam_y = torch.meshgrid(cam_y, cam_x)
            # cam_xy = torch.stack([cam_x, cam_y], dim=-1) * depth[..., None]
            # cam_xyz = torch.cat([cam_xy, depth[..., None]], axis=-1)         
            # cam_xyz = cam_xyz @ torch.inverse(self.intrinsics.to(depth.device).t())
            cam_xyz = ndc_2_cam(ndc_xyz[None, None, None, ...], self.intrinsics.to(depth.device), depth_width, depth_height)
        if coordinate == 'camera':
            return cam_xyz.reshape(*depth.shape, 3)
        elif coordinate == 'world':
            cam_xyz = cam_xyz.reshape(-1,3)
            world_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,0:1])], axis=-1) @ torch.inverse(self.extrinsics.to(depth.device)).transpose(0,1)
            world_xyz = world_xyz[..., :3]
            return world_xyz.reshape(*depth.shape, 3)
        else:
            raise ValueError("Invalid coordinate system.")
        return None

    # Adapted from https://github.com/baegwangbin/DSINE/blob/main/utils/d2n/cross.py
    def depth2normal(self, depth=None, k: int = 3, d_min: float = 1e-3, d_max: float = 10.0, coordinate='camera'):
        if depth is None:
            depth = self.depth
        if depth is None:
            raise ValueError("Depth is not available.")
        
        points = self.depth2point(depth, coordinate='camera')[None, ...]
        points = points.permute(0, 3, 1, 2)
        k = (k - 1) // 2

        B, _, H, W = points.size()
        points_pad = F.pad(points, (k,k,k,k), mode='constant', value=0)             # (B, 3, k+H+k, k+W+k)
        valid_pad = (points_pad[:,2:,:,:] > d_min) & (points_pad[:,2:,:,:] < d_max) # (B, 1, k+H+k, k+W+k)
        valid_pad = valid_pad.float()

        # vertical vector (top - bottom)
        vec_vert = points_pad[:, :, :H, k:k+W] - points_pad[:, :, 2*k:2*k+H, k:k+W]   # (B, 3, H, W)

        # horizontal vector (left - right)
        vec_hori = points_pad[:, :, k:k+H, :W] - points_pad[:, :, k:k+H, 2*k:2*k+W]   # (B, 3, H, W)

        # valid_mask (all five depth values - center/top/bottom/left/right should be valid)
        valid_mask = valid_pad[:, :, k:k+H,     k:k+W       ] * \
                    valid_pad[:, :, :H,        k:k+W       ] * \
                    valid_pad[:, :, 2*k:2*k+H, k:k+W       ] * \
                    valid_pad[:, :, k:k+H,     :W          ] * \
                    valid_pad[:, :, k:k+H,     2*k:2*k+W   ]
        valid_mask = valid_mask > 0.5
        
        # get cross product (B, 3, H, W)
        cross_product = - torch.linalg.cross(vec_vert, vec_hori, dim=1)
        normal = F.normalize(cross_product, p=2.0, dim=1, eps=1e-12)
        
        if coordinate == 'world':
            normal = normal.permute(0, 2, 3, 1) @ self.extrinsics[:3, :3].inverse().t()
            normal = normal.permute(0, 3, 1, 2)
        normal[~valid_mask.repeat(1, 3, 1, 1)] = -1
        
        return normal.squeeze(0).permute(1, 2, 0)

    def downsample_scale(self, scale):
        resolution = round(self.image_width/scale), round(self.image_height/scale)
        if self.image is not None:
            resized_image_rgb = resizeTorch(self.image, resolution)
            self.image = resized_image_rgb[..., :3].clamp(0.0, 1.0)
        if self.bg_image is not None:
            resized_bg_image_rgb = resizeTorch(self.bg_image, resolution)
            self.bg_image = resized_bg_image_rgb[..., :3].clamp(0.0, 1.0)
        if self.depth is not None:
            self.depth = resizeDepthTorch(self.depth, resolution)
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