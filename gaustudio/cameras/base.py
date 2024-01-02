import torch

class BaseCamera:
    def __init__(self, width, height, fovx, fovy, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVx = fovx
        self.FoVy = fovy 
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        
        # Compute camera center from inverse view matrix
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]