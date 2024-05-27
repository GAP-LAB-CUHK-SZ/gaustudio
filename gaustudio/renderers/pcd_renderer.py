from gaustudio.renderers.base import BaseRenderer
from gaustudio import renderers
import torch

@renderers.register('pcd_renderer')
class PCDRenderer(BaseRenderer):
    default_conf = {
        'kernel_size': 0.,
        'scaling_modifier': 1.,
        'white_background': False,
        'debug': False,
        'convert_SHs_python': True
    }
    def __init__(self, config) -> None:
        super().__init__()
        self.config = {**self.default_conf, **config}
        self.kernel_size = self.config['kernel_size']
        self.scaling_modifier = self.config['scaling_modifier']
        self.white_background = self.config['white_background']
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32) if self.white_background else torch.tensor([0, 0, 0], dtype=torch.float32)
        self.debug = self.config['debug']
    
    def get_gaussians_properties(self, viewpoint_camera, gaussian_model):
        xyz = gaussian_model.get_attribute("xyz")
        opacity = torch.ones_like(xyz, device=xyz.device)
        cov3D_precomp = None
        scales = torch.ones_like(xyz, device=xyz.device) * self.kernel_size
        rotations = torch.zeros((xyz.shape[0], 4), device=xyz.device)
        rotations[:, 0] = 1
        rotations = torch.nn.functional.normalize(rotations)
        shs = None
        colors_precomp = gaussian_model._rgb / 255
        
        return xyz, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp