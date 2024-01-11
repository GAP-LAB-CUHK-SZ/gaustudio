from gaustudio.models.base import BaseRenderer
from gaustudio import models
import torch
import math
from gaustudio.utils.sh_utils import eval_sh

@models.register('vanilla_renderer')
class VanillaRenderer(BaseRenderer):
    default_conf = {
        'kernel_size': 0.,
        'scaling_modifier': 1.,
        'white_background': False,
        'convert_SHs_python': False,
        'compute_cov3D_python': False,
        'debug': False,
    }
    def __init__(self, config) -> None:
        super().__init__()
        self.config = {**self.default_conf, **config}
        self.kernel_size = self.config['kernel_size']
        self.scaling_modifier = self.config['scaling_modifier']
        self.white_background = self.config['white_background']
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.convert_SHs_python = self.config['convert_SHs_python']
        self.compute_cov3D_python = self.config['compute_cov3D_python']
        self.debug = self.config['debug']
    
    def get_gaussians_properties(self, viewpoint_camera, gaussian_model):
        xyz = gaussian_model.get_attribute("xyz")
        opacity = gaussian_model.get_attribute("opacity")
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.compute_cov3D_python:
            cov3D_precomp = gaussian_model.get_covariance(self.scaling_modifier)
        else:
            scales = gaussian_model.get_attribute("scale")
            rotations = gaussian_model.get_attribute("rot")
        shs = None
        colors_precomp = None
        if self.convert_SHs_python:
            shs_view = gaussian_model.get_features.transpose(1, 2).view(-1, 3, (gaussian_model.max_sh_degree+1)**2)
            dir_pp = (gaussian_model.get_attribute("xyz") - viewpoint_camera.camera_center.repeat(gaussian_model.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gaussian_model.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = gaussian_model.get_features
        return xyz, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp