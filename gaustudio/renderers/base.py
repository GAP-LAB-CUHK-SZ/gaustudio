import numpy as np
from typing import Dict
from plyfile import PlyData, PlyElement
import torch.nn as nn
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class BaseRenderer:
    def render(self, viewpoint_camera, gaussian_model):
        xyz, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp = self.get_gaussians_properties(viewpoint_camera, gaussian_model)
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=gaussian_model.active_sh_degree if shs != None else 1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, rendered_depth, rendered_median_depth, rendered_final_opacity = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}