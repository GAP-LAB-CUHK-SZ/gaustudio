from gaustudio import renderers
import torch
import math
from gaustudio.utils.sh_utils import eval_sh

@renderers.register('surfel_renderer')
class SurfelRenderer:
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
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32) if self.white_background else torch.tensor([0, 0, 0], dtype=torch.float32)
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
    
    def render(self, viewpoint_camera, gaussian_model):
        '''
        Code adapted from https://github.com/hbb1/2d-gaussian-splatting/blob/main/gaussian_renderer/__init__.py
        '''
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

        from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color.to(viewpoint_camera.world_view_transform.device),
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
        rendered_image, radii, allmap = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
        
        # additional regularizations
        render_alpha = allmap[1:2]

        # get normal map
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
        
        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        return {"render": rendered_image,
                "rendered_depth": render_depth_expected,
                "rendered_median_depth": render_depth_median,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "rendered_final_opacity": render_alpha,
                "radii": radii}