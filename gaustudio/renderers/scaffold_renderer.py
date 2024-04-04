from gaustudio.renderers.base import BaseRenderer
from gaustudio import renderers
import torch
import math
from gaustudio.utils.sh_utils import eval_sh
from gaustudio_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from einops import repeat
from gaustudio.models.utils import get_activation

@renderers.register('scaffold_renderer')
class ScaffoldRenderer(BaseRenderer):
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
    
    def prefilter_voxel(self, viewpoint_camera, gaussian_model):
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(gaussian_model.get_attribute("anchor"), dtype=gaussian_model.get_attribute("anchor").dtype, requires_grad=True, device="cuda") + 0
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
            kernel_size=self.kernel_size, #***new***
            bg=self.bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gaussian_model.get_attribute("anchor")


        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.compute_cov3D_python:
            cov3D_precomp = gaussian_model.get_covariance(self.scaling_modifier)
        else:
            scales = gaussian_model.get_attribute("scale")
            rotations = gaussian_model.get_attribute("rot")

        radii_pure = rasterizer.visible_filter(
            means3D = means3D,
            scales = scales[:,:3],
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        return radii_pure > 0

    def get_gaussians_properties(self, viewpoint_camera, gaussian_model):
        visible_mask = self.prefilter_voxel(viewpoint_camera, gaussian_model)
        if visible_mask is None:
            visible_mask = torch.ones(gaussian_model.get_attribute("anchor").shape[0], dtype=torch.bool, device = gaussian_model.get_attribute("anchor").device)
        feat = gaussian_model.get_attribute("anchor_feat")[visible_mask]
        anchor = gaussian_model.get_attribute("anchor")[visible_mask]
        grid_offsets = gaussian_model.get_attribute("offset")[visible_mask]
        grid_scaling = gaussian_model.get_attribute("scale")[visible_mask]

        ## get view properties for anchor
        ob_view = anchor - viewpoint_camera.camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        
        ## view-adaptive feature
        if gaussian_model.use_feat_bank:
            cat_view = torch.cat([ob_view, ob_dist], dim=1)
            
            bank_weight = gaussian_model.get_attribute("mlp_feature_bank")(cat_view).unsqueeze(dim=1) # [n, 1, 3]

            ## multi-resolution feat
            feat = feat.unsqueeze(dim=-1)
            feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
                feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
                feat[:,::1, :1]*bank_weight[:,:,2:]
            feat = feat.squeeze(dim=-1) # [n, c]


        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3]

        # get offset's opacity
        neural_opacity = gaussian_model.get_attribute("mlp_opacity")(cat_local_view) # [N, k]

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity>0.0)
        mask = mask.view(-1)

        # select opacity 
        opacity = neural_opacity[mask]

        # get offset's color
        color = gaussian_model.get_attribute("mlp_color")(cat_local_view)
        color = color.reshape([anchor.shape[0]*gaussian_model.n_offsets, 3])# [mask]

        # get offset's cov
        scale_rot = gaussian_model.get_attribute("mlp_cov")(cat_local_view)
        scale_rot = scale_rot.reshape([anchor.shape[0]*gaussian_model.n_offsets, 7]) # [mask]
        
        # offsets
        offsets = grid_offsets.view([-1, 3]) # [mask]
        
        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=gaussian_model.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
        
        # post-process cov
        scales = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
        rotations = get_activation(gaussian_model.config["activations"]['rot'])(scale_rot[:,3:7])

        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets
        shs = None
        cov3D_precomp = None
        return xyz, shs, color, opacity, scales, rotations, cov3D_precomp
    
    def render(self, viewpoint_camera, gaussian_model, subpixel_offset=None):
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

        if subpixel_offset is None:
            subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=self.kernel_size, #***new***
            subpixel_offset=subpixel_offset, #***new***
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
        rendered_image, radii = rasterizer(
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