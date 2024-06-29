from gaustudio import renderers
import torch
from gaustudio.utils.sh_utils import eval_sh

@renderers.register('gsplat_renderer')
class GsplatRenderer:
    default_conf = {
        'kernel_size': 0.,
        'white_background': False,
        'debug': False,
    }
    def __init__(self, config) -> None:
        super().__init__()
        self.config = {**self.default_conf, **config}
        self.kernel_size = self.config['kernel_size']
        self.white_background = self.config['white_background']
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32) if self.white_background else torch.tensor([0, 0, 0], dtype=torch.float32)
        self.debug = self.config['debug']
    
    def get_gaussians_properties(self, viewpoint_camera, gaussian_model):
        xyz = gaussian_model.get_attribute("xyz")
        opacity = gaussian_model.get_attribute("opacity")
        scales = gaussian_model.get_attribute("scale")
        rotations = gaussian_model.get_attribute("rot")
        cov3D_precomp = None
        shs = None
        shs_view = gaussian_model.get_features.transpose(1, 2).view(-1, 3, (gaussian_model.max_sh_degree+1)**2)
        dir_pp = (gaussian_model.get_attribute("xyz") - viewpoint_camera.camera_center.repeat(gaussian_model.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(gaussian_model.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return xyz, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp
    
    def render(self, viewpoint_camera, gaussian_model, glob_scale=1, block_width=16):
        '''
        Code adapted from https://github.com/hbb1/2d-gaussian-splatting/blob/main/gaussian_renderer/__init__.py
        '''
        xyz, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp = self.get_gaussians_properties(viewpoint_camera, gaussian_model)
        intrinsics = viewpoint_camera.intrinsics
        fx,fy,cx,cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
        img_height = int(viewpoint_camera.image_height)
        img_width = int(viewpoint_camera.image_width)
        
        import gsplat
        (xys, depths, radii, conics, compensation, num_tiles_hit, cov3ds) = (
            gsplat.project_gaussians(
                means3d=xyz,
                scales=scales,
                quats=rotations,
                viewmat=viewpoint_camera.extrinsics,
                img_height=img_height,
                img_width=img_width,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                glob_scale=glob_scale,
                block_width=block_width,
            )
        )

        # Attempt to keep position gradients to update densification stats
        if xys.requires_grad:
            xys.retain_grad()
            
        rendered_image, rendered_final_opacity = gsplat.rasterize_gaussians(
            xys=xys,
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=colors_precomp,
            opacity=opacity,
            img_height=img_height,
            img_width=img_width,
            block_width=block_width,
            return_alpha=True,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image.permute(2, 0, 1),
                "viewspace_points": xys,
                "rendered_final_opacity": rendered_final_opacity.unsqueeze(0),
                "visibility_filter" : radii > 0,
                "radii": radii}
