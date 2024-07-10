from gaustudio import renderers
import torch
from gaustudio.utils.sh_utils import eval_sh
from gaustudio.renderers.gsplat_renderer import GsplatRenderer
@renderers.register('latentgs_renderer')
class LatentGSRenderer(GsplatRenderer):    
    def __init__(self, config) -> None:
        super().__init__(config)
        from diffusers import ConsistencyDecoderVAE, AutoencoderTiny
        # self.vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16).cuda()
        self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3").cuda()
        self.vae.enable_tiling()
    def get_gaussians_properties(self, viewpoint_camera, gaussian_model):
        xyz = gaussian_model.get_attribute("xyz")
        opacity = gaussian_model.get_attribute("opacity")
        scales = gaussian_model.get_attribute("scale")
        rotations = gaussian_model.get_attribute("rot")
        cov3D_precomp = None
        shs = None
        colors_precomp = gaussian_model.get_features
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

        rendered_image = rendered_image
        rendered_image = vae.decode(rendered_image.permute(2, 0, 1).unsqueeze(0)).sample[0]
        # rendered_image = vae.decode(rendered_image.half().permute(2, 0, 1).unsqueeze(0)/ 0.1825).sample[0]
        rendered_image = (rendered_image+1) /2
        rendered_image = rendered_image.clip(0, 1)
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": xys,
                "rendered_final_opacity": rendered_final_opacity.unsqueeze(0),
                "visibility_filter" : radii > 0,
                "radii": radii}
