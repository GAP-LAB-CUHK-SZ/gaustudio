import sys
import argparse
import os
import time
import logging
from datetime import datetime
import torch
import json
from pathlib import Path
import cv2
import torchvision
from tqdm import tqdm
import open3d as o3d
import numpy as np

import pytorch3d
from pytorch3d.io import load_ply, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRendererWithFragments, 
    MeshRasterizer,  
    SoftPhongShader,
    hard_rgb_blend,
    TexturesAtlas,
)

from pytorch3d.renderer.mesh.shader import ShaderBase
class VertexColorShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        return hard_rgb_blend(texels, fragments, blend_params)

def np_depth_to_colormap(depth):
    """ depth: [H, W] """
    depth_normalized = np.zeros(depth.shape)

    valid_mask = depth > -0.9 # valid
    if valid_mask.sum() > 0:
        d_valid = depth[valid_mask]
        depth_normalized[valid_mask] = (d_valid - d_valid.min()) / (d_valid.max() - d_valid.min())

        depth_np = (depth_normalized * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
        depth_normalized = depth_normalized
    else:
        print('!!!! No depth projected !!!')
        depth_color = depth_normalized = np.zeros(depth.shape, dtype=np.uint8)
    return depth_color, depth_normalized


def get_normals_from_fragments(meshes, fragments):
    """ z """
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]
    ones = torch.ones_like(fragments.bary_coords, device="cuda")
    pixel_normals = pytorch3d.ops.interpolate_face_attributes(
        fragments.pix_to_face, ones, faces_normals
    )
    return pixel_normals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--camera', '-c', default=None, help='path to cameras.json')
    parser.add_argument('--mesh', '-m', default=None, help='path to the mesh')
    parser.add_argument('--source_path', '-s', help='path to the dataset')
    parser.add_argument('--output-dir', '-o',  help='path to the output dir')
    parser.add_argmuent('--color', action='store_true', help='render color')
    args, extras = parser.parse_known_args()
    
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))
    
    from gaustudio.utils.misc import load_config
    from gaustudio import datasets
    from gaustudio.utils.cameras_utils import JSON_to_camera

    # Load mesh
    if args.mesh.endswith('.obj'):
        mesh = load_objs_as_meshes([args.mesh]).to("cuda")
    elif args.mesh.endswith('.ply'):
        verts, faces = load_ply(args.mesh)
        mesh = Meshes(verts=[verts], faces=[faces]).to("cuda")
    else:
        exit("Mesh file must be .obj or .ply")
    mesh_bbox = mesh.get_bounding_boxes()[0]
    mesh_center = mesh_bbox.mean(dim=1).cpu().numpy()
    
    if args.camera is not None and os.path.exists(args.camera):
        print("Loading camera data from {}".format(args.camera))
        with open(args.camera, 'r') as f:
            camera_data = json.load(f)
        cameras = []
        for camera_json in camera_data:
            camera = JSON_to_camera(camera_json, "cuda")
            cameras.append(camera)
    elif args.source_path is not None:
        dataset_config = { "name":"colmap", "source_path": args.source_path, "images":"images", "resolution":-1, "data_device":"cuda", "eval": False}
        dataset = datasets.make(dataset_config)
        cameras = dataset.all_cameras
    else:
        from gaustudio.cameras.camera_paths import get_path_from_orbit
        cameras = get_path_from_orbit(mesh_center, 3, elevation=30)

    work_dir = args.output_dir if args.output_dir is not None else os.path.dirname(args.mesh)
    render_path = os.path.join(work_dir, "color")
    normal_path = os.path.join(work_dir, "normal")
    mask_path = os.path.join(work_dir, "mask")
    pose_path = os.path.join(work_dir, "pose")
    intrinsic_path = os.path.join(work_dir, "intrinsic")
    render_depths_path = os.path.join(work_dir, "depth")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(normal_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(pose_path, exist_ok=True)
    os.makedirs(intrinsic_path, exist_ok=True)
    os.makedirs(render_depths_path, exist_ok=True)
    _id = 0
    for camera in tqdm(cameras):
        c2w = torch.inverse(camera.extrinsics) # to c2w
        R, T = c2w[:3, :3], c2w[:3, 3:]
        R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1) # from RDF to LUF for Rotation

        new_c2w = torch.cat([R, T], 1)
        w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]])), 0))
        R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3] # convert R to row-major matrix
        R = R[None] # batch 1 for rendering
        T = T[None] # batch 1 for rendering
        
        intrinsics = camera.intrinsics
        image_size = ((camera.image_height, camera.image_width),)  # (h, w)
        fcl_screen = ((intrinsics[0, 0], intrinsics[1, 1]),)  # fcl_ndc * min(image_size) / 2
        prp_screen = ((intrinsics[0, 2], intrinsics[1, 2]), )  # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2
        view = PerspectiveCameras(focal_length=fcl_screen, principal_point=prp_screen, in_ndc=False, image_size=image_size, R=R, T=T, device="cuda")
        raster_settings = RasterizationSettings(
            image_size=image_size[0],
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        lights = AmbientLights(device="cuda")
        rasterizer = MeshRasterizer(
            cameras=view,
            raster_settings=raster_settings
        )
        if args.color:
            shader = VertexColorShader()
        else:
            shader = pytorch3d.renderer.SoftSilhouetteShader()
        renderer = MeshRendererWithFragments(
            rasterizer = rasterizer,
            shader=shader
        )
        images, fragments = renderer(mesh)
        
        id_str = camera.image_name
        
        if color:
            _image = images[0, ..., :3]
            _mask = images[0, ..., 3]
            _image[_mask < 1] = 0
            torchvision.utils.save_image(_image.permute(2, 0, 1), os.path.join(render_path, f"{_id}.png"))        

        mask = images[0, ..., 3].cpu().numpy() > 0
        cv2.imwrite(os.path.join(mask_path, f"{_id}.png"), (mask * 255).astype(np.uint8))

        rendered_depth = fragments.zbuf[0, :, :, 0].cpu().numpy()
        rendered_depth_vis, _ = np_depth_to_colormap(rendered_depth)
        cv2.imwrite(os.path.join(render_depths_path, f"{_id}.png"), (rendered_depth * 1000).astype(np.uint16))
        cv2.imwrite(os.path.join(render_depths_path, f"{_id}_vis.png"), rendered_depth_vis.astype(np.uint8))
        
        # Save extrinsic and intrinsic
        P_inv = camera.extrinsics.inverse()
        np.savetxt(os.path.join(pose_path, f"{_id}.txt"), P_inv.cpu().numpy())
        np.savetxt(os.path.join(intrinsic_path, f"intrinsic_depth.txt"), camera.intrinsics.cpu().numpy())
        np.savetxt(os.path.join(intrinsic_path, f"intrinsic_color.txt"), camera.intrinsics.cpu().numpy())
        
        """ obtain normal map """
        normal = get_normals_from_fragments(mesh, fragments)[0, :, :, 0] # [H,W,3]
        normal = torch.nn.functional.normalize(normal, 2.0, 2) # normalize to unit-vector
        w2c_R = camera.extrinsics.inverse()[:3, :3].to(normal.device) # 3x3, column-major
        camera_normal = normal @ w2c_R # from world_normal to camera_normal
        normal = camera_normal.cpu().numpy()
        normal[..., 2] *=-1
        normal[..., 1] *=-1
        
        # normal = -normal
        
        normal = cv2.cvtColor(((normal+1)/2*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(normal_path, f"{_id}.png"), normal)
        
        # Save camera infromation
        cam_path = os.path.join(render_path, f"{_id}.cam")
        K = camera.intrinsics.cpu().numpy()
        fx = K[0, 0]
        fy = K[1, 1]
        paspect = fy / fx
        width, height = camera.image_width, camera.image_height
        dim_aspect = width / height
        img_aspect = dim_aspect * paspect
        if img_aspect < 1.0:
            flen = fy / height
        else:
            flen = fx / width
        ppx = K[0, 2] / width
        ppy = K[1, 2] / height

        P = camera.extrinsics
        P = P.cpu().numpy()
        with open(cam_path, 'w') as f:
            s1, s2 = '', ''
            for i in range(3):
                for j in range(3):
                    s1 += str(P[i][j]) + ' '
                s2 += str(P[i][3]) + ' '
            f.write(s2 + s1[:-1] + '\n')
            f.write(str(flen) + ' 0 0 ' + str(paspect) + ' ' + str(ppx) + ' ' + str(ppy) + '\n')
        _id += 1
    

if __name__ == '__main__':
    main()