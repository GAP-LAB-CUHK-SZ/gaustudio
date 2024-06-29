# Modified from https://github.com/AiuniAI/Unique3D/blob/main/scripts/project_mesh.py
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
import trimesh

import pytorch3d
from pytorch3d.io import load_ply, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings, 
    MeshRendererWithFragments, 
    MeshRasterizer,  
)

def get_visible_faces(meshes, fragments):
    pix_to_face = fragments.pix_to_face[..., 0]

    unique_faces = torch.unique(pix_to_face.flatten())
    unique_faces = unique_faces[unique_faces != -1]
    return unique_faces

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--camera', '-c', default=None, help='path to cameras.json')
    parser.add_argument('--mesh', '-m', default=None, help='path to the mesh')
    parser.add_argument('--source-path', '-s', required=True, help='path to the dataset')
    parser.add_argument('--output-path', '-o', required=True, help='path to the output dir')
    args, extras = parser.parse_known_args()
    
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))
    
    from gaustudio.utils.misc import load_config
    from gaustudio import datasets
    from gaustudio.utils.cameras_utils import JSON_to_camera

    if args.camera is not None and os.path.exists(args.camera):
        print("Loading camera data from {}".format(args.camera))
        with open(args.camera, 'r') as f:
            camera_data = json.load(f)
        cameras = []
        for camera_json in camera_data:
            camera = JSON_to_camera(camera_json, "cuda")
            cameras.append(camera)
    else:
        dataset_config = { "name":"colmap", "source_path": args.source_path, "images":"images", "resolution":-1, "data_device":"cuda", "eval": False}
        dataset = datasets.make(dataset_config)
        cameras = dataset.all_cameras
    
    # Load mesh
    if args.mesh.endswith('.obj'):
        mesh = load_objs_as_meshes([args.mesh]).to("cuda")
    elif args.mesh.endswith('.ply'):
        verts, faces = load_ply(args.mesh)
        mesh = Meshes(verts=[verts], faces=[faces]).to("cuda")
    else:
        exit("Mesh file must be .obj or .ply")
    vertex_colors = torch.zeros((verts.shape[0], 3), device="cuda")  # RGB zero (can change size as needed)
    
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
        rasterizer = MeshRasterizer(
            cameras=view,
            raster_settings=raster_settings
        )
        shader = pytorch3d.renderer.SoftSilhouetteShader()
        renderer = MeshRendererWithFragments(
            rasterizer = rasterizer,
            shader=shader
        )
        images, fragments = renderer(mesh)
        
        """ obtain unique faces"""
        unique_faces = get_visible_faces(mesh, fragments)

        # visible faces
        faces_normals = mesh.faces_normals_packed()[unique_faces]
        faces_normals = faces_normals / faces_normals.norm(dim=1, keepdim=True)
        world_points = view.unproject_points(torch.tensor([[[0., 0., 0.1], [0., 0., 0.2]]]).to("cuda"))[0]
        view_direction = world_points[1] - world_points[0]
        view_direction = view_direction / view_direction.norm(dim=0, keepdim=True)
        
        # find invalid faces
        cos_angles = (faces_normals * view_direction).sum(dim=1)
        assert cos_angles.mean() < 0, f"The view direction is not correct. cos_angles.mean()={cos_angles.mean()}"
        selected_faces = unique_faces[cos_angles < -0.05]

        # find verts
        faces = mesh.faces_packed()[selected_faces]   # [N, 3]
        verts = torch.unique(faces.flatten())   # [N, 1]
        verts_coordinates = mesh.verts_packed()[verts]   # [N, 3]

        # compute color
        pt_tensor = view.transform_points(verts_coordinates)[..., :2] # NDC space points
        
        _image = camera.image.to("cuda").permute(2, 0, 1)[None]
        _, _, image_height, image_width = _image.shape
        # Normalize x coordinates (column index) to [-1, 1]
        pt_tensor[..., 0] = 2 * (pt_tensor[..., 0] / (image_width - 1)) - 1
        pt_tensor[..., 1] = 2 * (pt_tensor[..., 1] / (image_height - 1)) - 1

        valid = ~((pt_tensor.isnan()|(pt_tensor<-1)|(1<pt_tensor)).any(dim=1))
        valid_pt = pt_tensor[valid, :]
        valid_idx = verts[valid]
        valid_color = torch.nn.functional.grid_sample(_image.flip((-1, -2)), valid_pt[None, :, None, :], align_corners=False, padding_mode="reflection", mode="bilinear")[0, :, :, 0].T.clamp(0, 1)   # [N, 4], note that bicubic may give invalid value
        vertex_colors[valid_idx] = valid_color
    
    verts = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy()
    trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors.cpu().numpy())
    trimesh_mesh.export(args.output_path)



if __name__ == '__main__':
    main()