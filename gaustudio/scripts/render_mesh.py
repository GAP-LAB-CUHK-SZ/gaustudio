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
import click

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

def fix_normals_from_fragments(meshes, fragments, camera):
    """
    Compute pixel normals from mesh fragments and adjust based on camera position.

    Args:
        meshes: A pytorch3d Meshes object containing the mesh data.
        fragments: A pytorch3d Fragments object containing the fragment data.
        camera: A pytorch3d PerspectiveCameras object containing the camera parameters.

    Returns:
        A tensor of shape (H, W, 3) representing the adjusted normals at each pixel.
    """
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]
    
    # Interpolate face normals to get pixel normals
    ones = torch.ones_like(fragments.bary_coords, device=faces_normals.device)
    pixel_normals = pytorch3d.ops.interpolate_face_attributes(
        fragments.pix_to_face, ones, faces_normals
    )
    
    # Obtain pixel positions using depth buffer
    rendered_depth = fragments.zbuf[0, :, :, 0].cpu().numpy()  # Depth buffer in numpy
    pixel_positions = get_pixel_positions_from_depth(rendered_depth, camera)

    # Convert pixel positions from camera space to world space
    pixel_positions = torch.tensor(pixel_positions, device=pixel_normals.device).float()

    # Calculate the direction from each pixel to the camera
    camera_position = camera.camera_center.cpu().numpy()
    camera_position = torch.tensor(camera_position, device=pixel_normals.device).float()
    directions_to_camera = camera_position - pixel_positions
    directions_to_camera = directions_to_camera / torch.norm(directions_to_camera, dim=-1, keepdim=True)  # Normalize directions
    
    # Fix normals based on the camera direction
    pixel_normals = fix_normal_based_on_camera_direction(pixel_normals.squeeze(-2), directions_to_camera)
    
    return pixel_normals

def get_pixel_positions_from_depth(depth, camera):
    """
    Compute pixel positions from the depth buffer using the camera parameters.

    Args:
        depth: A numpy array of shape (H, W) representing the depth buffer.
        camera: A pytorch3d PerspectiveCameras object containing the camera parameters.

    Returns:
        A numpy array of shape (H, W, 3) representing the pixel positions in world coordinates.
    """
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.flatten()
    y = y.flatten()
    z = depth.flatten()

    # Convert from pixel coordinates to camera coordinates using the camera intrinsics
    intrinsics = camera.intrinsics.cpu().numpy()
    x_camera = (x - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y_camera = (y - intrinsics[1, 2]) * z / intrinsics[1, 1]
    z_camera = z

    # Stack to create (H * W, 3) array
    pixel_positions_camera = np.stack([x_camera, y_camera, z_camera], axis=-1)

    # Convert from camera coordinates to world coordinates using the camera extrinsics
    extrinsics = camera.extrinsics.cpu().numpy()
    pixel_positions_world = np.dot(extrinsics[:3, :3], pixel_positions_camera.T).T + extrinsics[:3, 3]
    
    # Reshape to (H, W, 3)
    return pixel_positions_world.reshape((height, width, 3))

def fix_normal_based_on_camera_direction(normals, directions):
    """
    Adjust normals based on the direction from each pixel to the camera.

    Args:
        normals: A tensor of shape (N, 3) representing the normals.
        directions: A tensor of shape (N, 3) representing the directions to the camera.

    Returns:
        A tensor of shape (N, 3) representing the adjusted normals.
    """
    # Compute dot product between normals and directions to the camera
    dot_product = torch.sum(normals * directions, dim=-1)
    
    # Create a mask for normals pointing away from the camera
    mask = dot_product < 0
    
    # Invert normals that are pointing away from the camera
    fixed_normals = normals.clone()
    fixed_normals[mask] = -fixed_normals[mask]
    
    return fixed_normals

def fix_mesh_normals(meshes: Meshes, camera_positions: torch.Tensor):
    """
    Fix mesh normals so they face towards the given camera positions.

    Args:
        meshes: A pytorch3d Meshes object containing the mesh data.
        camera_positions: A tensor of shape (N, 3) representing the positions of N cameras in world coordinates.

    Returns:
        A Meshes object with updated normals.
    """
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    vertices = meshes.verts_packed()  # (V, 3)

    # Convert vertex positions and normals to the same device as the camera positions
    device = camera_positions.device
    vertices = vertices.to(device)
    vertex_normals = vertex_normals.to(device)

    # Initialize the new normals
    new_normals = torch.zeros_like(vertex_normals)

    for camera_position in camera_positions:
        # Compute the direction vectors from each vertex to the camera
        directions_to_camera = camera_position - vertices
        directions_to_camera = directions_to_camera / torch.norm(directions_to_camera, dim=-1, keepdim=True)  # Normalize directions

        # Compute the dot product between vertex normals and directions to the camera
        dot_product = torch.sum(vertex_normals * directions_to_camera, dim=-1)

        # Create a mask for normals pointing away from the camera
        mask = dot_product < 0

        # Invert normals that are pointing away from the camera
        fixed_normals = vertex_normals.clone()
        fixed_normals[mask] = -fixed_normals[mask]

        # Update the new normals (average of all camera views)
        new_normals += fixed_normals

    # Normalize the new normals
    new_normals = new_normals / torch.norm(new_normals, dim=-1, keepdim=True)

    # Create a new Meshes object with updated normals
    fixed_meshes = Meshes(
        verts=[vertices.cuda()],
        faces=[meshes.faces_packed().cuda()],
        verts_normals=[new_normals.cuda()]
    )

    return fixed_meshes
    
@click.command()
@click.option('--gpu', default='0', help='GPU(s) to be used')
@click.option('--dataset', '-d', type=str, default='colmap')
@click.option('--camera', '-c', default=None, help='path to cameras.json')
@click.option('--mesh', '-m', default=None, help='path to the mesh')
@click.option('--source_path', '-s', required=True, help='path to the dataset')
@click.option('--output-dir', '-o', required=True, help='path to the output dir')
@click.option('--color', is_flag=True, help='render color')
def main(gpu, dataset, camera, mesh, source_path, output_dir, color):
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    n_gpus = len(gpu.split(','))
    
    from gaustudio.utils.misc import load_config
    from gaustudio import datasets
    from gaustudio.utils.cameras_utils import JSON_to_camera

    # Load mesh
    if mesh.endswith('.obj'):
        mesh = load_objs_as_meshes([mesh]).to("cuda")
    elif mesh.endswith('.ply'):
        verts, faces = load_ply(mesh)
        mesh = Meshes(verts=[verts], faces=[faces]).to("cuda")
    else:
        exit("Mesh file must be .obj or .ply")
    mesh_bbox = mesh.get_bounding_boxes()[0]
    mesh_center = mesh_bbox.mean(dim=1).cpu().numpy()
    
    if camera is not None and os.path.exists(camera):
        print("Loading camera data from {}".format(camera))
        with open(camera, 'r') as f:
            camera_data = json.load(f)
        cameras = []
        for camera_json in camera_data:
            camera = JSON_to_camera(camera_json, "cuda")
            cameras.append(camera)
    elif source_path is not None:
        dataset_config = { "name": dataset, "source_path": source_path, "images":"images", "resolution":-1, "data_device":"cuda", "eval": False}
        dataset = datasets.make(dataset_config)
        cameras = dataset.all_cameras
    else:
        from gaustudio.cameras.camera_paths import get_path_from_orbit
        cameras = get_path_from_orbit(mesh_center, 3, elevation=30)

    work_dir = output_dir if output_dir is not None else os.path.dirname(mesh)
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
    mesh = fix_mesh_normals(mesh, torch.stack([camera.camera_center for camera in cameras], dim=0))
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
        if color:
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
        elif camera.image is not None:
            images = camera.image[None, ...]
            images = images.permute(0, 3, 1, 2)
            torchvision.utils.save_image(images, os.path.join(render_path, f"{_id}.png"))
            
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
        normal = fix_normals_from_fragments(mesh, fragments, camera)[0, :, :] # [H,W,3]
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