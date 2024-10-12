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
import trimesh
import numpy as np
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

import open3d as o3d
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    L = R @ L
    return L

def create_ellipsoid_points(num_points=100):
    phi = torch.linspace(0, 2*np.pi, num_points, device='cuda')
    theta = torch.linspace(-np.pi/2, np.pi/2, num_points, device='cuda')
    phi, theta = torch.meshgrid(phi, theta, indexing='ij')
    
    x = torch.cos(theta) * torch.cos(phi)
    y = torch.cos(theta) * torch.sin(phi)
    z = torch.sin(theta)
    
    points = torch.stack([x, y, z], dim=-1).view(-1, 3)
    return points

def process_batch(xyz, scaling, rotation, normals=None, target_voxel_size=0.04):
    batch_size = xyz.shape[0]
    scaling = torch.cat([scaling, torch.zeros((scaling.shape[0], 1), device=scaling.device)], dim=-1)
    
    # Calculate the number of points for each ellipsoid based on its volume
    volumes = scaling[..., :2].prod(dim=1)
    target_points = (volumes / (target_voxel_size ** 2)).ceil().clamp(min=1, max=1000).long()
    
    max_points = target_points.max().item()
    side_points = int(np.ceil(np.sqrt(max_points)))
    base_points = create_ellipsoid_points(side_points)
    
    # Create points for all ellipsoids
    points = base_points.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Create a mask for valid points
    point_indices = torch.arange(points.shape[1], device=xyz.device)
    mask = point_indices.unsqueeze(0) < target_points.unsqueeze(1)
    
    # Apply mask to points
    points = points[mask]
    
    # Compute the number of points for each ellipsoid
    points_per_ellipsoid = mask.sum(dim=1)
    
    # Build transformation matrices for all ellipsoids in the batch
    transforms = build_scaling_rotation(scaling, rotation)
    
    # Apply transformations to all points in the batch
    ellipsoids = torch.bmm(transforms, points.T.unsqueeze(0).expand(batch_size, -1, -1))
    ellipsoids = ellipsoids.transpose(1, 2)
    
    # Add xyz offsets
    ellipsoids += xyz.unsqueeze(1)
    
    # Flatten the ellipsoids
    ellipsoids = ellipsoids.reshape(-1, 3)
    
    if normals is not None:
        # Repeat normals for each point in the ellipsoid
        repeated_normals = torch.repeat_interleave(normals, points_per_ellipsoid, dim=0)
        return ellipsoids, repeated_normals
    
    return ellipsoids

def sample_ellipsoids(xyz, scaling, rotation, normals=None, target_voxel_size=0.01, batch_size=1000):
    all_ellipsoids = []
    all_normals = []
    
    for i in tqdm(range(0, xyz.shape[0], batch_size)):
        batch_xyz = xyz[i:i+batch_size]
        batch_scaling = scaling[i:i+batch_size]
        batch_rotation = rotation[i:i+batch_size]
        batch_normals = normals[i:i+batch_size] if normals is not None else None
        
        if batch_normals is not None:
            batch_ellipsoids, batch_normals = process_batch(batch_xyz, batch_scaling, batch_rotation, batch_normals, target_voxel_size)
            all_normals.append(batch_normals.cpu().numpy())
        else:
            batch_ellipsoids = process_batch(batch_xyz, batch_scaling, batch_rotation, target_voxel_size=target_voxel_size)
    
    
        batch_ellipsoids = process_batch(batch_xyz, batch_scaling, batch_rotation, target_voxel_size)
        
        batch_ellipsoids = process_batch(batch_xyz, batch_scaling, batch_rotation, target_voxel_size)
        all_ellipsoids.append(batch_ellipsoids.cpu().numpy())

    all_ellipsoids = np.concatenate(all_ellipsoids, axis=0).astype(np.float32)
    print(len(all_ellipsoids))
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_ellipsoids)
    
    if normals is not None:
        all_normals = np.concatenate(all_normals, axis=0).astype(np.float32)
        pcd.normals = o3d.utility.Vector3dVector(all_normals)
    
    return pcd
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', default='vanilla')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--camera', '-c', default=None, help='path to cameras.json')
    parser.add_argument('--model', '-m', default=None, help='path to the model')
    parser.add_argument('--output-dir', '-o', default=None, help='path to the output dir')
    parser.add_argument('--load_iteration', default=-1, type=int, help='iteration to be rendered')
    parser.add_argument('--resolution', default=2, type=int, help='downscale resolution')
    parser.add_argument('--sh', default=0, type=int, help='default SH degree')
    parser.add_argument('--white_background', action='store_true', help='use white background')
    parser.add_argument('--clean', action='store_true', help='perform a clean operation')
    args, extras = parser.parse_known_args()
    
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))
    
    from gaustudio.utils.misc import load_config
    from gaustudio import models, renderers
    try:
        from gaustudio.datasets.utils import JSON_to_camera
    except:
        from gaustudio.utils.cameras_utils import JSON_to_camera
    # parse YAML config to OmegaConf
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, '../configs', args.config+'.yaml')
    config = load_config(config_path, cli_args=extras)
    config.cmd_args = vars(args)  
    
    pcd = models.make(config.model.pointcloud)
    renderer = renderers.make(config.renderer)
    pcd.active_sh_degree = args.sh
    
    model_path = args.model
    if os.path.isdir(model_path):
        if args.load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(args.model, "point_cloud"))
        else:
            loaded_iter = args.load_iteration
        work_dir = os.path.join(model_path, "renders", "iteration_{}".format(loaded_iter)) if args.output_dir is None else args.output_dir
        
        print("Loading trained model at iteration {}".format(loaded_iter))
        pcd.load(os.path.join(args.model,"point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))
    elif model_path.endswith(".ply"):
        work_dir = os.path.join(os.path.dirname(model_path), os.path.basename(model_path)[:-4]) if args.output_dir is None else args.output_dir
        pcd.load(model_path)
    else:
        print("Model not found at {}".format(model_path))
    pcd.to("cuda")
    
    if args.camera is None:
        args.camera = os.path.join(model_path, "cameras.json")
    if os.path.exists(args.camera):
        print("Loading camera data from {}".format(args.camera))
        with open(args.camera, 'r') as f:
            camera_data = json.load(f)
        cameras = []
        for camera_json in camera_data:
            camera = JSON_to_camera(camera_json, "cuda")
            cameras.append(camera)
    else:
        assert "Camera data not found at {}".format(args.camera)
    
    from gaustudio.utils.sh_utils import SH2RGB
    from gaustudio.datasets.utils import getNerfppNorm
    scene_radius = getNerfppNorm(cameras)["radius"]
    all_ids = []
    all_normals = []
    for camera in tqdm(cameras):
        camera.downsample_scale(args.resolution)
        camera = camera.to("cuda")
        with torch.no_grad():
            render_pkg = renderer.render(camera, pcd)
        rendered_final_opacity =  render_pkg["rendered_final_opacity"][0]
        rendered_depth = render_pkg["rendered_depth"][0] / rendered_final_opacity
        normals = camera.depth2normal(rendered_depth, coordinate='world')
        median_point_depths =  render_pkg["rendered_median_depth"][0]
        median_point_ids =  render_pkg["rendered_median_depth"][2].int()
        median_point_weights =  render_pkg["rendered_median_depth"][1]
        valid_mask = (median_point_depths < scene_radius * 0.8) & (rendered_final_opacity > 0.5)
        valid_mask = (normals.sum(dim=-1) > -3) & valid_mask

        median_point_ids = median_point_ids[valid_mask]
        median_point_normals = -normals[valid_mask]
        
        all_ids.append(median_point_ids)
        all_normals.append(median_point_normals)
    all_ids = torch.cat(all_ids, dim=0)
    all_normals = torch.cat(all_normals, dim=0)
    
    # fusion
    unique_ids, inverse_indices = torch.unique(all_ids, return_inverse=True)
    
    num_unique_ids = len(unique_ids)
    sum_normals = torch.zeros((num_unique_ids, all_normals.size(1)), device=all_normals.device)
    counts = torch.zeros(num_unique_ids, device=all_ids.device)
    sum_normals.index_add_(0, inverse_indices, all_normals)
    counts.index_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float))
    mean_normals = sum_normals / counts.unsqueeze(1)
    mean_normals = torch.nn.functional.normalize(mean_normals, p=2, dim=1)

    surface_xyz = pcd._xyz[unique_ids]
    surface_scaling = pcd.get_scaling[unique_ids]
    surface_ratation = pcd.get_rotation[unique_ids]
    surface_color = SH2RGB(pcd._f_dc[unique_ids]).clip(0,1)
    surface_normal = mean_normals
    
    surface_xyz_np = surface_xyz.cpu().numpy()
    surface_color_np = surface_color.cpu().numpy()
    surface_normal_np = surface_normal.cpu().numpy()
    
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_xyz_np)
    pcd.colors = o3d.utility.Vector3dVector(surface_color_np)
    pcd.normals = o3d.utility.Vector3dVector(surface_normal_np)
    o3d.io.write_point_cloud(os.path.join(args.output_dir, "fused.ply"), pcd)
    
    from nksr import Reconstructor, utils, fields
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_xyz = torch.from_numpy(surface_xyz_np).float().to(device)
    input_normal = torch.from_numpy(surface_normal_np).float().to(device)
    
    # Perform reconstruction
    reconstructor = Reconstructor(device)
    field = reconstructor.reconstruct(input_xyz, input_normal, voxel_size=0.04, detail_level=0)
    mesh = field.extract_dual_mesh(mise_iter=2)
    mesh = trimesh.Trimesh(vertices=mesh.v.cpu().numpy(), faces=mesh.f.cpu().numpy())
    mesh.export(os.path.join(args.output_dir, "fused_mesh.ply"))

if __name__ == '__main__':
    main()