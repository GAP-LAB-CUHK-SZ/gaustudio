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
    from gaustudio import models, datasets, renderers
    from gaustudio.datasets.utils import JSON_to_camera
    from gaustudio.utils.graphics_utils import depth2point
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
    from gaustudio.utils.graphics_utils import depth_to_normal
    scene_radius = getNerfppNorm(cameras)["radius"]
    all_points = []
    all_colors = []
    all_normals = []
    for camera in tqdm(cameras[::3]):
        camera.downsample_scale(args.resolution)
        camera = camera.to("cuda")
        with torch.no_grad():
            render_pkg = renderer.render(camera, pcd)
        rendered_final_opacity =  render_pkg["rendered_final_opacity"][0]
        rendered_depth = render_pkg["rendered_depth"]
        depth_gradients = depth_to_normal(camera, rendered_depth)

        median_point_depths =  render_pkg["rendered_median_depth"][0]
        median_point_ids =  render_pkg["rendered_median_depth"][2].int()
        median_point_weights =  render_pkg["rendered_median_depth"][1]
        valid_mask = (rendered_final_opacity > 0.5) & (median_point_weights > 0.1)
        valid_mask = (median_point_depths < scene_radius * 1.5) & valid_mask
        valid_mask = (depth_gradients.sum(dim=-1) > 0) & valid_mask
        
        median_point_ids = median_point_ids[valid_mask]
        median_point_points = pcd._xyz[median_point_ids]
        median_point_colors = SH2RGB(pcd._f_dc[median_point_ids])
        median_point_normals = depth_gradients[valid_mask]
        
        all_points.append(median_point_points)
        all_colors.append(median_point_colors)
        all_normals.append(median_point_normals)
    all_points = torch.cat(all_points, dim=0)
    all_colors = torch.cat(all_colors, dim=0)
    all_normals = torch.cat(all_normals, dim=0)
    
    import open3d as o3d
    surface_xyz_np = all_points.cpu().numpy()
    surface_color_np = all_colors.cpu().numpy() * 255
    surface_normal_np = all_normals.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_xyz_np)
    pcd.colors = o3d.utility.Vector3dVector(surface_color_np / 255.0)
    pcd.normals = o3d.utility.Vector3dVector(surface_normal_np)
    
    o3d.io.write_point_cloud(os.path.join(args.output_dir, "fused.ply"), pcd,)
    o3d.visualization.draw_geometries([pcd])
if __name__ == '__main__':
    main()