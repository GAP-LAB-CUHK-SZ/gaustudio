import os
import torch
import json
import torchvision
import trimesh
import argparse

from tqdm import tqdm
import numpy as np
import open3d as o3d

from gaustudio.utils.sh_utils import SH2RGB
from gaustudio.utils.misc import load_config
from gaustudio import models, renderers
from gaustudio.datasets.utils import JSON_to_camera, getNerfppNorm

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def create_point_cloud(surface_xyz_np, surface_color_np, surface_normal_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_xyz_np)
    pcd.colors = o3d.utility.Vector3dVector(surface_color_np)
    pcd.normals = o3d.utility.Vector3dVector(surface_normal_np)
    return pcd

def clean_point_cloud(pcd, nb_neighbors=50, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

def mesh_nksr(input_xyz, input_normal, voxel_size=0.04, detail_level=0):
    try:
        from nksr import Reconstructor, utils, fields
    except:
        raise ImportError("Please install nksr to use this feature.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_xyz = input_xyz.to(device)
    input_normal = input_normal.to(device)
    reconstructor = Reconstructor(device)
    field = reconstructor.reconstruct(input_xyz, input_normal, voxel_size=voxel_size, detail_level=detail_level)
    mesh = field.extract_dual_mesh(mise_iter=2)
    return trimesh.Trimesh(vertices=mesh.v.cpu().numpy(), faces=mesh.f.cpu().numpy())

def mesh_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )
    return mesh
    
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
    parser.add_argument('--meshing', choices=['nksr', 'poisson', 
                                                     'poisson-8', 'poisson-9', None], 
                        default='nksr', help='Meshing method to use')
    args, extras = parser.parse_known_args()
    
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))
    
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
    
    render_path = os.path.join(work_dir, "images")
    normal_path = os.path.join(work_dir, "normals")
    mask_path = os.path.join(work_dir, "masks")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(normal_path, exist_ok=True)

    scene_radius = getNerfppNorm(cameras)["radius"]
    all_ids = []
    all_normals = []

    for camera in tqdm(cameras):
        camera.downsample_scale(args.resolution)
        camera = camera.to("cuda")
        with torch.no_grad():
            render_pkg = renderer.render(camera, pcd)
        
        rendering = render_pkg["render"]
        rendered_final_opacity = render_pkg["rendered_final_opacity"][0]
        rendered_depth = render_pkg["rendered_depth"][0] / rendered_final_opacity
        cam_normals = camera.depth2normal(rendered_depth, coordinate='camera')
        normals = camera.normal2worldnormal(cam_normals)
        median_point_depths = render_pkg["rendered_median_depth"][0]
        median_point_ids = render_pkg["rendered_median_depth"][2].int()
        
        fg_mask = rendered_final_opacity > 0.1
        valid_mask = (median_point_depths < scene_radius * 0.8) & (rendered_final_opacity > 0.5)
        valid_mask = (normals.sum(dim=-1) > -3) & valid_mask

        median_point_ids = median_point_ids[valid_mask]
        median_point_normals = -normals[valid_mask]
        
        all_ids.append(median_point_ids)
        all_normals.append(median_point_normals)

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{camera.image_name}.png"))
        torchvision.utils.save_image((cam_normals.permute(2, 0, 1) + 1)/2, os.path.join(normal_path, f"{camera.image_name}.png"))
        torchvision.utils.save_image(fg_mask.float(), os.path.join(mask_path, f"{camera.image_name}.png"))

        # Save camera information
        cam_path = os.path.join(render_path, f"{camera.image_name}.cam")
        K = camera.intrinsics.cpu().numpy()
        fx, fy = K[0, 0], K[1, 1]
        paspect = fy / fx
        width, height = camera.image_width, camera.image_height
        dim_aspect = width / height
        img_aspect = dim_aspect * paspect
        flen = fy / height if img_aspect < 1.0 else fx / width
        ppx, ppy = K[0, 2] / width, K[1, 2] / height

        P = camera.extrinsics.cpu().numpy()
        with open(cam_path, 'w') as f:
            s1, s2 = '', ''
            for i in range(3):
                for j in range(3):
                    s1 += str(P[i][j]) + ' '
                s2 += str(P[i][3]) + ' '
            f.write(s2 + s1[:-1] + '\n')
            f.write(f"{flen} 0 0 {paspect} {ppx} {ppy}\n")

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
    surface_color = SH2RGB(pcd._f_dc[unique_ids]).clip(0,1)
    surface_normal = mean_normals
    
    surface_xyz_np = surface_xyz.cpu().numpy()
    surface_color_np = surface_color.cpu().numpy()
    surface_normal_np = surface_normal.cpu().numpy()
    pcd = create_point_cloud(surface_xyz_np, surface_color_np, surface_normal_np)
    
    pcd = clean_point_cloud(pcd)
    print(f"Point cloud cleaned. Remaining points: {len(pcd.points)}")

    o3d.io.write_point_cloud(os.path.join(work_dir, "fused.ply"), pcd)
    
    if args.meshing == 'nksr':
        input_xyz = torch.from_numpy(np.asarray(pcd.points)).float()
        input_normal = torch.from_numpy(np.asarray(pcd.normals)).float()
        mesh = mesh_nksr(input_xyz, input_normal)
        mesh.export(os.path.join(work_dir, "fused_mesh.ply"))
    elif args.meshing.startswith('poisson'):
        if args.meshing == 'poisson':
            depth = 8
        else:
            depth = int(args.meshing.split('-')[1])
        mesh = mesh_poisson(pcd, depth=depth)
        o3d.io.write_triangle_mesh(os.path.join(work_dir, f"fused_mesh.ply"), mesh)
    elif args.meshing == 'None':
        print("Skipping meshing as requested.")
    else:
        print(f"Unknown meshing method: {args.meshing}")

if __name__ == '__main__':
    main()