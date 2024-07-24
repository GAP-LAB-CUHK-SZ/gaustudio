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
import vdbfusion
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
    parser.add_argument('--model', '-m', default=None, help='path to the model')
    parser.add_argument('--source_path', '-s', help='path to the dataset')
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
    
    if args.source_path is None:
        args.source_path = os.path.join(os.path.dirname(model_path), "cameras.json")

    if args.source_path.endswith(".json"):
        print("Loading camera data from {}".format(args.source_path))
        with open(args.source_path, 'r') as f:
            camera_data = json.load(f)
        cameras = []
        for camera_json in camera_data:
            camera = JSON_to_camera(camera_json, "cuda")
            cameras.append(camera)
    else:
        dataset_config = { "name":"colmap", "source_path": args.source_path, 
                          "images":"images", "resolution":-1, "data_device":"cuda", "eval": False}
        dataset = datasets.make(dataset_config)
        cameras = dataset.all_cameras
    vdb_volume = vdbfusion.VDBVolume(voxel_size=0.01, sdf_trunc=0.04, space_carving=False) # For Scene

    bg_color = [1,1,1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_path = os.path.join(work_dir, "images")
    mask_path = os.path.join(work_dir, "masks")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    for camera in tqdm(cameras[::3]):
        camera.downsample_scale(args.resolution)
        camera = camera.to("cuda")
        with torch.no_grad():
            render_pkg = renderer.render(camera, pcd)
        rendering = render_pkg["render"]
        rendered_final_opacity =  render_pkg["rendered_final_opacity"][0]
        # rendered_depth = render_pkg["rendered_depth"][0] / rendered_final_opacity
        rendered_depth = render_pkg["rendered_median_depth"][0]
        invalid_mask = rendered_final_opacity < 0.5

        rendering[:, invalid_mask] = 0.
        rendered_depth[invalid_mask] = 0

        rendered_pcd_world = camera.depth2point(rendered_depth, coordinate='world')
        rendered_pcd_world = rendered_pcd_world[~invalid_mask]
        
        P = camera.extrinsics
        P_inv = P.inverse()
        cam_center = P_inv[:3, 3]
        vdb_volume.integrate(rendered_pcd_world.double().cpu().numpy(), extrinsic=cam_center.double().cpu().numpy())
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{camera.image_name}.png"))
        torchvision.utils.save_image((~invalid_mask).float(), os.path.join(mask_path, f"{camera.image_name}.png"))
        
        # Save camera infromation
        cam_path = os.path.join(render_path, f"{camera.image_name}.cam")
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

        P = P.cpu().numpy()
        with open(cam_path, 'w') as f:
            s1, s2 = '', ''
            for i in range(3):
                for j in range(3):
                    s1 += str(P[i][j]) + ' '
                s2 += str(P[i][3]) + ' '
            f.write(s2 + s1[:-1] + '\n')
            f.write(str(flen) + ' 0 0 ' + str(paspect) + ' ' + str(ppx) + ' ' + str(ppy) + '\n')
        
    vertices, faces = vdb_volume.extract_triangle_mesh(min_weight=5)
    geo_mesh = trimesh.Trimesh(vertices, faces)
    geo_mesh.export(os.path.join(work_dir, 'fused_mesh.ply'))
    
    # Clean Mesh
    if args.clean:
        import open3d as o3d
        ratio_threshold = 0.5
        mesh_path = os.path.join(work_dir, 'fused_mesh.ply')
        print(f"Loading mesh from {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        print("Analyzing connected components...")
        (triangle_clusters, 
        cluster_n_triangles,
        cluster_area) = mesh.cluster_connected_triangles()
        
        print("Finding largest component...") 
        triangle_clusters = np.array(triangle_clusters)  
        cluster_n_triangles = np.array(cluster_n_triangles)
        
        largest_cluster_idx = cluster_n_triangles.argmax()
        largest_cluster_n_triangles = cluster_n_triangles[largest_cluster_idx]

        print(f"Largest component has {largest_cluster_n_triangles} triangles")
        
        triangles_keep_mask = np.zeros_like(triangle_clusters, dtype=np.int32)
        saved_clusters = []
        
        print("Removing small components...")
        for i, n_tri in enumerate(cluster_n_triangles):
            if n_tri > ratio_threshold * largest_cluster_n_triangles:
                saved_clusters.append(i)
                triangles_keep_mask += triangle_clusters == i
                
        triangles_to_remove = triangles_keep_mask == 0
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        
        print(f"Removed {triangles_to_remove.sum()} triangles")
        print(f"Saving processed mesh to {mesh_path}") 
        o3d.io.write_triangle_mesh(mesh_path, mesh)

if __name__ == '__main__':
    main()