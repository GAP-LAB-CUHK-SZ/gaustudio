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
import copy
from kornia.geometry.depth import depth_from_disparity
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
    parser.add_argument('--baseline', default=0.2, type=float, help='default baseline length')
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
    
    disparity_predictor = torch.hub.load("hugoycj/unimatch-hub", "UniMatchStereo", trust_repo=True)
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
        vdb_volume = vdbfusion.VDBVolume(voxel_size=0.01, sdf_trunc=0.04, space_carving=True) # For Scene
    else:
        assert "Camera data not found at {}".format(args.camera)

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

            stereo_camera = copy.deepcopy(camera)
            stereo_camera.translate([-1 * args.baseline, 0, 0])
            stereo_render_pkg = renderer.render(stereo_camera.to("cuda"), pcd)
        rendering = render_pkg["render"]
        stereo_rendering = stereo_render_pkg["render"]
        
        rendering_np = (rendering.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        stereo_rendering_np = (stereo_rendering.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        with torch.no_grad():
            disparity = disparity_predictor.infer_cv2(rendering_np, stereo_rendering_np)

        invalid_mask = render_pkg['rendered_final_opacity'][0].cpu() < 0.5
        stereo_depth = depth_from_disparity(torch.tensor(disparity), args.baseline, camera.focal_x)
        stereo_depth[invalid_mask] = 0
        
        rendered_pcd_cam, rendered_pcd_world = depth2point(stereo_depth, camera.intrinsics.to(stereo_depth.device), 
                                                                      camera.extrinsics.to(stereo_depth.device))
        rendered_pcd_world = rendered_pcd_world[~invalid_mask]
        
        # rendered_depth = render_pkg["rendered_median_depth"][0]
        # invalid_mask = render_pkg["rendered_final_opacity"][0] < 0.5
        # rendered_depth[invalid_mask] = 0
        
        # rendered_pcd_cam, rendered_pcd_world = depth2point(rendered_depth, camera.intrinsics.to(rendered_depth.device), 
        #                                                               camera.extrinsics.to(rendered_depth.device))
        # rendered_pcd_world = rendered_pcd_world[~invalid_mask]
        
        P = camera.extrinsics
        P_inv = P.inverse()
        cam_center = P_inv[:3, 3]
        vdb_volume.integrate(rendered_pcd_world.double().cpu().numpy(), extrinsic=cam_center.double().cpu().numpy())
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{camera.image_name}.png"))
        torchvision.utils.save_image(stereo_rendering, os.path.join(render_path, f"{camera.image_name}_right.png"))
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
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(os.path.join(work_dir, 'fused_mesh.ply'))
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_null_faces()
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=20000)
        ms.save_current_mesh(os.path.join(work_dir, 'fused_mesh.ply'))

if __name__ == '__main__':
    main()