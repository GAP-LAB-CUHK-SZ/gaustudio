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
from gaustudio.datasets.utils import JSON_to_camera
def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('-c', default=None, help='path to cameras.json')
    parser.add_argument('-m', default=None, help='path to the model')
    parser.add_argument('--load_iteration', default=-1, type=int, help='iteration to be rendered')
    parser.add_argument('--white_background', action='store_true', help='use white background')
    args, extras = parser.parse_known_args()
    
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))
    
    from gaustudio.utils.misc import load_config
    from gaustudio import models, datasets, renderers
    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)  
    
    model_path = args.m
    split_name = "images"
    if args.load_iteration:
        if args.load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(args.m, "point_cloud"))
        else:
            loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(loaded_iter))

    pcd = models.make(config.model.pointcloud.name, config.model.pointcloud)
    renderer = renderers.make(config.renderer.name, config.renderer)
    with open(args.c, 'r') as f:
        camera_data = json.load(f)
    cameras = []
    for camera_json in camera_data:
        camera = JSON_to_camera(camera_json, "cuda")
        cameras.append(camera)

    pcd.load(os.path.join(args.m,"point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))
    bg_color = [1,1,1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_path = os.path.join(model_path, split_name, "ours_{}".format(loaded_iter), "renders")
    gts_path = os.path.join(model_path, split_name, "ours_{}".format(loaded_iter), "gt")
    render_depths_path = os.path.join(model_path, split_name, "ours_{}".format(loaded_iter), "rendered_depth")
    gt_depths_path = os.path.join(model_path, split_name, "ours_{}".format(loaded_iter), "gt_depth")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(render_depths_path, exist_ok=True)
    os.makedirs(gt_depths_path, exist_ok=True)
    
    for camera in cameras:
        render_pkg = renderer.render(camera, pcd)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{camera.image_name}.png"))

if __name__ == '__main__':
    main()