import os
from pathlib import Path
import math
import glob
import time
import json
import argparse
from typing import Tuple, Literal, List

import numpy as np
import viser
import viser.transforms as vtf
import torch
from utils.viewer import ClientThread

import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser

import copy
import numpy as np
import cv2

DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE = "@Direct"

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

class Viewer:
    def __init__(
            self,
            gaussians,
            renderer,
            host: str = "0.0.0.0",
            port: int = 8080,
            image_format: Literal["jpeg", "png"] = "jpeg",
            reorient: Literal["auto", "enable", "disable"] = "auto",
            cameras_json: str = None,
    ):
        self.device = torch.device("cuda")

        self.host = host
        self.port = port
        self.image_format = image_format

        self.up_direction = np.asarray([0., 0., 1.])
        self.gaussians = gaussians
        self.renderer = renderer

        cameras_json_path = cameras_json
        # reorient the scene
        self.camera_transform = self._reorient(cameras_json_path, mode=reorient)
        # load camera poses
        self.camera_poses = self.load_camera_poses(cameras_json_path)

        self.clients = {}
        
    def _reorient(self, cameras_json_path: str, mode: str, dataset_type: str = None):
        transform = torch.eye(4, dtype=torch.float)

        if mode == "disable":
            return transform

        # detect whether cameras.json exists
        is_cameras_json_exists = os.path.exists(cameras_json_path)

        if is_cameras_json_exists is False:
            if mode == "enable":
                raise RuntimeError("{} not exists".format(cameras_json_path))
            else:
                return transform

        # skip reorient if dataset type is blender
        if dataset_type in ["blender", "nsvf"] and mode == "auto":
            print("skip reorient for {} dataset".format(dataset_type))
            return transform

        print("load {}".format(cameras_json_path))
        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)
        up = torch.zeros(3)
        for i in cameras:
            up += torch.tensor(i["rotation"])[:3, 1]
        up = -up / torch.linalg.norm(up)

        print("up vector = {}".format(up))
        self.up_direction = up

        return transform

    def load_camera_poses(self, cameras_json_path: str):
        if os.path.exists(cameras_json_path) is False:
            return []
        with open(cameras_json_path, "r") as f:
            return json.load(f)

    def add_cameras_to_scene(self, viser_server):
        if len(self.camera_poses) == 0:
            return

        self.camera_handles = []

        camera_pose_transform = np.linalg.inv(self.camera_transform.cpu().numpy())
        for camera in self.camera_poses:
            name = camera["img_name"]
            c2w = np.eye(4)
            c2w[:3, :3] = np.asarray(camera["rotation"])
            c2w[:3, 3] = np.asarray(camera["position"])
            c2w[:3, 1:3] *= -1
            c2w = np.matmul(camera_pose_transform, c2w)

            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)

            cx = camera["width"] // 2
            cy = camera["height"] // 2
            fx = camera["fx"]

            camera_handle = viser_server.add_camera_frustum(
                name="cameras/{}".format(name),
                fov=float(2 * np.arctan(cx / fx)),
                scale=0.1,
                aspect=float(cx / cy),
                wxyz=R.wxyz,
                position=c2w[:3, 3],
                color=(205, 25, 0),
            )

            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles.append(camera_handle)

        self.camera_visible = True

        def toggle_camera_visibility(_):
            with viser_server.atomic():
                self.camera_visible = not self.camera_visible
                for i in self.camera_handles:
                    i.visible = self.camera_visible

        with viser_server.add_gui_folder("Cameras"):
            self.toggle_camera_button = viser_server.add_gui_button("Toggle Camera Visibility")
        self.toggle_camera_button.on_click(toggle_camera_visibility)


    def start(self):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        server.configure_theme(
            control_layout="collapsible",
            show_logo=False,
        )
        # register hooks
        server.on_client_connect(self._handle_new_client)
        server.on_client_disconnect(self._handle_client_disconnect)

        tabs = server.add_gui_tab_group()

        with tabs.add_tab("General"):
            reset_up_button = server.add_gui_button(
                "Reset up direction",
                icon=viser.Icon.ARROW_AUTOFIT_UP,
                hint="Reset the orbit up direction.",
            )

            @reset_up_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                event.client.camera.up_direction = vtf.SO3(event.client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

            # add cameras
            self.add_cameras_to_scene(server)

            # add render options
            with server.add_gui_folder("Render"):
                self.max_res_when_static = server.add_gui_slider(
                    "Max Res",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1920,
                )
                self.max_res_when_static.on_update(self._handle_option_updated)
                self.jpeg_quality_when_static = server.add_gui_slider(
                    "JPEG Quality",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=100,
                )
                self.jpeg_quality_when_static.on_update(self._handle_option_updated)

                self.max_res_when_moving = server.add_gui_slider(
                    "Max Res when Moving",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1280,
                )
                self.jpeg_quality_when_moving = server.add_gui_slider(
                    "JPEG Quality when Moving",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=60,
                )

        while True:
            time.sleep(999)

    def _handle_option_updated(self, _):
        """
        Simply push new render to all client
        """
        return self.rerender_for_all_client()

    def handle_option_updated(self, _):
        return self._handle_option_updated(_)

    def rerender_for_client(self, client_id: int):
        """
        Render for specific client
        """
        try:
            # switch to low resolution mode first, then notify the client to render
            self.clients[client_id].state = "low"
            self.clients[client_id].render_trigger.set()
        except:
            # ignore errors
            pass

    def rerender_for_all_client(self):
        for i in self.clients:
            self.rerender_for_client(i)

    def _handle_new_client(self, client: viser.ClientHandle) -> None:
        """
        Create and start a thread for every new client
        """

        # create client thread
        # client_thread = ClientThread(self, self.viewer_renderer, client)
        # client_thread.start()
        # store this thread
        # self.clients[client.client_id] = client_thread

    def _handle_client_disconnect(self, client: viser.ClientHandle):
        """
        Destroy client thread when client disconnected
        """

        try:
            self.clients[client.client_id].stop()
            del self.clients[client.client_id]
        except Exception as err:
            print(err)


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('-m', default=None, help='path to the model')
    parser.add_argument('--load_iteration', default=-1, type=int, help='iteration to be rendered')
    parser.add_argument("--host", "-a", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--image_format", "--image-format", "-f", type=str, default="jpeg")
    parser.add_argument("--reorient", type=str, default="auto",
                        help="whether reorient the scene, available values: auto, enable, disable")
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
    if args.load_iteration:
        if args.load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(args.m, "point_cloud"))
        else:
            loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(loaded_iter))

    pcd = models.make(config.model.pointcloud.name, config.model.pointcloud)
    pcd.load(os.path.join(args.m,"point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))
    
    renderer = renderers.make(config.renderer.name, config.renderer)

    # create viewer
    viewer = Viewer(pcd, renderer, cameras_json=os.path.join(args.m, 'cameras.json'))

    # start viewer server
    viewer.start()