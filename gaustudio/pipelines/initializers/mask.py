import os
import numpy as np
import torch
import trimesh
from tqdm import tqdm
import tempfile
from copy import deepcopy
from gaustudio.pipelines.initializers.base import BaseInitializer
from gaustudio.pipelines import initializers
try:
    import mcubes
    mcubes_available = True
except:
    mcubes_available = False

@initializers.register('VisualHull')
class VisualHullInitializer(BaseInitializer):
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        self.ws_dir = self.initializer_config.get('workspace_dir')
        if self.ws_dir is None:
            self.ws_dir = tempfile.mkdtemp()
            print(f"No workspace directory provided. Using temporary directory: {self.ws_dir}")
        
        if not mcubes_available:
            raise ImportError("PyMCubes is not installed. Please install PyMCubes to use VisualHullInitializer.")
        os.makedirs(self.ws_dir, exist_ok=True)
        self.resolution = self.initializer_config.get('resolution', 128)
        self.threshold = self.initializer_config.get('threshold', 0.5)
        self.radius_scale = self.initializer_config.get('radius_scale', 1.2)

    def __call__(self, model, dataset, overwrite=False):
        if not os.path.exists(f'{self.ws_dir}/visual_hull.ply') or overwrite:
            self.construct_visual_hull(dataset)
        model = self.build_model(model)
        return model

    def construct_visual_hull(self, dataset):
        print("Constructing visual hull...")
        translate = dataset.cameras_center
        radius = dataset.cameras_min_extent * self.radius_scale
        # Create a grid of points in the normalized scene space
        x, y, z = np.meshgrid(
            np.linspace(-radius, radius, self.resolution),
            np.linspace(-radius, radius, self.resolution),
            np.linspace(-radius, radius, self.resolution)
        )
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        
        # Transform points to world space
        points_world = points - translate
        pcd = trimesh.Trimesh(vertices=points_world)
        pcd.export(os.path.join(self.ws_dir, 'voxel_points.ply'))
        print(f"Visual hull mesh saved to {self.ws_dir}/voxel_points.ply")

        points_world = torch.from_numpy(points_world).float().cuda()
        filled = torch.ones((points_world.shape[0])).cuda().bool()

        for camera in tqdm(dataset[::10]):
            camera = deepcopy(camera)
            camera = camera.to('cuda')

            inside_view = camera.insideView(points_world)
            inside_view_idx = torch.where(inside_view)[0]
            inside_mask = camera.insideView(points_world[inside_view_idx], camera.mask)
            camera_filled = torch.zeros((points_world.shape[0])).cuda().bool()
            camera_filled[inside_view_idx] = inside_mask
            filled = filled & camera_filled
        
        trimmed_points = points_world[filled].cpu().numpy()
        volume = filled.reshape(self.resolution, self.resolution, self.resolution).cpu().numpy()
        pcd_trimmed = trimesh.PointCloud(vertices=trimmed_points)
        pcd_trimmed.export(os.path.join(self.ws_dir, 'trim_points.ply'))
        print(f"Trimmed points saved to {self.ws_dir}/trim_points.ply")
        print(f"Remaining points: {trimmed_points.shape[0]}")

        vertices, faces = self.extract_mesh(volume, translate, radius)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(os.path.join(self.ws_dir, 'visual_hull.ply'))
        print(f"Visual hull mesh saved to {self.ws_dir}/visual_hull.ply")

    def extract_mesh(self, volume, translate, radius):
        vertices, faces = mcubes.marching_cubes(volume, self.threshold)

        faces = np.fliplr(faces)
        vertices = vertices.dot(np.array([[0, 1, 0],
                                          [1, 0, 0],
                                          [0, 0, 1]]))
        # Scale vertices back to world space
        vertices = vertices / (self.resolution - 1) * (2 * radius) - radius
        vertices = vertices - translate
        
        return vertices, faces

    def build_model(self, model):
        print("Building model from visual hull...")
        
        mesh = trimesh.load(os.path.join(self.ws_dir, 'visual_hull.ply'))
        xyz = torch.from_numpy(mesh.vertices).float().cuda()
        
        num_points = xyz.shape[0]
        opacity = torch.ones((num_points, 1)).cuda() * 0.1  # Initialize with low opacity
        rgb = torch.ones((num_points, 3)).cuda() * 0.5  # Initialize with gray color
        scales = torch.ones((num_points, 3)).cuda() * 0.01  # Initialize with small scale
        
        model.create_from_attribute(xyz=xyz, rgb=rgb, opacity=opacity, scale=scales)
        print(f"Initialized model with {num_points} Gaussians")
        return model