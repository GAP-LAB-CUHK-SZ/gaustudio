import os
import pycolmap
import numpy as np
import torchvision
import torch
from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.base import BaseInitializer
from gaustudio.pipelines.initializers.pcd import PcdInitializer
import math
import uuid
from plyfile import PlyData, PlyElement

def fibonacci_sphere(samples=1):
    points = []
    normals = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y**2)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        point = (x, y, z)
        points.append(point)

        # Calculate normal (pointing towards center)
        normal = (-x, -y, -z)
        magnitude = math.sqrt(sum(n**2 for n in normal))
        normalized_normal = tuple(n / magnitude for n in normal)
        normals.append(normalized_normal)

    return np.array(points), np.array(normals)

def euclidean_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def inverse_sigmoid(x):
    return np.log(x / (1 - x))

@initializers.register('gaussiansky')
class GaussianSkyInitializer(PcdInitializer):
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        self.resolution = initializer_config.get('resolution', 100)
        self.radius = initializer_config.get('radius', 10.0)
        self.model_path = os.path.join('/tmp', str(uuid.uuid4())+'.ply')
    
    def cache_dataset(self, dataset=None):
        num_background_points = self.resolution**2
        xyz, normals = fibonacci_sphere(num_background_points)
        xyz = np.array(xyz) * self.radius
        
        # Create a structured array that includes vertex coordinates, normals, and colors
        vertex_data = np.zeros(xyz.shape[0], dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        vertex_data['x'] = xyz[:, 0]
        vertex_data['y'] = xyz[:, 1]
        vertex_data['z'] = xyz[:, 2]
        vertex_data['nx'] = normals[:, 0]
        vertex_data['ny'] = normals[:, 1]
        vertex_data['nz'] = normals[:, 2]
        vertex_data['red'] = 255
        vertex_data['green'] = 255
        vertex_data['blue'] = 255

        # Create a PlyElement and save it
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(self.model_path)

@initializers.register('multigaussiansky')
class MultiGaussianSkyInitializer(BaseInitializer):
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        self.resolution = initializer_config.get('resolution', 100)
        self.radius = initializer_config.get('radius', [50, 100, 150])

    def build_model(self, model):
        num_background_points = self.resolution**2
        xyzs = []
        scales = []
        opacities = []
        rgbs = []
        for radius in self.radius:
            num_background_points = num_background_points
            sphere_xyz = np.array(fibonacci_sphere(num_background_points))
            xyz = sphere_xyz * radius
            dist = euclidean_distance(xyz[0], xyz[1])
            dist = math.log(dist)
            scale = np.ones_like(xyz) * dist

            opacity = inverse_sigmoid(0.01 * np.ones((xyz.shape[0], 1)))
            rgb = np.ones((xyz.shape[0], 3))
            rgbs.append(rgb)
            xyzs.append(xyz)
            scales.append(scale)
            opacities.append(opacity)
        xyzs = np.concatenate(xyzs, axis=0)
        rgbs = np.concatenate(rgbs, axis=0)
        scales = np.concatenate(scales, axis=0)
        opacities = np.concatenate(opacities, axis=0)
        try:
            model.create_from_attribute(xyz=xyzs, rgb=rgbs, scale=scales, opacity=opacities)
        except Exception as e:
            print(f"Failed to update point cloud: {e}")
            raise
        return model