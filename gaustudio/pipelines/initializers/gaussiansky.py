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
        self.radius = initializer_config.get('radius', 100.0)
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
class MultiGaussianSkyInitializer(PcdInitializer):
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        self.resolution = initializer_config.get('resolution', 100)
        self.radii = initializer_config.get('radius', list(range(50, 100, 5)))
        self.model_path = os.path.join('/tmp', str(uuid.uuid4())+'.ply')
    
    def cache_dataset(self, dataset=None):
        num_background_points = self.resolution**2
        total_points = num_background_points * len(self.radii)
        
        # Initialize arrays to store data for all spheres
        xyz = np.zeros((total_points, 3))
        normals = np.zeros((total_points, 3))
        colors = np.zeros((total_points, 3), dtype=np.uint8)
        
        # Generate points for each sphere
        for i, radius in enumerate(self.radii):
            start = i * num_background_points
            end = (i + 1) * num_background_points
            
            sphere_xyz, sphere_normals = fibonacci_sphere(num_background_points)
            xyz[start:end] = np.array(sphere_xyz) * radius
            normals[start:end] = sphere_normals
            
            # Set color based on the sphere (you can modify this as needed)
            colors[start:end] = [255, 255, 255]  # White for all spheres
        
        # Create a structured array for the PLY file
        vertex_data = np.zeros(total_points, dtype=[
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
        vertex_data['red'] = colors[:, 0]
        vertex_data['green'] = colors[:, 1]
        vertex_data['blue'] = colors[:, 2]

        # Create a PlyElement and save it
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_element]).write(self.model_path)
