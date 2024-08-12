#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
import numpy as np
from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.base import BaseInitializer
import open3d as o3d
import torch

def inverse_sigmoid(x):
    return np.log(x / (1 - x))

# adopted from https://github.com/turandai/gaussian_surfels/blob/main/utils/general_utils.py
def normal2rotation(n):
    # construct a random rotation matrix from normal
    # it would better be positive definite and orthogonal
    n = torch.nn.functional.normalize(n)
    w0 = torch.tensor([[1, 0, 0]]).expand(n.shape)
    R0 = w0 - torch.sum(w0 * n, -1, True) * n
    R0 *= torch.sign(R0[:, :1])
    R0 = torch.nn.functional.normalize(R0)
    R1 = torch.cross(n, R0)
    
    R1 *= torch.sign(R1[:, 1:2]) * torch.sign(n[:, 2:])
    R = torch.stack([R0, R1, n], -1)
    q = rotmat2quaternion(R)

    return q

def rotmat2quaternion(R, normalize=False):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr) / 2
    # print(torch.sum(torch.isnan(r)))
    q = torch.stack([
        r,
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * r),
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * r)
    ], -1)
    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1)
    return q


@initializers.register('mesh')
class MeshInitializer(BaseInitializer):
    n_gaussians_per_surface_triangle = 1
    
    # This code is adopted from https://github.com/Anttwo/SuGaR/blob/main/sugar_scene/sugar_model.py
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        if self.n_gaussians_per_surface_triangle == 1:
            self.surface_triangle_circle_radius = 1. / 2. / np.sqrt(3.)
            self.surface_triangle_bary_coords = torch.tensor(
                [[1/3, 1/3, 1/3]],
                dtype=torch.float32,
            )[..., None]
        elif self.n_gaussians_per_surface_triangle == 3:
            self.surface_triangle_circle_radius = 1. / 2. / (np.sqrt(3.) + 1.)
            self.surface_triangle_bary_coords = torch.tensor(
                [[1/2, 1/4, 1/4],
                [1/4, 1/2, 1/4],
                [1/4, 1/4, 1/2]],
                dtype=torch.float32,
            )[..., None]
        elif self.n_gaussians_per_surface_triangle == 4:
            self.surface_triangle_circle_radius = 1 / (4. * np.sqrt(3.))
            self.surface_triangle_bary_coords = torch.tensor(
                [[1/3, 1/3, 1/3],
                [2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3]],
                dtype=torch.float32,
            )[..., None]  # n_gaussians_per_face, 3, 1  
        elif self.n_gaussians_per_surface_triangle == 6:
            self.surface_triangle_circle_radius = 1 / (4. + 2.*np.sqrt(3.))
            self.surface_triangle_bary_coords = torch.tensor(
                [[2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3],
                [1/6, 5/12, 5/12],
                [5/12, 1/6, 5/12],
                [5/12, 5/12, 1/6]],
                dtype=torch.float32,
            )[..., None]
            
    def __call__(self, model, mesh, dataset=None, overwrite=False):
        self.mesh = mesh
        self.mesh.compute_vertex_normals()
        model = self.build_model(model)

    def build_model(self, model):
        faces = torch.tensor(np.array(self.mesh.triangles))
        vertex_points = torch.tensor(np.array(self.mesh.vertices)).float()
        vertex_colors = torch.tensor(np.array(self.mesh.vertex_colors)).float()
        vertex_normals = torch.tensor(np.array(self.mesh.vertex_normals)).float()
        
        has_color = vertex_colors.shape[0] > 0
        
        faces_verts = vertex_points[faces]  # n_faces, 3, n_coords
        faces_normals = vertex_normals[faces]  # n_faces, 3, n_coords
        
        points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        points = points.reshape(-1, 3)
        
        # Calculate points_normal
        points_normal = faces_normals[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        points_normal = points_normal.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
        points_normal = points_normal.reshape(-1, 3)
        points_normal = torch.nn.functional.normalize(points_normal, dim=-1)
        rotations = normal2rotation(points_normal)
        
        if has_color:
            faces_colors = vertex_colors[faces]  # n_faces, 3, n_coords
            colors = faces_colors[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_colors
            colors = colors.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_colors
            colors = colors.reshape(-1, 3)  # n_faces * n_gaussians_per_face, n_colors
        else:
            colors = None
            
        scales = (faces_verts - faces_verts[:, [1, 2, 0]]).norm(dim=-1).min(dim=-1)[0] * self.surface_triangle_circle_radius
        scales = scales.clamp_min(0.).reshape(len(faces_verts), -1, 1).expand(-1, self.n_gaussians_per_surface_triangle, 2).clone().reshape(-1, 2)
        scales = torch.cat([scales, torch.zeros((scales.shape[0], 1))], dim=-1)
        log_scales = torch.log(scales * 2 + 1e-7)
        
        opacity = inverse_sigmoid(np.ones((points.shape[0], 1)))
        model.create_from_attribute(xyz=points, rgb=colors, scale=log_scales, opacity=opacity, rot=rotations)

        return model
