from gaustudio.models.base import BasePointCloud
from gaustudio import models
import torch
import numpy as np
from plyfile import PlyData, PlyElement


@models.register('general_pcd')
class GeneralPointCloud(BasePointCloud):
    default_conf = {
        'attributes':  {
            "xyz": 3, 
            'rgb': 3,
            "normal": 3,
        },
    }

    def export(self, path):
        # Define the dtype for the structured array
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        
        xyz = self._xyz
        normals = self._normals
        rgb = self._rgb
        
        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, normals, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))

        # Create the PlyData object and write to file
        vertex_element = PlyElement.describe(elements, 'vertex')
        ply_data = PlyData([vertex_element])
        ply_data.write(path)
