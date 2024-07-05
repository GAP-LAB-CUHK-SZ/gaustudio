from gaustudio.models.base import BasePointCloud
from gaustudio.models.utils import get_activation, build_covariance_from_scaling_rotation
from gaustudio import models

import torch

@models.register('vanilla_pcd')
class VanillaPointCloud(BasePointCloud):
    default_conf = {
        'sh_degree': 3,
        'attributes':  {
            "xyz": 3, 
            'opacity': 1,
            "f_dc": 3,
            "f_rest": 45,
            "scale": 3,
            "rot": 4
        },
        'activations':{
            "scale": "exp",
            "opacity": "sigmoid",
            "rot": "normalize"
        }
    }
    
    def __init__(self, config, device=None) -> None:
        super().__init__(config, device)
        self.active_sh_degree = 0
        self.max_sh_degree = self.config["sh_degree"]
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        

        # TODO: Move resume to datasets
        resume_path = self.config.get('resume_path', None)
        if resume_path is not None:
            print(f"Resuming pointcloud")
            self.load(resume_path)

    def setup_functions(self):        
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.scaling_inverse_activation = torch.log
        self.inverse_opacity_activation = lambda x: torch.log(x/(1-x))

    def get_attribute(self, attribute):
        if attribute in self.config["activations"]:
            activation_function = get_activation(self.config["activations"][attribute])
            return activation_function(getattr(self, '_'+attribute))
        else:
            return getattr(self, '_'+attribute)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_attribute("scale"), scaling_modifier, self._rot)

    @property
    def get_features(self):
        features_dc = self._f_dc.reshape(len(self._f_dc), -1, 3)
        features_rest = self._f_rest.reshape(len(self._f_dc), -1, 3)
        return torch.cat((features_dc, features_rest), dim=1)
    
    def export(self, path):
        xyz = self._xyz
        normals = np.zeros_like(xyz)
        f_dc = self._f_dc
        f_rest = self._f_rest
        opacities = self._opacity
        scale = self._scale
        rotation = self._rot

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"Exported {len(self._xyz)} points to {path}")

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._f_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rot.shape[1]):
            l.append('rot_{}'.format(i))
        return l
