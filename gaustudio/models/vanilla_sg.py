from gaustudio.models.base import BasePointCloud
from gaustudio.models.utils import get_activation, build_covariance_from_scaling_rotation
from gaustudio import models
from gaustudio.utils.sh_utils import RGB2SH
import numpy as np
from plyfile import PlyElement, PlyData
import torch

def calculate_dist2_python(xyz):
    from scipy.spatial import KDTree
    points_np = xyz.cpu().float().numpy()
    dists, _ = KDTree(points_np).query(points_np, k=4)
    mean_dists = (dists[:, 1:] ** 2).mean(axis=1)
    return torch.tensor(mean_dists, dtype=xyz.dtype, device=xyz.device)

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

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

    def create_from_attribute(self, xyz, rgb=None, scale=None, rot=None, opacity=None, **args):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._xyz = torch.tensor(xyz, dtype=torch.float32).to(device)
        self.num_points = xyz.shape[0]

        if rgb is None:
            rgb = torch.ones_like(self._xyz)
        fused_color = RGB2SH(torch.tensor(rgb, dtype=torch.float32).to(device))
        self._f_dc = fused_color.unsqueeze(-1).transpose(1, 2).contiguous()
        f_rest_shape = (fused_color.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3)
        self._f_rest = torch.zeros(f_rest_shape, dtype=torch.float32).to(device)

        if scale is None:
            dist2 = self.calculate_dist2()(self._xyz)
            self._scale = torch.log(torch.sqrt(dist2 + 1e-7))[..., None].repeat(1, 3)
        else:
            self._scale = torch.tensor(scale, dtype=torch.float32).to(device)

        if rot is None:
            self._rot = torch.zeros((self._xyz.shape[0], 4), dtype=torch.float32).to(device)
            self._rot[:, 0] = 1
        else:
            self._rot = torch.tensor(rot, dtype=torch.float32).to(device)

        if opacity is None:
            self._opacity = inverse_sigmoid(0.1 * torch.ones((self._xyz.shape[0], 1), dtype=torch.float32).to(device))
        else:
            self._opacity = torch.tensor(opacity, dtype=torch.float32).to(device)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_attribute("scale"), scaling_modifier, self._rot)

    @property
    def get_features(self):
        features_dc = self._f_dc.reshape(len(self._f_dc), -1, 3)
        features_rest = self._f_rest.reshape(len(self._f_dc), -1, 3)
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.get_attribute("opacity")
    
    @property
    def get_scaling(self):
        return self.get_attribute("scale")
    
    @property
    def get_rotation(self):
        return self.get_attribute("rot")
    
    @property
    def get_xyz(self):
        return self.get_attribute("xyz")
    
    @property
    def get_num_points(self):
        return self.num_points
    
    @property
    def get_features_dc(self):
        return self.get_attribute("f_dc").reshape(self.num_points, -1, 3)
    
    @property
    def get_features_rest(self):
        return self.get_attribute("f_rest").reshape(self.num_points, -1, 3)
    
    def calculate_dist2(self):
        try:
            from simple_knn._C import distCUDA2
            dist2 = distCUDA2
        except ImportError:
            dist2 = calculate_dist2_python
        return dist2

    def export(self, path):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._f_dc.reshape(len(self._f_dc), -1, 3).transpose(1, 2).flatten(start_dim=1).detach().cpu().numpy()
        f_rest = self._f_rest.reshape(len(self._f_dc), -1, 3).transpose(1, 2).flatten(start_dim=1).detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scale.detach().cpu().numpy()
        rotation = self._rot.detach().cpu().numpy()
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
        if len(self._f_dc.shape) == 2:
            _f_dc_len = self._f_dc.shape[1]
        else:
            _f_dc_len = self._f_dc.shape[1]*self._f_dc.shape[2]
        if len(self._f_rest.shape) == 2:
            _f_rest_len = self._f_rest.shape[1]
        else:
            _f_rest_len = self._f_rest.shape[1]*self._f_rest.shape[2]
            
        for i in range(_f_dc_len):
            l.append('f_dc_{}'.format(i))
        for i in range(_f_rest_len):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rot.shape[1]):
            l.append('rot_{}'.format(i))
        return l
