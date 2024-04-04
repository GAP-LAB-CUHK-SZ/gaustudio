from gaustudio.models.base import BasePointCloud
# from gaustudio.utils import build_scaling_rotation, strip_symmetric, build_covariance_from_scaling_rotation, build_covariance_from_scaling_rotation_and_rotation_matrix, build_covariance_from_scaling_rotation_and_rotation_
from gaustudio.models.utils import get_activation, build_covariance_from_scaling_rotation
from gaustudio import models

import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement

@models.register('scaffold_pcd')
class ScaffoldPointCloud(BasePointCloud):
    default_conf = {
        'sh_degree': 3,
        'attributes':  {
            "anchor": 3, 
            "offset": 3,
            "anchor_feat": 32,
            "opacity": 1,
            "scale": 3,
            "rot": 4
        },
        'activations':{
            "scale": "exp",
            "opacity": "sigmoid",
            "rot": "normalize"
        }
    }
    
    def __init__(self, config) -> None:
        super().__init__()
        self.config = {**self.default_conf, **config}
        self.setup()
        self.setup_functions()

        self.feat_dim = self.config["anchor_feat"]
        self.n_offsets = self.config["n_offsets"]
        self.voxel_size = self.config["voxel_size"]
        self.update_depth = self.config["update_depth"]
        self.update_init_factor = self.config["update_init_factor"]
        self.update_hierachy_factor = self.config["update_hierachy_factor"]
        self.use_feat_bank = self,config["use_feat_bank"]
        self.opacity_accum = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.anchor_demon = torch.empty(0)

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()


        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim+3+1, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim+3+1, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7*self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim+3+1, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()
        
        # TODO: Move resume to datasets
        resume_path = self.config.get('resume_path', None)
        if resume_path is not None:
            print(f"Resuming pointcloud")
            self.load(resume_path)

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def setup_functions(self):        
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.scaling_inverse_activation = torch.log
        self.inverse_opacity_activation = lambda x: torch.log(x/(1-x))

    def get_attribute(self, attribute):
        if attribute in self.config["activations"]:
            activation_function = get_activation(self.config["activations"][attribute])
            print(activation_function)
            return activation_function(getattr(self, '_'+attribute))
        elif attribute.startswith('mlp_'):
            return getattr(self, attribute)
        else:
            return getattr(self, '_'+attribute)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_attribute("scale"), scaling_modifier, self._rot)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    def export(self, path):
        anchor = self._anchor
        normals = np.zeros_like(anchor)
        offset = self._offset
        anchor_feat = self._anchor_feat
        opacities = self._opacity
        scale = self._scale
        rotation = self._rot

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
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

    def load_scaffold(self, ply_path: str):
        plydata = PlyData.read(ply_path)  
        self.num_points = plydata['vertex'].count

        for elem in self.config["attributes"]:
            if elem == 'anchor':
                self._anchor = np.stack((plydata.elements[0]['x'], 
                                     plydata.elements[0]['y'],
                                     plydata.elements[0]['z']), axis=1)
            
            elif elem == 'opacity':
                self._opacity = plydata.elements[0]['opacity'][..., np.newaxis]
            else:
                names = [n.name for n in plydata.elements[0].properties if n.name.startswith(elem)]
                names = sorted(names, key=lambda n: int(n.split('_')[-1]))
                
                assert len(names) == self.config["attributes"][elem]
                
                data = np.zeros((self.num_points, len(names)))   
                for i, name in enumerate(names):
                    data[:,i] = plydata.elements[0][name]
                if elem == 'offset':
                    data = data.reshape((self.num_points.shape[0], 3, -1)).transpose(1, 2).contiguous()
                setattr(self, '_'+elem, data)

        print(f"Loaded {self.num_points} anchors from {ply_path}")
