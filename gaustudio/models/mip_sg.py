from gaustudio.models.base import BasePointCloud
# from gaustudio.utils import build_scaling_rotation, strip_symmetric, build_covariance_from_scaling_rotation, build_covariance_from_scaling_rotation_and_rotation_matrix, build_covariance_from_scaling_rotation_and_rotation_
from gaustudio.models.utils import get_activation, build_covariance_from_scaling_rotation
from gaustudio import models

import torch

@models.register('mip_pcd')
class MipPointCloud(BasePointCloud):
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
    
    def __init__(self, config) -> None:
        super().__init__()
        self.config = {**self.default_conf, **config}
        self.setup()
        self.setup_functions()

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
            print(activation_function)
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
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_attribute["scale"]
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_opacity_with_3D_filter(self):
        opacity = get_activation(self.config["activations"]["opacity"])(self._opacity)
        # apply 3D filter
        scales = self.get_attribute["scale"]
        
        scales_square = torch.square(scales)
        det1 = scales_square[:,0] * scales_square[:,1] * scales_square[:,2] # det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square[:,0] * scales_after_square[:,1] * scales_after_square[:,2] # det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2 #0.2? hyper param?
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None] #compute s/v_k of the 3D filter in the paper

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
