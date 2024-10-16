import numpy as np
from typing import Dict
from plyfile import PlyData, PlyElement
import torch.nn as nn
import torch

class BasePointCloud(nn.Module):
    def __repr__(self):
        properties = self.config["attributes"].keys()
        return f"{self.__class__.__name__}(num_points={self.num_points}, properties={properties})"

    def __init__(self, config, device=None) -> None:
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = {**self.default_conf, **config}
        self.setup(device)
        self.setup_functions()

    @torch.no_grad()
    def to(self, device):
        self.device = device
        for elem in self.config["attributes"]:
            if elem == 'xyz':
                self._xyz = self._xyz.to(device)
            elif elem == 'opacity':
                self._opacity = self._opacity.to(device)
            else:
                setattr(self, '_'+elem, getattr(self, '_'+elem).to(device))
        return self
    
    @property
    def get_center_and_size(self):
        import numpy as np
        _xyz = self._xyz.cpu().numpy()
        lower_bound = np.percentile(_xyz, 5, axis=0)
        upper_bound = np.percentile(_xyz, 95, axis=0)
        center = (lower_bound + upper_bound) / 2
        size = upper_bound - lower_bound
        return center, size
    
    @property
    def get_center(self):
        min_xyz, _ = torch.min(self._xyz, dim=0)
        max_xyz, _ = torch.max(self._xyz, dim=0)
        return (min_xyz + max_xyz) / 2
    
    def setup(self, device,  num_points = 0):
        self.device = device
        self.num_points = num_points
        for elem in self.config["attributes"]:
            dummy_data = torch.empty(num_points, device=device)
            setattr(self, '_'+elem, dummy_data)
    
    def setup_functions(self):
        pass
    
    def update(self, **args):
        for elem in self.config["attributes"]:
            if elem in args:
                setattr(self, '_'+elem, args[elem])
        self.num_points = self._xyz.shape[0]
        
    def create_from_attribute(self, **args):
        for elem in args:
            if args[elem] is None:
                continue
            tensor_value = torch.as_tensor(args[elem])
            setattr(self, '_' + elem, tensor_value)
            self.num_points = tensor_value.shape[0]
        self.config["attributes"] = list(args.keys())
        
    def load(self, ply_path: str):
        plydata = PlyData.read(ply_path)  
        self.num_points = plydata['vertex'].count

        for elem in self.config["attributes"]:
            if elem == 'xyz':
                xyz = np.stack((plydata.elements[0]['x'], 
                                     plydata.elements[0]['y'],
                                     plydata.elements[0]['z']), axis=1)
                self._xyz = torch.from_numpy(xyz).float().to(self.device)
                
            elif elem == 'opacity':
                opacity = plydata.elements[0]['opacity'][..., np.newaxis]
                self._opacity = torch.from_numpy(opacity).float().to(self.device)
                  
            elif elem == 'rgb':
                rgb = np.stack((plydata.elements[0]['red'],
                                plydata.elements[0]['green'],
                                plydata.elements[0]['blue']), axis=1)
                self._rgb = torch.from_numpy(rgb).float().to(self.device) / 255
            else:
                names = [n.name for n in plydata.elements[0].properties if n.name.startswith(elem)]
                names = sorted(names, key=lambda n: int(n.split('_')[-1]))
                if len(names) == 0:
                    continue
                
                
                data = np.zeros((self.num_points, len(names)))
                for i, name in enumerate(names):
                    data[:,i] = plydata.elements[0][name]
                setattr(self, '_'+elem, torch.from_numpy(data).float().to(self.device))

        print(f"Loaded {self.num_points} points from {ply_path}")
    
    def get_attribute(self, attribute):
        return getattr(self, '_'+attribute)
    
    def __add__(self, other):
        if not isinstance(other, BasePointCloud):
            raise TypeError("Unsupported operand type for +: '{}' and '{}'".format(type(self).__name__, type(other).__name__))
        
        # Create a new instance of the same class as self
        result = self.__class__(self.config, device=self.device)
        
        # Combine attributes
        for attr in self.config["attributes"]:
            self_attr = getattr(self, f'_{attr}')
            other_attr = getattr(other, f'_{attr}', None)
            
            if other_attr is not None:
                # Concatenate the attributes if both point clouds have them
                combined_attr = torch.cat([self_attr, other_attr], dim=0)
            else:
                # If the other point cloud doesn't have this attribute, pad with zeros
                padding = torch.zeros(other.num_points, self_attr.shape[1], device=self.device)
                combined_attr = torch.cat([self_attr, padding], dim=0)
            
            setattr(result, f'_{attr}', combined_attr)
        
        # Update the number of points
        result.num_points = self.num_points + other.num_points
        
        return result