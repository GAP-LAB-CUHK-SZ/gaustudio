import numpy as np
from typing import Dict
from plyfile import PlyData, PlyElement
import torch.nn as nn
import torch

class BasePointCloud:
    def setup(self):
        for elem in self.config["attributes"]:
            dummy_data = torch.empty(0)
            setattr(self, '_'+elem, dummy_data)
            
    def load(self, ply_path: str):
        plydata = PlyData.read(ply_path)  
        self.num_points = plydata['vertex'].count

        for elem in self.config["attributes"]:
            if elem == 'xyz':
                xyz = np.stack((plydata.elements[0]['x'], 
                                     plydata.elements[0]['y'],
                                     plydata.elements[0]['z']), axis=1)
                self._xyz = torch.from_numpy(xyz).float()
                
            elif elem == 'opacity':
                opacity = plydata.elements[0]['opacity'][..., np.newaxis]
                self._opacity = torch.from_numpy(opacity).float()
                
            else:
                names = [n.name for n in plydata.elements[0].properties if n.name.startswith(elem)]
                names = sorted(names, key=lambda n: int(n.split('_')[-1]))
                
                assert len(names) == self.config["attributes"][elem]
                
                data = np.zeros((self.num_points, len(names)))
                for i, name in enumerate(names):
                    data[:,i] = plydata.elements[0][name]
                setattr(self, '_'+elem, torch.from_numpy(data).float())

        print(f"Loaded {self.num_points} points from {ply_path}")