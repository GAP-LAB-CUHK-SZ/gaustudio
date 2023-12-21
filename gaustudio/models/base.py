import numpy as np
from typing import Dict
from plyfile import PlyData, PlyElement

class BasePointCloud:
    def __init__(self, config: Dict[str, int]):
        self.config = config

    def load(self, ply_path: str):
        plydata = PlyData.read(ply_path)  
        self.num_points = plydata['vertex'].count

        for elem in self.config:
            if elem == 'xyz':
                self._xyz = np.stack((plydata.elements[0]['x'], 
                                     plydata.elements[0]['y'],
                                     plydata.elements[0]['z']), axis=1)
            
            elif elem == 'opacity':
                self._opacity = plydata.elements[0]['opacity'][..., np.newaxis]
            else:
                names = [n.name for n in plydata.elements[0].properties if n.name.startswith(elem)]
                names = sorted(names, key=lambda n: int(n.split('_')[-1]))
                
                assert len(names) == self.config[elem]
                
                data = np.zeros((self.num_points, len(names)))
                for i, name in enumerate(names):
                    data[:,i] = plydata.elements[0][name]
                setattr(self, '_'+elem, data)

        print(f"Loaded {self.num_points} points from {ply_path}")
        
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
        print(self._f_dc.shape)
        for i in range(self._f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        print(self._f_rest.shape)
        for i in range(self._f_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rot.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def get_dtype(self):
        dtype = []
        for elem in self.config:
            dtype.append((elem, self.config[elem], 'f4'))
        return dtype

if __name__ == "__main__":
    config = {
        "xyz": 3, 
        'opacity': 1,
        "f_dc": 3,
        "f_rest": 45,  
        "scale": 3,
        "rot": 4
    }

    pcd = BasePointCloud(config)