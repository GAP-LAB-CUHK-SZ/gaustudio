from gaustudio.models.base import BasePointCloud
from gaustudio import models

@models.register('vanilla_pcd')
class VanillaPointCloud(BasePointCloud):
    default_conf = {
        'sh_degree': 3,
        'attributes':  
            {
            "xyz": 3, 
            'opacity': 1,
            "f_dc": 3,
            "f_rest": 45,
            "scale": 3,
            "rot": 4
            }
    }
    
    def __init__(self, config) -> None:
        super().__init__()
        self.config = {**self.default_conf, **config}
        self.setup()

        # TODO: Move resume to datasets
        resume_path = self.config.get('resume_path', None)
        if resume_path is not None:
            print(f"Resuming pointcloud")
            self.load(resume_path)

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
