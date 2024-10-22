import torch
import numpy as np
import trimesh
import open3d as o3d
from gaustudio import models
from gaustudio.models.base import BasePointCloud
from gaustudio.utils.graphics_utils import PSR2Mesh, DPSR
from typing import Dict

@models.register('sap_pcd')
class ShapeAsPoints(BasePointCloud):
    default_conf = {
        "attributes": {
            "xyz": 3,
            "normals": 3
        },
        "dpsr_res": 256,
        "dpsr_sig": 2,
        "dpsr_scale": True,
        "dpsr_shift": True,
        "dpsr_weighted": False,
        "num_sample": 100000
    }

    def setup_functions(self):
        self.dpsr = DPSR(
            res=(self.config["dpsr_res"], self.config["dpsr_res"], self.config["dpsr_res"]),
            sig=self.config["dpsr_sig"],
            scale=self.config["dpsr_scale"],
            shift=self.config["dpsr_shift"],
            weighted=self.config["dpsr_weighted"]
        ).to(self.device)
        self.psr2mesh = PSR2Mesh.apply
    
    def transform(self, verts, center, scale, inverse=False):
        if inverse:
            out = verts * 2. - 1.
            out = out * scale + center
        else:
            out = (verts - center) / scale
            out = (out + 1.) / 2.
        return out

    @classmethod
    def from_mesh(cls, mesh_path: str, config: Dict = None):
        if config is None:
            config = cls.default_conf

        mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
        sap_pc = cls(config)
        return sap_pc._from_mesh(mesh.vertices, mesh.faces, mesh.face_normals)

    @classmethod
    def from_pointcloud(cls, points: np.ndarray, normals: np.ndarray = None, config: Dict = None):
        if config is None:
            config = cls.default_conf

        sap_pc = cls(config)
        
        center = np.mean(points, axis=0)
        scale = np.abs(points - center).max() * 1.2
        
        transformed_points = sap_pc.transform(points, center, scale)
        
        if normals is None:
            # If normals are not provided, you might want to estimate them
            # This is a placeholder - you may want to implement a more sophisticated normal estimation
            normals = np.zeros_like(points)
        
        return sap_pc._from_point(transformed_points, normals, center, scale)

    @classmethod
    def from_mesh(cls, mesh_path: str, config: Dict = None):
        if config is None:
            config = cls.default_conf

        mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
        sap_pc = cls(config)
        return sap_pc._from_mesh(mesh.vertices, mesh.faces, mesh.face_normals)
    
    def create_from_attribute(self, xyz, **args):
        xyz = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        
        if 'faces' in args and 'face_normals' in args:
            faces = torch.tensor(args['faces'], dtype=torch.int64, device=self.device)
            face_normals = torch.tensor(args['face_normals'], dtype=torch.float32, device=self.device)
            return self._from_mesh(xyz, faces, face_normals)
        elif 'normals' in args:
            normals = torch.tensor(args['normals'], dtype=torch.float32, device=self.device)
            center = xyz.mean(dim=0)
            scale = (xyz - center).abs().max() * 1.2
            transformed_points = self.transform(xyz, center, scale)
            return self._from_point(transformed_points, normals, center, scale)
        else:
            raise ValueError("Invalid input. Expected either 'faces' and 'face_normals' or 'normals' in the arguments.")

    def create_from_attribute(self, xyz, **args):
        if 'faces' in args and 'face_normals' in args:
            return self._from_mesh(xyz, args['faces'], args['face_normals'])
        elif 'normals' in args:
            center = np.mean(xyz, axis=0)
            scale = np.abs(xyz - center).max() * 1.2
            transformed_points = self.transform(xyz, center, scale)
            return self._from_point(transformed_points, args['normals'], center, scale)
        else:
            raise ValueError("Invalid input. Expected either 'faces' and 'face_normals' or 'normals' in the arguments.")
        
    def _from_mesh(self, vertices, faces, face_normals):
        center = vertices.mean(dim=0)
        scale = (vertices - center).abs().max() * 1.2

        transformed_verts = self.transform(vertices, center, scale)
        tmp_mesh = trimesh.Trimesh(transformed_verts.cpu().numpy(), faces.cpu().numpy(), 
                                   face_normals=face_normals.cpu().numpy(), process=False, maintain_order=True)

        points, face_idx = trimesh.sample.sample_surface_even(tmp_mesh, self.config["num_sample"])
        normals = tmp_mesh.face_normals[face_idx]

        points = torch.tensor(points, dtype=torch.float32, device=self.device)
        normals = torch.tensor(normals, dtype=torch.float32, device=self.device)

        return self._from_point(points, normals, center, scale)

    @classmethod
    def from_o3d_pointcloud(cls, o3d_pcd: o3d.geometry.PointCloud, config: Dict = None):
        if config is None:
            config = cls.default_conf

        sap_pc = cls(config)

        # Extract points and normals from the Open3D point cloud
        points = np.asarray(o3d_pcd.points)
        normals = np.asarray(o3d_pcd.normals)

        # Convert to PyTorch tensors and move to the correct device
        points = torch.tensor(points, dtype=torch.float32, device=sap_pc.device)
        normals = torch.tensor(normals, dtype=torch.float32, device=sap_pc.device)

        center = points.mean(dim=0)
        scale = (points - center).abs().max() * 1.2

        transformed_points = sap_pc.transform(points, center, scale)

        if len(normals) == 0:
            # If normals are not available in the o3d point cloud, estimate them
            o3d_pcd.estimate_normals()
            normals = torch.tensor(np.asarray(o3d_pcd.normals), dtype=torch.float32, device=sap_pc.device)

        return sap_pc._from_point(transformed_points, normals, center, scale)

    def _from_point(self, points, normals, center, scale):
        points = torch.log(points / (1 - points))  # inverse sigmoid

        self.update(xyz=points, normals=normals)
        self.center = center
        self.scale = scale

        return self

    def to(self, device):
        super().to(device)
        self.dpsr = self.dpsr.to(device)
        return self

    def generate_mesh(self):
        points = torch.sigmoid(self._xyz.unsqueeze(0))
        normals = self._normals.unsqueeze(0)

        psr_grid = self.dpsr(points, normals).squeeze(1)
        psr_grid = torch.tanh(psr_grid)

        v, faces = self.psr2mesh(psr_grid)
        vertices = self.transform(v, self.center, self.scale, True).squeeze(0)
        faces = faces.squeeze(0).int()

        return vertices, faces, v
    
    def to_o3d_mesh(self) -> o3d.geometry.TriangleMesh:
        vertices, faces, _ = self.generate_mesh()
        
        # Convert to numpy arrays
        vertices_np = vertices.cpu().numpy()
        faces_np = faces.cpu().numpy()
        
        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
        mesh.triangles = o3d.utility.Vector3iVector(faces_np)
        
        # Compute vertex normals
        mesh.compute_vertex_normals()
        
        return mesh