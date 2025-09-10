#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
import numpy as np
from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.base import BaseInitializer
import open3d as o3d
import torch
import math

def inverse_sigmoid(x):
    """Compute the inverse sigmoid function: log(x / (1 - x))"""
    return np.log(x / (1 - x))

def normal2rotation(n):
    """
    Construct a rotation matrix from surface normals.
    
    Args:
        n: Surface normals tensor of shape (N, 3)
        
    Returns:
        Quaternion representations of rotation matrices
        
    Note: Adopted from https://github.com/turandai/gaussian_surfels/blob/main/utils/general_utils.py
    """
    # Normalize input normals
    n = torch.nn.functional.normalize(n)
    
    # Create orthogonal basis vectors
    w0 = torch.tensor([[1, 0, 0]]).expand(n.shape)
    R0 = w0 - torch.sum(w0 * n, -1, True) * n
    R0 *= torch.sign(R0[:, :1])
    R0 = torch.nn.functional.normalize(R0)
    R1 = torch.cross(n, R0)
    
    R1 *= torch.sign(R1[:, 1:2]) * torch.sign(n[:, 2:])
    R = torch.stack([R0, R1, n], -1)
    q = rotmat2quaternion(R)

    return q

def rotmat2quaternion(R, normalize=False):
    """
    Convert rotation matrices to quaternions.
    
    Args:
        R: Rotation matrices of shape (N, 3, 3)
        normalize: Whether to normalize the output quaternions
        
    Returns:
        Quaternions of shape (N, 4)
    """
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr) / 2
    
    q = torch.stack([
        r,
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * r),
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * r)
    ], -1)
    
    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1)
    return q


@initializers.register('mesh')
class MeshInitializer(BaseInitializer):
    """
    Initialize Gaussian splats from triangle mesh surfaces.
    
    This initializer places Gaussian splats on mesh triangles using barycentric coordinates.
    The number of Gaussians per triangle and their positioning can be configured.
    
    Code adapted from: https://github.com/Anttwo/SuGaR/blob/main/sugar_scene/sugar_model.py
    """
    
    # Number of Gaussians to place per mesh triangle
    n_gaussians_per_surface_triangle = 1
    
    def __init__(self, initializer_config):
        """
        Initialize the mesh-based Gaussian initializer.
        
        Args:
            initializer_config: Configuration object for the initializer
        """
        super().__init__(initializer_config)
        self._setup_barycentric_coordinates()
    
    def _setup_barycentric_coordinates(self):
        """Setup barycentric coordinates and circle radius based on number of Gaussians per triangle."""
        if self.n_gaussians_per_surface_triangle == 1:
            # Single Gaussian at triangle centroid
            self.surface_triangle_circle_radius = 1. / 2. / np.sqrt(3.)
            self.surface_triangle_bary_coords = torch.tensor(
                [[1/3, 1/3, 1/3]],
                dtype=torch.float32,
            )[..., None]
        elif self.n_gaussians_per_surface_triangle == 3:
            # Three Gaussians positioned towards each vertex
            self.surface_triangle_circle_radius = 1. / 2. / (np.sqrt(3.) + 1.)
            self.surface_triangle_bary_coords = torch.tensor(
                [[1/2, 1/4, 1/4],
                [1/4, 1/2, 1/4],
                [1/4, 1/4, 1/2]],
                dtype=torch.float32,
            )[..., None]
        elif self.n_gaussians_per_surface_triangle == 4:
            # Four Gaussians: one at center, three towards vertices
            self.surface_triangle_circle_radius = 1 / (4. * np.sqrt(3.))
            self.surface_triangle_bary_coords = torch.tensor(
                [[1/3, 1/3, 1/3],  # Center
                [2/3, 1/6, 1/6],   # Towards vertex 0
                [1/6, 2/3, 1/6],   # Towards vertex 1
                [1/6, 1/6, 2/3]],  # Towards vertex 2
                dtype=torch.float32,
            )[..., None]  # Shape: (n_gaussians_per_face, 3, 1)
        elif self.n_gaussians_per_surface_triangle == 6:
            # Six Gaussians: three towards vertices, three towards edge midpoints
            self.surface_triangle_circle_radius = 1 / (4. + 2.*np.sqrt(3.))
            self.surface_triangle_bary_coords = torch.tensor(
                [[2/3, 1/6, 1/6],   # Towards vertex 0
                [1/6, 2/3, 1/6],   # Towards vertex 1
                [1/6, 1/6, 2/3],   # Towards vertex 2
                [1/6, 5/12, 5/12], # Towards edge 0-1
                [5/12, 1/6, 5/12], # Towards edge 0-2
                [5/12, 5/12, 1/6]], # Towards edge 1-2
                dtype=torch.float32,
            )[..., None]
    
    def __call__(self, model, mesh, dataset=None, overwrite=False):
        """
        Initialize the model with Gaussians placed on the mesh surface.
        
        Args:
            model: Point cloud model to initialize
            mesh: Open3D triangle mesh
            dataset: Dataset object (optional)
            overwrite: Whether to overwrite existing data
            
        Returns:
            Initialized model
        """
        self.mesh = mesh
        self.mesh.compute_vertex_normals()
        return self.build_model(model)

    def build_model(self, model):
        """
        Build the Gaussian model from mesh data.
        
        Args:
            model: Point cloud model to populate
            
        Returns:
            Populated model with Gaussian attributes
        """
        # Extract mesh data
        faces = torch.tensor(np.array(self.mesh.triangles))
        vertex_points = torch.tensor(np.array(self.mesh.vertices)).float()
        vertex_colors = torch.tensor(np.array(self.mesh.vertex_colors)).float()
        vertex_normals = torch.tensor(np.array(self.mesh.vertex_normals)).float()
        
        has_color = vertex_colors.shape[0] > 0
        
        # Get per-face vertex data
        faces_verts = vertex_points[faces]    # Shape: (n_faces, 3, 3)
        faces_normals = vertex_normals[faces] # Shape: (n_faces, 3, 3)
        
        # Compute Gaussian positions using barycentric coordinates
        points = self._compute_gaussian_positions(faces_verts)
        
        # Compute surface normals at Gaussian positions
        points_normal = self._compute_surface_normals(faces_normals)
        
        # Convert normals to rotations
        rotations = normal2rotation(points_normal)
        
        # Compute colors if available
        colors = self._compute_colors(faces, vertex_colors, has_color)
        
        # Compute scales based on triangle edge lengths
        scales = self._compute_scales(faces_verts)
        
        # Set opacity (all Gaussians start fully opaque)
        opacity = inverse_sigmoid(np.ones((points.shape[0], 1)))
        
        # Create model from computed attributes
        model.create_from_attribute(
            xyz=points, 
            rgb=colors, 
            scale=scales, 
            opacity=opacity, 
            rot=rotations
        )

        return model
        
    def _compute_gaussian_positions(self, faces_verts):
        """Compute 3D positions of Gaussians using barycentric coordinates."""
        # Shape: (n_faces, n_gaussians_per_face, 3, 3) -> (n_faces, n_gaussians_per_face, 3)
        points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]
        points = points.sum(dim=-2)  # Sum over the 3 vertices
        return points.reshape(-1, 3)  # Flatten to (total_gaussians, 3)
        
    def _compute_surface_normals(self, faces_normals):
        """Compute surface normals at Gaussian positions."""
        # Interpolate normals using barycentric coordinates
        points_normal = faces_normals[:, None] * self.surface_triangle_bary_coords[None]
        points_normal = points_normal.sum(dim=-2)
        points_normal = points_normal.reshape(-1, 3)
        return torch.nn.functional.normalize(points_normal, dim=-1)
        
    def _compute_colors(self, faces, vertex_colors, has_color):
        """Compute colors at Gaussian positions."""
        if has_color:
            faces_colors = vertex_colors[faces]  # Shape: (n_faces, 3, 3)
            colors = faces_colors[:, None] * self.surface_triangle_bary_coords[None]
            colors = colors.sum(dim=-2)  # Interpolate using barycentric coordinates
            return colors.reshape(-1, 3)  # Shape: (total_gaussians, 3)
        else:
            return None
            
    def _compute_scales(self, faces_verts):
        """Compute scale values based on triangle edge lengths."""
        # Compute minimum edge length for each triangle
        edge_lengths = (faces_verts - faces_verts[:, [1, 2, 0]]).norm(dim=-1)
        min_edge_lengths = edge_lengths.min(dim=-1)[0]
        
        # Scale based on triangle size and circle radius
        scales = min_edge_lengths * self.surface_triangle_circle_radius
        scales = scales.clamp_min(0.)  # Ensure positive scales
        
        # Expand to all Gaussians and create 3D scales (x, y, z)
        scales = scales.reshape(len(faces_verts), -1, 1).expand(-1, self.n_gaussians_per_surface_triangle, 2)
        scales = scales.clone().reshape(-1, 2)
        
        # Add zero z-scale (flat Gaussians on surface)
        scales = torch.cat([scales, torch.zeros((scales.shape[0], 1))], dim=-1)
        
        # Convert to log space for Gaussian representation
        return torch.log(scales * 2 + 1e-7)

@initializers.register('voxel')
class VoxelInitializer(BaseInitializer):
    """
    Initialize Gaussian splats from mesh voxelization.
    
    This initializer creates a voxel grid from the input mesh and places
    Gaussian splats at voxel centers that intersect with the mesh.
    """
    
    def __init__(self, initializer_config):
        """
        Initialize the voxel-based Gaussian initializer.
        
        Args:
            initializer_config: Configuration object containing voxel_size parameter
        """
        super().__init__(initializer_config)
        self.voxel_size = getattr(initializer_config, 'voxel_size', 1/256)
        
    def __call__(self, model, mesh, dataset=None, overwrite=False):
        """
        Initialize the model with Gaussians placed at voxel centers.
        
        Args:
            model: Point cloud model to initialize
            mesh: Open3D triangle mesh
            dataset: Dataset object (optional)
            overwrite: Whether to overwrite existing data
            
        Returns:
            Initialized model
        """
        self.mesh = mesh
        self.mesh.compute_vertex_normals()
        return self.build_model(model)
        
    def build_model(self, model):
        """
        Build the Gaussian model from voxelized mesh data.
        
        Args:
            model: Point cloud model to populate
            
        Returns:
            Populated model with Gaussian attributes
        """
        # Normalize mesh to unit cube for consistent voxelization
        normalized_mesh, scale, center = self._normalize_mesh()
        
        # Create voxel grid from normalized mesh
        voxel_centers = self._create_voxel_grid(normalized_mesh)
        
        if len(voxel_centers) == 0:
            raise ValueError("No voxels generated from mesh")
        
        # Transform voxel centers back to original coordinate system
        points = self._transform_to_original_coordinates(voxel_centers, scale, center)
        
        # Generate Gaussian attributes
        colors = self._interpolate_colors_from_mesh(points)
        scales = self._compute_voxel_scales(scale, len(points))
        opacity = self._compute_opacity(len(points))
        rotations = self._generate_random_rotations(len(points))
        
        # Create model from computed attributes
        model.create_from_attribute(
            xyz=points, 
            rgb=colors, 
            scale=scales, 
            opacity=opacity, 
            rot=rotations
        )
        
        return model
        
    def _normalize_mesh(self):
        """
        Normalize mesh to fit within a unit cube centered at origin.
        
        Returns:
            Tuple of (normalized_mesh, scale_factor, original_center)
        """
        vertices = np.asarray(self.mesh.vertices)
        
        # Compute bounding box
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        
        # Normalize vertices to [-0.5, 0.5] range
        normalized_vertices = (vertices - center) / scale
        normalized_vertices = np.clip(normalized_vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Create normalized mesh
        normalized_mesh = o3d.geometry.TriangleMesh()
        normalized_mesh.vertices = o3d.utility.Vector3dVector(normalized_vertices)
        normalized_mesh.triangles = self.mesh.triangles
        normalized_mesh.vertex_colors = self.mesh.vertex_colors
        normalized_mesh.compute_vertex_normals()
        
        return normalized_mesh, scale, center
        
    def _create_voxel_grid(self, normalized_mesh):
        """
        Create voxel grid from normalized mesh.
        
        Args:
            normalized_mesh: Mesh normalized to unit cube
            
        Returns:
            Array of voxel center coordinates in normalized space
        """
        # Create voxel grid within normalized bounds
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            normalized_mesh, 
            voxel_size=self.voxel_size, 
            min_bound=(-0.5, -0.5, -0.5), 
            max_bound=(0.5, 0.5, 0.5)
        )
        
        # Extract voxel centers
        voxel_centers = []
        for voxel in voxel_grid.get_voxels():
            grid_index = voxel.grid_index
            voxel_center = voxel_grid.get_voxel_center_coordinate(grid_index)
            voxel_centers.append(voxel_center)
        
        return np.array(voxel_centers) if voxel_centers else np.empty((0, 3))
        
    def _transform_to_original_coordinates(self, voxel_centers, scale, center):
        """Transform voxel centers back to original mesh coordinate system."""
        points = voxel_centers * scale + center
        return torch.tensor(points).float()
        
    def _interpolate_colors_from_mesh(self, points):
        """
        Interpolate colors from mesh vertices if vertex colors are available.
        
        Args:
            points: Tensor of 3D points
            
        Returns:
            Color tensor or None if no vertex colors available
        """
        has_color = len(np.asarray(self.mesh.vertex_colors)) > 0
        if not has_color:
            return None
            
        colors = []
        for point in points:
            try:
                # Find closest point on mesh surface
                closest_point_idx = self.mesh.get_closest_point_on_triangle_mesh(point.numpy())
                if len(closest_point_idx) > 0:
                    vertex_color = np.asarray(self.mesh.vertex_colors)[closest_point_idx[1]]
                    colors.append(vertex_color)
                else:
                    colors.append([0.5, 0.5, 0.5])  # Default gray
            except:
                colors.append([0.5, 0.5, 0.5])  # Fallback to gray
                
        return torch.tensor(np.array(colors)).float()
        
    def _compute_voxel_scales(self, original_scale, num_points):
        """
        Compute scale values for Gaussians based on voxel size.
        
        Args:
            original_scale: Scale factor from mesh normalization
            num_points: Number of Gaussian points
            
        Returns:
            Log-space scale tensor of shape (num_points, 3)
        """
        # Set scales to be slightly smaller than voxel size
        voxel_scale = self.voxel_size * original_scale * 0.8  # 80% of voxel size
        scales = torch.full((num_points, 3), voxel_scale)
        return torch.log(scales + 1e-7)
        
    def _compute_opacity(self, num_points):
        """Compute opacity values for all Gaussians."""
        return inverse_sigmoid(np.ones((num_points, 1)))
        
    def _generate_random_rotations(self, num_points):
        """
        Generate random rotation quaternions for Gaussians.
        
        Note: Could be improved by using surface normals if needed.
        """
        rotations = torch.randn(num_points, 4)
        return torch.nn.functional.normalize(rotations, dim=-1)


@initializers.register('tsdf')
class TsdfInitializer(MeshInitializer):
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        self.voxel_size = getattr(initializer_config, 'voxel_size', 0.02)
        self.sdf_trunc = getattr(initializer_config, 'sdf_trunc', 0.04)
        self.max_depth = getattr(initializer_config, 'max_depth', 5.0)
        self.downsample_scale = max(1, int(getattr(initializer_config, 'downsample_scale', 1)))

    def __call__(self, model, dataset, overwrite=False):
        mesh = self._fuse_tsdf_mesh(dataset)
        self.mesh = mesh
        self.mesh.compute_vertex_normals()
        return self.build_model(model)

    def _fuse_tsdf_mesh(self, dataset):
        """Iterate dataset, integrate RGB-D with poses into TSDF, and extract a triangle mesh."""
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(self.voxel_size),
            sdf_trunc=float(self.sdf_trunc),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        # Iterate dataset and integrate frames
        for camera in dataset:
            cam = camera
            # Optional downsample to save memory and speed up
            if self.downsample_scale > 1 and hasattr(cam, 'downsample_scale'):
                try:
                    cam = cam.downsample_scale(self.downsample_scale)
                except Exception:
                    pass

            # Require depth
            if getattr(cam, 'depth', None) is None:
                continue

            # Convert color and depth to Open3D images
            color_np = self._to_numpy_color(cam)
            depth_np = self._to_numpy_depth(cam)
            if depth_np is None or color_np is None:
                continue

            color_o3d = o3d.geometry.Image(color_np)
            depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))

            # Camera intrinsics
            intrinsic = self._build_intrinsic_from_camera(cam, depth_np.shape)

            # Camera extrinsics (world-to-camera)
            extrinsic = self._build_extrinsic_w2c(cam)
            if extrinsic is None:
                # Skip if pose is unavailable
                continue

            # RGBD (depth in meters)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=float(self.max_depth),
                convert_rgb_to_intensity=False,
            )

            # Integrate
            volume.integrate(rgbd, intrinsic, extrinsic)

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def _to_numpy_color(self, cam):
        try:
            img = cam.image
            if hasattr(img, 'detach'):
                img = img.detach().cpu().numpy()
            else:
                img = np.asarray(img)

            # Expect HxWx3, value range [0,1] or [0,255]
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                # Possibly CxHxW -> HxWxC
                img = np.transpose(img, (1, 2, 0))

            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=2)

            if img.dtype != np.uint8:
                # Assume float [0,1]
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

            if img.shape[2] == 4:
                img = img[:, :, :3]

            return img
        except Exception:
            return None

    def _to_numpy_depth(self, cam):
        try:
            depth = cam.depth
            if hasattr(depth, 'detach'):
                depth = depth.detach().cpu().numpy()
            else:
                depth = np.asarray(depth)

            # Expect HxW, meters
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            depth = depth.astype(np.float32)

            # Clamp invalid values, truncate far depths
            depth[~np.isfinite(depth)] = 0
            if self.max_depth is not None:
                depth = np.where(depth > float(self.max_depth), 0.0, depth)
            depth = np.maximum(depth, 0.0)
            return depth
        except Exception:
            return None

    def _build_intrinsic_from_camera(self, cam, depth_hw):
        h, w = depth_hw
        # Prefer explicit fx/fy/cx/cy
        fx = getattr(cam, 'fx', None)
        fy = getattr(cam, 'fy', None)
        cx = getattr(cam, 'cx', None)
        cy = getattr(cam, 'cy', None)

        if fx is None or fy is None:
            # Fallback: compute from FoV
            FoVx = getattr(cam, 'FoVx', None)
            FoVy = getattr(cam, 'FoVy', None)
            if FoVx is not None and FoVy is not None:
                fx = w / (2.0 * math.tan(float(FoVx) / 2.0))
                fy = h / (2.0 * math.tan(float(FoVy) / 2.0))
            else:
                # Last-resort fallback to avoid crash
                fx = fy = 0.5 * (w + h)

        if cx is None or cy is None:
            pp_ndc = getattr(cam, 'principal_point_ndc', None)
            if pp_ndc is not None and len(pp_ndc) >= 2:
                try:
                    cx = float(pp_ndc[0]) * w
                    cy = float(pp_ndc[1]) * h
                except Exception:
                    cx, cy = w * 0.5, h * 0.5
            else:
                cx, cy = w * 0.5, h * 0.5

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=int(w), height=int(h), fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy)
        )
        return intrinsic

    def _build_extrinsic_w2c(self, cam):
        # Build a 4x4 world-to-camera matrix.
        # Try extrinsics/w2c, then inv(c2w), then (R,T).
        mat = getattr(cam, 'extrinsics', None)
        if mat is None:
            mat = getattr(cam, 'w2c', None)
        if mat is None:
            c2w = getattr(cam, 'c2w', None)
            if c2w is not None:
                try:
                    if hasattr(c2w, 'detach'):
                        c2w = c2w.detach().cpu().numpy()
                    else:
                        c2w = np.asarray(c2w)
                    mat = np.linalg.inv(c2w)
                except Exception:
                    mat = None
        if mat is None:
            R = getattr(cam, 'R', None)
            T = getattr(cam, 'T', None)
            if R is not None and T is not None:
                try:
                    if hasattr(R, 'detach'):
                        R = R.detach().cpu().numpy()
                    else:
                        R = np.asarray(R)
                    if hasattr(T, 'detach'):
                        T = T.detach().cpu().numpy()
                    else:
                        T = np.asarray(T)
                    mat = np.eye(4, dtype=np.float64)
                    mat[:3, :3] = R
                    mat[:3, 3] = T.reshape(-1)[:3]
                except Exception:
                    mat = None

        if mat is None:
            return None

        if hasattr(mat, 'detach'):
            mat = mat.detach().cpu().numpy()
        mat = np.asarray(mat, dtype=np.float64)

        # Ensure shape 4x4
        if mat.shape == (3, 4):
            tmp = np.eye(4, dtype=np.float64)
            tmp[:3, :4] = mat
            mat = tmp
        elif mat.shape == (4, 4):
            pass
        else:
            return None

        return mat
    
