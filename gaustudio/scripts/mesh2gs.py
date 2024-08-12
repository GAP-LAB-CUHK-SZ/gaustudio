import argparse
import os
import torch
from tqdm import tqdm
from random import randint
from gaustudio import models, datasets, renderers
from gaustudio.pipelines import initializers
import open3d as o3d
import numpy as np
import click

def get_colors_from_texture(texture, uvs):
    # Convert Open3D Image to numpy array
    texture_np = np.asarray(texture) / 255
    
    # Get image dimensions
    height, width = texture_np.shape[:2]
    
    # Scale and flip UV coordinates
    uvs_scaled = np.copy(uvs)
    uvs_scaled[:, 0] *= (width - 1)
    uvs_scaled[:, 1] *= (height - 1)
    
    # Convert to integer coordinates
    uvs_int = np.round(uvs_scaled).astype(int)
    
    # Clip coordinates to ensure they're within bounds
    uvs_int[:, 0] = np.clip(uvs_int[:, 0], 0, width - 1)
    uvs_int[:, 1] = np.clip(uvs_int[:, 1], 0, height - 1)
    
    # Get colors for all UVs at once
    colors = texture_np[uvs_int[:, 1], uvs_int[:, 0]]
    
    return colors

@click.command()
@click.option('--mesh', '-m', required=True, help='Path to the input mesh')
@click.option('--output_path', '-o', required=True, help='Path to the output file')
@click.option('--target_faces', '-f', default=30000, help='Target number of faces for remeshing')
def main(mesh: str, output_path: str, target_faces: int) -> None:
    # Load the mesh
    _mesh = o3d.io.read_triangle_mesh(mesh, True)

    # Ensure the mesh has vertex colors
    if _mesh.has_textures():
        # If the mesh has a texture, we need to map it to vertex colors
        _mesh.compute_vertex_normals()
        vertices = np.asarray(_mesh.vertices)
        triangles = np.asarray(_mesh.triangles)
        
        # Assuming the first texture is the main one
        texture = _mesh.textures[0]
        uv = np.asarray(_mesh.triangle_uvs).reshape(-1, 2)
        
        colors = get_colors_from_texture(texture, uv)
        
        # Reshape colors to match the original triangle UV shape
        colors = colors.reshape(-1, 3, 3)
        
        vertex_colors = np.zeros_like(vertices)
        for i, triangle in enumerate(triangles):
            for j in range(3):
                vertex_colors[triangle[j]] += colors[i, j]
        
        # Normalize vertex colors
        vertex_count = np.zeros(len(vertices), dtype=int)
        for triangle in triangles:
            for vertex_idx in triangle:
                vertex_count[vertex_idx] += 1
        vertex_colors /= vertex_count[:, np.newaxis]
        
        print(vertex_colors)
        _mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    elif not _mesh.has_vertex_colors():
        print("No texture or face colors found. Using a uniform color.")
        _mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Set a default gray color

    # Create and initialize Gaussians
    _gaussians = models.make({"name": "vanilla_pcd", "sh_degree": 1}).to("cuda")
    initializers.make({"name": "mesh", "n_gaussians_per_surface_triangle": 3})(_gaussians, _mesh)
    
    # Export the result
    _gaussians.export(output_path)
    print(f"Gaussians exported to {output_path}")

if __name__ == "__main__":
    main()