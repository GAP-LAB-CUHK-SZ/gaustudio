import argparse
import os
import torch
from tqdm import tqdm
from random import randint
from gaustudio import models, datasets, renderers
from gaustudio.pipelines import initializers
import open3d as o3d
import click

@click.command()
@click.option('--mesh', '-m', default=None, help='path to the mesh')
@click.option('--output_path', '-o', required=True, help='Path to the output path')
def main(mesh: str, output_path: str) -> None:
    _mesh = o3d.io.read_triangle_mesh(mesh)
    _gaussians = models.make({"name": "vanilla_pcd", "sh_degree": 1}).to("cuda")
    initializers.make({"name":"mesh", "n_gaussians_per_surface_triangle": 3})(_gaussians, _mesh)
    _gaussians.export(output_path)
    
    