import click
from typing import Optional
import os
from gaustudio.utils.misc import load_config

def _process_mesh_based_initializer(initializer_obj, pcd_model, mesh_path, dataset_obj, overwrite):
    """
    Process mesh-based initializers (mesh and voxel).
    
    Args:
        initializer_obj: The initializer object
        pcd_model: Point cloud model
        mesh_path: Path to the mesh file
        dataset_obj: Dataset object
        overwrite: Whether to overwrite existing files
        
    Returns:
        The processed point cloud
    """
    import open3d as o3d
    
    # Load and validate mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh from {mesh_path} or mesh is empty")
    
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Process mesh with initializer
    return initializer_obj(pcd_model, mesh, dataset=dataset_obj, overwrite=overwrite)

def _run_pose_initializer(output_dir, config, overwrite, dataset_obj):
    """
    Run pose initializer with fallback mechanism.
    
    Args:
        output_dir: Output directory path
        config: Configuration file path
        overwrite: Whether to overwrite existing files
        dataset_obj: Dataset object
        
    Returns:
        The final point cloud from pose initialization
    """
    from gaustudio import models
    from gaustudio.pipelines import initializers
    
    pcd_model = models.make("general_pcd")
    
    # Try hloc first
    print("Trying hloc pose initializer...")
    hloc_config = {"name": "hloc", "workspace_dir": output_dir}
    
    if config:
        if not os.path.exists(config):
            raise FileNotFoundError(f"Configuration file not found: {config}")
        custom_config = load_config(config)
        hloc_config.update(custom_config.get('pose_initializer', {}))
    
    try:
        hloc_initializer_obj = initializers.make(hloc_config)
        final_pcd = hloc_initializer_obj(pcd_model, dataset_obj, overwrite=overwrite)
        print("Pose initialization completed using hloc")
        return final_pcd
    except Exception as e:
        print(f"Error with hloc pose initializer: {e}")
        print("Falling back to colmap pose initializer...")
        
        # Fallback to colmap
        try:
            colmap_config = {"name": 'colmap', "workspace_dir": output_dir}
            if config:
                custom_config = load_config(config)
                colmap_config.update(custom_config.get('pose_initializer', {}))
            colmap_initializer = initializers.make(colmap_config)
            final_pcd = colmap_initializer(pcd_model, dataset_obj, overwrite=overwrite)
            print("Pose initialization completed using colmap")
            return final_pcd
        except Exception as colmap_e:
            print(f"Error with colmap pose initializer: {colmap_e}")
            raise Exception("Both hloc and colmap pose initializers failed")

def _export_point_cloud(final_pcd, output_dir, initializer_name):
    """
    Export the final point cloud to the output directory.
    
    Args:
        final_pcd: The point cloud to export
        output_dir: Output directory path
        initializer_name: Name of the initializer used
    """
    output_path = os.path.join(output_dir, 'sparse', '0', 'points3D.ply')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_pcd.export(output_path)
    print(f"Point cloud exported to: {output_path}")
    print(f"Geometry initialization completed using {initializer_name}")

def _validate_inputs(source_path, initializer, mesh_path):
    """
    Validate input parameters.
    
    Args:
        source_path: Path to the dataset
        initializer: Geometry initializer name
        mesh_path: Path to mesh file
    """
    # Validate source_path (always required)
    if not source_path:
        raise ValueError("--source_path is required")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source path not found: {source_path}")

    # Validate mesh path for mesh and voxel geometry initializers
    if initializer in ['mesh', 'voxel']:
        if not mesh_path:
            raise ValueError(f"{initializer} geometry initializer requires --mesh_path to be specified")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        print(f"Using mesh file: {mesh_path}")

def _create_dataset(dataset, source_path, w_mask, resolution):
    """
    Create and configure the dataset.
    
    Args:
        dataset: Dataset name
        source_path: Path to the dataset
        w_mask: Mask directory name
        resolution: Resolution scale
        
    Returns:
        Configured dataset object
    """
    from gaustudio import datasets
    
    # Setup dataset configuration
    dataset_config = {
        "name": dataset,
        "source_path": source_path,
        "masks": w_mask,
        "w_mask": w_mask is not None,
        "camera_number": 1,
    }

    # Create dataset and adjust camera resolution
    dataset_obj = datasets.make(dataset_config)
    dataset_obj.all_cameras = [_camera.downsample_scale(resolution) for _camera in dataset_obj.all_cameras]
    
    return dataset_obj

@click.command()
@click.option('--dataset', '-d', type=str, default='colmap', help='Dataset name (polycam, mvsnet, nerf, scannet, waymo)')
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
@click.option('--output_dir', '-o', required=True, help='Path to the output directory')
@click.option('--overwrite', help='Overwrite existing files', is_flag=True)
@click.option('--w_mask', default=None, help='mask dir name')
@click.option('--resolution', '-r', default=1, type=int, help='Resolution')
@click.option('--initializer', '-i', type=click.Choice(['depth', 'VisualHull', 'mesh', 'voxel', 'tsdf']), 
              help='Geometry initializer to use for point cloud generation')
@click.option('--mesh_path', '-m', help='Path to mesh file (required for mesh and voxel geometry initializers)')
@click.option('--config', '-c', help='Path to configuration file for advanced initializer settings')
def main(dataset: str, source_path: Optional[str], output_dir: Optional[str], 
        overwrite: bool, w_mask: str, resolution: int, 
        initializer: Optional[str], mesh_path: Optional[str], config: Optional[str]) -> None:
    """
    Initialize Gaussian Splatting with pose and optional geometry initialization.
    
    This script performs a two-step initialization process:
    1. Pose initialization using hloc (with colmap fallback)
    2. Optional geometry initialization using various methods
    """
    from gaustudio import models
    from gaustudio.pipelines import initializers

    # Step 0: Validate inputs
    _validate_inputs(source_path, initializer, mesh_path)

    # Step 1: Create and configure dataset
    dataset_obj = _create_dataset(dataset, source_path, w_mask, resolution)

    # Step 2: Run pose initializer (try hloc first, fallback to colmap)
    _run_pose_initializer(output_dir, config, overwrite, dataset_obj)

    # Step 3: Run geometry initializer if specified
    if initializer:
        print(f"Step 2: Running {initializer} geometry initializer...")
        
        # Create point cloud model for geometry initialization
        pcd_model = models.make("vanilla_pcd")
        
        # Setup geometry initializer configuration
        geometry_config = {"name": initializer, "workspace_dir": output_dir}
        
        # Apply custom configuration if provided
        if config:
            custom_config = load_config(config)
            geometry_config.update(custom_config.get('initializer', {}))
        
        try:
            initializer_obj = initializers.make(geometry_config)
            
            # Handle mesh-based initializers (mesh and voxel)
            if initializer in ['mesh', 'voxel']:
                final_pcd = _process_mesh_based_initializer(
                    initializer_obj, pcd_model, mesh_path, dataset_obj, overwrite
                )
            else:
                final_pcd = initializer_obj(pcd_model, dataset_obj, overwrite=overwrite)
            
            # Export the final point cloud
            _export_point_cloud(final_pcd, output_dir, initializer)
            
        except Exception as e:
            print(f"Error with {initializer} geometry initializer: {e}")
            raise e
    else:
        print("No geometry initializer specified. Only pose initialization was performed.")

    print("Initialization completed successfully!")

if __name__ == '__main__':
    main()