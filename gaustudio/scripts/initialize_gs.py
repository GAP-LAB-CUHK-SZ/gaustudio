import click
from typing import Optional
import os
from gaustudio.utils.misc import load_config

@click.command()
@click.option('--dataset', '-d', type=str, default='colmap', help='Dataset name (polycam, mvsnet, nerf, scannet, waymo)')
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
@click.option('--output_dir', '-o', required=True, help='Path to the output directory')
@click.option('--overwrite', help='Overwrite existing files', is_flag=True)
@click.option('--w_mask', default=None, help='mask dir name')
@click.option('--resolution', '-r', default=1, type=int, help='Resolution')
@click.option('--model', '-m', help='path to model')
@click.option('--initializer', '-i', type=click.Choice(['colmap', 'hloc', 'depth','VisualHull']), 
              default='colmap', help='Initializer to use for point cloud generation')
@click.option('--config', '-c', help='Path to configuration file for advanced initializer settings')
def main(dataset: str, source_path: Optional[str], output_dir: Optional[str], 
        overwrite: bool, w_mask: str, resolution: int, model: str, initializer: str, 
        config: Optional[str]) -> None:
    from gaustudio import datasets
    from gaustudio import models
    from gaustudio.pipelines import initializers

    # Load configuration from file if provided
    initializer_config = {"name": initializer, "workspace_dir": output_dir}
    
    if config:
        if not os.path.exists(config):
            raise FileNotFoundError(f"Configuration file not found: {config}")
        custom_config = load_config(config)
        initializer_config.update(custom_config.get('initializer', {}))
    
    # Add specific configuration options
    if initializer == 'dust3r' and prune_bg:
        initializer_config['prune_bg'] = True
    
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
    
    # Create point cloud model and initializer
    if model:
        # Load custom model if specified
        pcd_model = models.make({"name": "general_pcd", "path": model})
    else:
        pcd_model = models.make("general_pcd")
    
    # Create and run the specified initializer
    try:
        initializer_obj = initializers.make(initializer_config)
        print(f"Using {initializer} initializer...")
        final_pcd = initializer_obj(pcd_model, dataset_obj, overwrite=overwrite)
    except Exception as e:
        print(f"Error with {initializer} initializer: {e}")
        print("Falling back to colmap initializer...")
        # Fallback to colmap if the specified initializer fails
        colmap_config = {"name": 'colmap', "workspace_dir": output_dir}
        colmap_initializer = initializers.make(colmap_config)
        final_pcd = colmap_initializer(pcd_model, dataset_obj, overwrite=overwrite)

    # Export the final point cloud
    output_path = os.path.join(output_dir, 'sparse', '0', 'points3D.ply')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_pcd.export(output_path)
    
    print(f"Point cloud exported to: {output_path}")
    print(f"Initialization completed using {initializer} initializer")

if __name__ == '__main__':
    main()