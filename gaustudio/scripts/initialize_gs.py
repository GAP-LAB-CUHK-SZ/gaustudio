import click
from typing import Optional
import os
from gaustudio.utils.misc import load_config

@click.command()
@click.argument('init', default='colmap', type=str)
@click.option('--dataset', '-d', type=str, default='colmap', help='Dataset name (polycam, mvsnet, nerf, scannet, waymo)')
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
@click.option('--output_dir', '-o', required=True, help='Path to the output directory')
@click.option('--overwrite', help='Overwrite existing files', is_flag=True)
@click.option('--w_mask', '-w', is_flag=True, help='Use mask')
@click.option('--resolution', '-r', default=1, type=int, help='Resolution')
@click.option('--model', '-m', help='path to model')
@click.option('--config', help='path to config file', default='vanilla')
def main(init: str,dataset: str, source_path: Optional[str], output_dir: Optional[str], 
        overwrite: bool, w_mask: bool, resolution: int, model: str, config: str) -> None:
    from gaustudio import datasets
    from gaustudio import models
    from gaustudio.pipelines import initializers

    dataset_config = {
        "name": dataset,
        "source_path": source_path,
        "w_mask": w_mask,
        "camera_number": 1,
    }

    dataset = datasets.make(dataset_config)
    dataset.all_cameras = [_camera.downsample_scale(resolution) for _camera in dataset.all_cameras]
    
    # parse YAML config to OmegaConf
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, '../configs', config+'.yaml')
    config = load_config(config_path)
    pcd = models.make(config.model.pointcloud)
    
    initializer_config = {"name": init, "workspace_dir": os.path.join(output_dir, 'data'), 
                          "model_path":model}
    initializer_instance = initializers.make(initializer_config)

    initializer_instance(pcd, dataset, overwrite=overwrite)

    model_dir = os.path.join(output_dir, 'point_cloud', 'iteration_0')
    os.makedirs(model_dir, exist_ok=True)
    pcd.export(os.path.join(model_dir, 'point_cloud.ply'))
    dataset.export(os.path.join(output_dir, 'cameras.json'))
    
if __name__ == '__main__':
    main()