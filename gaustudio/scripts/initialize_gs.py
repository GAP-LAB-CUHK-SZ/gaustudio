import click
from typing import Optional
import os
from gaustudio.utils.misc import load_config

@click.command()
@click.option('--dataset', '-d', type=str, default='colmap', help='Dataset name (polycam, mvsnet, nerf, scannet, waymo)')
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
@click.option('--output_dir', '-o', required=True, help='Path to the output directory')
@click.option('--overwrite', help='Overwrite existing files', is_flag=True)
@click.option('--w_mask', '-w', is_flag=True, help='Use mask')
@click.option('--resolution', '-r', default=1, type=int, help='Resolution')
@click.option('--model', '-m', help='path to model')
def main(dataset: str, source_path: Optional[str], output_dir: Optional[str], 
        overwrite: bool, w_mask: bool, resolution: int, model: str) -> None:
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
    pcd = models.make("general_pcd")
    
    hloc_initializers = initializers.make({"name": 'hloc', "workspace_dir": output_dir})
    hloc_initializers(pcd, dataset, overwrite=overwrite)

    initializer_config = {"name": 'dust3r', "workspace_dir": os.path.join(output_dir, 'data'), 
                          "model_path":model}
    initializer_instance = initializers.make(initializer_config)
    initializer_instance(pcd, dataset, overwrite=overwrite)
    
    pcd.export(os.path.join(output_dir, 'sparse', '0', 'points3D.ply'))

if __name__ == '__main__':
    main()