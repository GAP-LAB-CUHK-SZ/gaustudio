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
@click.option('--pcd', type=click.Choice(['dust3r', 'combined']), default='dust3r', help='Point cloud to use: dust3r or combined (hloc+dust3r)')
def main(dataset: str, source_path: Optional[str], output_dir: Optional[str], 
        overwrite: bool, w_mask: bool, resolution: int, model: str, pcd: bool) -> None:
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
    dust3r_pcd = models.make("general_pcd")
    initializer_config = {"name": 'dust3r', "workspace_dir": os.path.join(output_dir, 'data'), 
                          "model_path":model}
    initializer_instance = initializers.make(initializer_config)
    initializer_instance(dust3r_pcd, dataset, overwrite=overwrite)
    
    if pcd == 'combined':
        hloc_pcd = models.make("general_pcd")
        hloc_initializers = initializers.make({"name": 'hloc', "workspace_dir": output_dir})
        hloc_initializers(hloc_pcd, dataset, overwrite=overwrite)
        final_pcd = hloc_pcd + dust3r_pcd
    else:
        final_pcd = dust3r_pcd
    
    final_pcd.export(os.path.join(output_dir, 'sparse', '0', 'points3D.ply'))

if __name__ == '__main__':
    main()