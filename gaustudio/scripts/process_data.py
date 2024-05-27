import click
from typing import Optional

@click.command()
@click.option('--dataset', '-d', type=str, default='colmap', help='Dataset name (polycam, mvsnet, nerf, scannet, waymo)')
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
@click.option('--output_dir', '-o', required=True, help='Path to the output directory')
@click.option('--initializer', default='colmap', type=str, help='Initializer name (colmap, loftr, dust3r, mvsplat, midas)')
@click.option('--overwrite', help='Overwrite existing files', is_flag=True)
@click.option('--w_mask', '-w', is_flag=True, help='Use mask')
def main(dataset: str, source_path: Optional[str], output_dir: Optional[str], initializer: str, overwrite: bool, w_mask: bool) -> None:
    """
    Main function to run the pipeline.

    Args:
        dataset (str): Name of the dataset.
        source_path (Optional[str]): Path to the dataset.
        output_dir (Optional[str]): Path to the output directory.
        initializer (str): Name of the initializer.
        overwrite (bool): Whether to overwrite existing files.
        with_mask (bool): Whether to use mask.
    """
    from gaustudio import datasets
    from gaustudio import models
    from gaustudio.pipelines import initializers

    dataset_config = {
        "name": dataset,
        "source_path": source_path,
        "images": "images",
        "w_mask": w_mask,
        "camera_number": 1,
        "resolution": 1,
    }

    dataset_instance = datasets.make(dataset_config)
    pcd = models.make("general_pcd")
    initializer_config = {"name": initializer, "workspace_dir": output_dir}
    initializer_instance = initializers.make(initializer_config)

    initializer_instance(pcd, dataset_instance, overwrite=overwrite)

if __name__ == '__main__':
    main()