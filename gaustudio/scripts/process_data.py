import click

@click.command()
@click.option('--source_path', '-s', help='path to the dataset')
@click.option('--output-dir', '-o', help='path to the output dir')
@click.option('--dataset', type=str, default='colmap', help='Dataset name (polycam, mvsnet, nerf, neus, scannet)')
@click.option('--initializer', type=str, help='Initializer name (colmap, loftr, midas, mvsnet, dust3r, pixelsplat)')
@click.option('--overwrite', help='Overwrite existing files', is_flag=True)
def main(source_path, output_dir, dataset, initializer, overwrite):
    from gaustudio import datasets
    dataset_config = { "name":dataset, "source_path": source_path, "images":"images", "resolution":-1, "data_device":"cuda", "eval": False}
    dataset = datasets.make(dataset_config)
    
    from gaustudio import models
    pcd = models.make("general_pcd")
    
    from gaustudio.pipelines import initializers
    initializer_config = {"name":initializer, "workspace_dir":output_dir}
    initializer = initializers.make(initializer_config)
    initializer(pcd, dataset, overwrite=overwrite)

if __name__ == '__main__':
    main()
