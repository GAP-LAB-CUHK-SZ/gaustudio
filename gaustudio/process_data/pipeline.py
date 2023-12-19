import click
from gaustudio.process_data.polycam_utils import PolycamDataset
from gaustudio.process_data.colmap_converter import ColmapConverter
@click.command()
@click.option('--data', type=str, help='Path to data directory')
@click.option('--output-dir', type=str, help='Path to output directory')
def main(data, output_dir):
    dataset = PolycamDataset(data)
    converter = ColmapConverter(dataset, output_dir)
    converter.preprocess() 
    converter.process()
    converter.postprocess()

if __name__ == '__main__':
    main()
