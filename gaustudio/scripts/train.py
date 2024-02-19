import sys
import argparse
import os
import time
import logging
from datetime import datetime

from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', default='gaustudio/configs/vanilla.yaml')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--source_path', '-s', required=True, help='path to the dataset')
    args, extras = parser.parse_known_args()
    
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))
    
    from gaustudio.utils.misc import load_config
    from gaustudio import models, datasets, renderers
    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    assert args.source_path, "Please specify an training dataset"
    config.dataset.source_path = args.source_path

    pcd = models.make(config.model.pointcloud)
    renderer = renderers.make(config.renderer)
    dataset = datasets.make(config.dataset)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch:[c.to("cuda") for c in batch])
    for batch in train_loader:
        for camera in batch:
            render_pkg = renderer.render(camera, pcd)
    
if __name__ == '__main__':
    main()