import sys
import argparse
import os
import time
import logging
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('-s', default=None, help='path to the dataset')
    parser.add_argument('-m', default=None, help='path to the model')
    
    args, extras = parser.parse_known_args()
    
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))
    
    from gaustudio.utils.misc import load_config
    from gaustudio import models, datasets
    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)  
    config.dataset.source_path = args.s
    
    pcd = models.make(config.model.pointcloud.name, config.model.pointcloud)
    