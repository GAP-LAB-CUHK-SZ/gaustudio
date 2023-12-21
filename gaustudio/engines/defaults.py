import os
import sys
import argparse
import random
from datetime import datetime
import numpy as np
import torch

from gaustudio.utils.config import Config, DictAction
from gaustudio.utils import comm

def get_random_seed():
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    return seed


def set_seed(seed=None):
    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    
def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Examples:
    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
    Change some config options:
        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        # default="tcp://127.0.0.1:{}".format(port),
        default="auto",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    return parser

def default_config_parser(file_path, options):
    # config name protocol: dataset_name/model_name-exp_name
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))

    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    return cfg

def default_setup(cfg):
    # scalar by world size
    world_size = comm.get_world_size()
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    # settle random seed
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed * cfg.num_worker_per_gpu + rank
    set_seed(seed)
    return cfg