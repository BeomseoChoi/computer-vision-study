# Torch
import torch
import torch.distributed as dist
from Common.Src.DeviceWrapper import DeviceWrapper
from datetime import datetime
import argparse
import os

from pathlib import Path
import random
import numpy as np


def train(impl, device_wrapper, args):
    for epoch in range(args.n_epoch):
        impl.set_epoch(epoch = epoch, args = args)

        impl.begin_epoch(epoch = epoch, args = args)

        impl.begin_train(epoch = epoch, args = args)
        impl.train(epoch = epoch, args = args)
        impl.end_train(epoch = epoch, args = args)

        impl.begin_validate(epoch = epoch, args = args)
        impl.validate(epoch = epoch, args = args)
        impl.end_validate(epoch = epoch, args = args)

        impl.end_epoch(epoch = epoch, args = args)

        if (epoch + 1) % args.save_period == 0:
            if device_wrapper.is_multi_gpu_mode():
                if device_wrapper.is_main_device():
                    model_dict = impl.check_point(epoch = epoch, args = args)
                    model_dir = Path(args.root_dir) / "Model" / args.save_dir_name
                    model_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model_dict, model_dir / f"{epoch + 1}.pt")
                dist.barrier()
            else:
                model_dict = impl.check_point(epoch = epoch, args = args)
                model_dir = Path(args.root_dir) / "Model" / args.save_dir_name
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model_dict, model_dir / f"{epoch + 1}.pt")

def test(impl, device_wrapper : DeviceWrapper, args : dict):
    impl.begin_test(args = args)
    impl.test(args = args)
    impl.end_test(args = args)
    if device_wrapper.is_multi_gpu_mode():
        dist.barrier()

# Dataset
def main_worker(device, n_device, args):
    device_wrapper : DeviceWrapper = DeviceWrapper(device, n_device, args.device_mode)

    if device_wrapper.is_multi_gpu_mode():
        torch.distributed.init_process_group(
            backend='nccl', 
            init_method='tcp://127.0.0.1:12349', 
            world_size=n_device, 
            rank=device)

    ########################################################################
    # TODO: import dynamically
    from Impl.MonocularDepthEstimation.Src.impl import MonocularDepthEstimationImpl
    impl = MonocularDepthEstimationImpl(device_wrapper = device_wrapper, args = args)
    impl.initialize(args = args)

    if args.type == "train":
        train(impl, device_wrapper, args)
    elif args.type == "test":
        test(impl, device_wrapper, args)

    impl.finalize()
    ########################################################################
    if device_wrapper.is_multi_gpu_mode():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

def fix_seed(seed : int, deterministic : bool) -> None:
    # https://pytorch.org/docs/stable/notes/randomness.html

    """ DDP weight initialization.
    # https://discuss.pytorch.org/t/setting-seed-in-torch-ddp/126638
    # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    """
    import torch.nn as nn
    # torch.manual_seed(seed)
    linear = nn.Linear(5, 2)

    # torch.manual_seed(seed)
    linear2 = nn.Linear(5, 2)

    print(linear.weight)
    print(linear2.weight)
    """

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["train", "test"] ,help="training or test")
    parser.add_argument("root_dir", help="the root impl directory. The dir will have two directories: 1.<root_dir>/Src/, where the impl file exists and 2. <root_dir>/Model/, where the model will be saved in.")
    parser.add_argument("-i", "--impl_filename", default="impl.py")
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-d", "--device_mode", choices=["cpu", "single-gpu", "multi-gpu"], default=DeviceWrapper.get_opt_device_mode())
    parser.add_argument("-e", "--n_epoch", default=150)
    parser.add_argument("-p", "--save_period", default=1)
    parser.add_argument("-n", "--save_dir_name", default=datetime.now().strftime("%Y-%m-%d: %H:%M:%S"))
    parser.add_argument("-s", "--seed", default=1234)
    parser.add_argument("-c", "--cudnn_seed_fix", action="store_false")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args : dict = parse_arguments()
    assert 1 <= args.save_period and args.save_period <= args.n_epoch

    fix_seed(args.seed, args.cudnn_seed_fix)

    if args.device_mode == "cpu":
        main_worker("cpu", 1, args)
    elif args.device_mode == "single-gpu":
        main_worker(torch.device("cuda"), 1, args)
    else:
        n_gpus = torch.cuda.device_count()
        torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args,), join=True)