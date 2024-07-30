# Torch
import torch
import torch.distributed
from Common.Src.DeviceWrapper import DeviceWrapper

# Dataset
def main_worker(device, mode, n_device):
    # 멀티프로세싱 통신 규약 정의
    if mode == "multi-gpu":
        torch.distributed.init_process_group(
            backend='nccl', 
            init_method='tcp://127.0.0.1:2568', 
            world_size=n_device, 
            rank=device)

    device_wrapper : DeviceWrapper = DeviceWrapper(device, n_device, mode)

    
    
    depth_estimation(device_wrapper)

    if mode == "multi-gpu":
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    """ TODO : parse args using argparser
    --impl_path : the python file path which has a class inherited from BaseImpl class.
    

    """
    mode : str = DeviceWrapper.get_device_mode()
    if mode == "cpu":
        main_worker("cpu", mode, -1)
    elif mode == "single-gpu":
        main_worker(torch.device("cuda"), mode, 1)
    else:
        n_gpus = torch.cuda.device_count()
        torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(mode, n_gpus,), join=True)