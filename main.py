# Torch
import torch
import torch.distributed as dist
from Common.Src.DeviceWrapper import DeviceWrapper
from datetime import datetime

# Dataset
def main_worker(device, mode, n_device, datetime_now):
    device_wrapper : DeviceWrapper = DeviceWrapper(device, n_device, mode)

    if device_wrapper.is_multi_gpu_mode():
        torch.distributed.init_process_group(
            backend='nccl', 
            init_method='tcp://127.0.0.1:2568', 
            world_size=n_device, 
            rank=device)

    ########################################################################
    # TODO: import dynamically
    from Impl.MonocularDepthEstimation.Src.impl import MonocularDepthEstimationImpl
    impl = MonocularDepthEstimationImpl(device_wrapper = device_wrapper)
    impl.initialize()
    save_opt = 1
    n_epoch : int = 10 # TODO: from args
    assert 1 <= save_opt and save_opt <= n_epoch
    for epoch in range(n_epoch):
        impl.begin_epoch(epoch = epoch, n_epoch = n_epoch)

        impl.begin_train(epoch = epoch, n_epoch = n_epoch)
        impl.train(epoch = epoch, n_epoch = n_epoch)
        impl.end_train(epoch = epoch, n_epoch = n_epoch)

        impl.begin_test(epoch = epoch, n_epoch = n_epoch)
        impl.test(epoch = epoch, n_epoch = n_epoch)
        impl.end_test(epoch = epoch, n_epoch = n_epoch)

        impl.end_epoch(epoch = epoch, n_epoch = n_epoch)

        if (epoch + 1) % save_opt == 0:
            if device_wrapper.is_multi_gpu_mode():
                if device_wrapper.get() == 0:
                    model_dir, model_filename, model_dict = impl.check_point(epoch = epoch, n_epoch = n_epoch)
                    model_dir = model_dir / datetime_now
                    model_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model_dict, model_dir / model_filename)
                dist.barrier()
            else:
                model_dir, model_filename, model_dict = impl.check_point(epoch = epoch, n_epoch = n_epoch)
                model_dir = model_dir / datetime_now
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model_dict, model_dir / model_filename)

    impl.finalize()
    ########################################################################
    if device_wrapper.is_multi_gpu_mode():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    """ TODO : parse args using argparser
    --impl_path : the python file path which has a class inherited from BaseImpl class.
    --n_epoch
    --mode : force the mode.
    --save_period

    """
    datetime_now : str = datetime.now().strftime("%Y-%m-%d: %H:%M:%S")
    mode : str = DeviceWrapper.get_opt_device_mode()
    if mode == "cpu":
        main_worker("cpu", mode, -1, datetime_now)
    elif mode == "single-gpu":
        main_worker(torch.device("cuda"), mode, 1, datetime_now)
    else:
        n_gpus = torch.cuda.device_count()
        torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(mode, n_gpus, datetime_now,), join=True)