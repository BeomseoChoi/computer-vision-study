import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from Common.Src.DeviceWrapper import DeviceWrapper
from pathlib import Path

class NetworkWrapper():
    def __init__(self, net, device_wrapper : DeviceWrapper):
        DeviceWrapper.check_if_valid_device_mode(device_wrapper.get_mode())

        self.__device_wrapper = device_wrapper
        self.__net = net
        self.__ddp_net = None

        if device_wrapper.is_cpu_mode():
            self.__net.to(device_wrapper.get())
        elif device_wrapper.is_single_gpu_mode():
            self.__net.to(device_wrapper.get())
        elif device_wrapper.is_multi_gpu_mode():
            torch.cuda.set_device(device_wrapper.get())
            self.__net.to(device_wrapper.get())
            self.__ddp_net = DDP(self.__net, device_ids=[device_wrapper.get()]) 

    def __call__(self, *args):
        net = self.get()
        return net(*args)

    def get(self):
        if self.__device_wrapper.is_multi_gpu_mode():
            return self.__ddp_net
        
        return self.__net

    def train(self):
        net = self.get()
        net.train()
    
    def eval(self):
        net = self.get()
        net.eval()
    
    def get_state_dict(self) -> dict:
        return self.__net.state_dict()
