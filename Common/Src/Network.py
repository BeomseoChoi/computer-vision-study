import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from Common.Src.DeviceWrapper import DeviceWrapper

class Network():
    def __init__(self, net, device_wrapper : DeviceWrapper, train_fn, test_fn):
        DeviceWrapper.check_if_valid_device_mode(device_wrapper.mode)
        if not callable(train_fn): raise RuntimeError("The argument 'train_fn' must be callable.")
        if not callable(test_fn): raise RuntimeError("The argument 'test_fn' must be callable.")

        self.device_wrapper = device_wrapper
        self.net = net
        self.ddp_net = None
        if device_wrapper.mode == "cpu":
            self.net.to(device_wrapper.device)
        elif device_wrapper.mode == "single-gpu":
            self.net.to(device_wrapper.device)
        elif device_wrapper.mode == "multi-gpu":
            torch.cuda.set_device(device_wrapper.device)
            self.net.to(device_wrapper.device)
            # 여기가 문제. DDP랑 Non-DDP를 구분해야한다. Get property를 만들어야 할 것 같다.
            self.ddp_net = DDP(self.net, device_ids=[device_wrapper.device]) 

        self.train_fn = train_fn
        self.test_fn = test_fn

    def train(self, *args, **kwargs):
        ret = 0
        if self.device_wrapper.mode == "multi-gpu":
            ret = self.train_fn(self.ddp_net, self.device_wrapper, *args, **kwargs)
        else:
            ret = self.train_fn(self.net, self.device_wrapper, *args, **kwargs)

        return ret
    
    def test(self, *args, **kwargs):
        ret = 0
        if self.device_wrapper.mode == "multi-gpu":
            ret = self.test_fn(self.ddp_net, self.device_wrapper, *args, **kwargs)
        else:
            ret = self.test_fn(self.net, self.device_wrapper, *args, **kwargs)

        return ret
