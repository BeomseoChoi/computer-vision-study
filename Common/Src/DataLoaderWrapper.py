from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from Common.Src.DeviceWrapper import DeviceWrapper

class DataLoaderWrapper():
    def __init__(self, dataset, device_wrapper : DeviceWrapper, n_batch_per_device : int):
        DeviceWrapper.check_if_valid_device_mode(device_wrapper.mode)
        self.dataset = dataset
        self.dataloader = None
        self.sampler = None
        if device_wrapper.mode == "multi-gpu":
            self.sampler = DistributedSampler(dataset)
            self.dataloader = DataLoader(dataset, batch_size=n_batch_per_device, num_workers=0, pin_memory=True, shuffle=False, sampler=self.sampler)
        else:
            self.dataloader = DataLoader(dataset, batch_size=n_batch_per_device, shuffle=True)
        
