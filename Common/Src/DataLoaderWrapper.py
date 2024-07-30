from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from Common.Src.DeviceWrapper import DeviceWrapper

class DataLoaderWrapper():
    def __init__(self, dataset, device_wrapper : DeviceWrapper, n_batch_per_device : int):
        DeviceWrapper.check_if_valid_device_mode(device_wrapper.get_mode())
        self.__dataset = dataset
        self.__dataloader = None
        self.__sampler = None
        
        if device_wrapper.is_multi_gpu_mode():
            self.__sampler = DistributedSampler(dataset)
            self.__dataloader = DataLoader(dataset, batch_size=n_batch_per_device, num_workers=0, pin_memory=True, shuffle=False, sampler=self.__sampler)
        else:
            self.__dataloader = DataLoader(dataset, batch_size=n_batch_per_device, shuffle=True)
    
    def __iter__(self):
        for x, y in self.__dataloader:
            yield x, y

    def __len__(self):
        return len(self.__dataloader)
    
    def get(self):
        return self.__dataloader
    
    def set_epoch(self, epoch : int):
        self.__sampler.set_epoch(epoch)