from Common.Src.NetworkWrapper import NetworkWrapper
from Common.Src.DataLoaderWrapper import DataLoaderWrapper
from Common.Src.DeviceWrapper import DeviceWrapper

class BaseImpl():
    def __init__(self, *args, **kwargs):
        self.__device_wrapper : DeviceWrapper = kwargs["device_wrapper"]
        self.__network_wrapper : NetworkWrapper = None
        self.__training_dataloader : DataLoaderWrapper = None
        self.__validation_dataloader : DataLoaderWrapper = None
        self.__test_dataloader : DataLoaderWrapper = None

    def get_device_wrapper(self) -> DeviceWrapper:
        return self.__device_wrapper

    def wrap_network(self, network) -> None:
        self.__network_wrapper = NetworkWrapper(network, self.__device_wrapper)
        return self.__network_wrapper

    def wrap_training_dataloader(self, dataloader : DataLoaderWrapper, n_batch_per_device : int) -> None:
        self.__training_dataloader = DataLoaderWrapper(dataloader, self.__device_wrapper, n_batch_per_device)
        return self.__training_dataloader
    
    def wrap_validation_dataloader(self, dataloader : DataLoaderWrapper, n_batch_per_device : int) -> None:
        self.__validation_dataloader = DataLoaderWrapper(dataloader, self.__device_wrapper, n_batch_per_device)
        return self.__validation_dataloader

    def wrap_test_dataloader(self, dataloader : DataLoaderWrapper, n_batch_per_device : int) -> None:
        self.__test_dataloader = DataLoaderWrapper(dataloader, self.__device_wrapper, n_batch_per_device)
        return self.__test_dataloader

    def initialize(self, *args, **kwargs) -> None:
        pass

    def set_epoch(self, *args, **kwargs) -> None:
        # TODO: This should be invisible to a derived class.
        epoch : int = kwargs["epoch"]
        if self.__device_wrapper.is_multi_gpu_mode():
            if self.__training_dataloader is not None:
                self.__training_dataloader.set_epoch(epoch)
            if self.__validation_dataloader is not None:
                self.__validation_dataloader.set_epoch(epoch)

    def finalize(self, *args, **kwargs) -> None:
        pass

    def begin_epoch(self, *args, **kwargs) -> None:
        pass

    def end_epoch(self, *args, **kwargs) -> None:
        pass
    
    def begin_train(self, *args, **kwargs) -> None:
        pass
    def train(self, *args, **kwargs) -> None:
        pass
    def end_train(self, *args, **kwargs) -> None:
        pass
    
    def begin_validate(self, *args, **kwargs) -> None:
        pass
    def validate(self, *args, **kwargs) -> None:
        pass
    def end_validate(self, *args, **kwargs) -> None:
        pass

    def begin_test(self, *args, **kwargs) -> None:
        pass
    def test(self, *args, **kwargs) -> None:
        pass
    def end_test(self, *args, **kwargs) -> None:
        pass

    def check_point(self, *args, **kwargs) -> None:
        pass

