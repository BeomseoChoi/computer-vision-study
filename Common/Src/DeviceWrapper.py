import torch

class DeviceWrapper():
    def __init__(self, device, n_device : int, mode : str):
        self.__device = device
        self.n_device : int = n_device
        self.__mode : str = mode
        
    def get(self) -> torch.device:
        return self.__device
    
    def get_mode(self) -> str:
        return self.__mode

    def is_cpu_mode(self) -> bool:
        return self.__mode == "cpu"
    
    def is_single_gpu_mode(self) -> bool:
        return self.__mode == "single-gpu"

    def is_multi_gpu_mode(self) -> bool:
        return self.__mode == "multi-gpu"

    @staticmethod
    def is_valid_device_mode(mode : str):
        return mode == "cpu" or mode == "single-gpu" or mode == "multi-gpu"

    @staticmethod
    def check_if_valid_device_mode(mode : str):
        if not DeviceWrapper.is_valid_device_mode(mode) : raise RuntimeError(f"Invalid mode. Got {mode}.")

    @staticmethod
    def get_opt_device_mode() -> str:
        """
        https://discuss.pytorch.org/t/single-machine-single-gpu-distributed-best-practices/169243
        """
        mode = ""
        n_gpu : int = torch.cuda.device_count()

        if not torch.cuda.is_available():
            mode = "cpu"
        else:
            if n_gpu == 1:
                mode = "single-gpu"
            elif n_gpu >= 2:
                mode = "multi-gpu"
            else:
                raise RuntimeError(f"Invalid GPU count. Expected >= 0, but got {n_gpu}")
        
        return mode

    @staticmethod
    def get_device_info(device):
        device_info = {}
        
        if device.type == 'cuda' and torch.cuda.is_available():
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            device_info['device_type'] = 'cuda'
            device_info['device_name'] = torch.cuda.get_device_name(device_index)
            device_info['device_capability'] = torch.cuda.get_device_capability(device_index)
            device_info['device_memory'] = torch.cuda.get_device_properties(device_index).total_memory
        elif device.type == 'cpu':
            device_info['device_type'] = 'cpu'
            device_info['device_name'] = 'CPU'
            device_info['device_capability'] = None
            device_info['device_memory'] = None
        else:
            raise ValueError("Unsupported device type or CUDA not available")

        return device_info

    @staticmethod
    def print_device_info(device : torch.device):
        device_info = DeviceWrapper.get_device_info(device)
        if device_info['device_type'] == 'cuda':
            print(f"Device Type: CUDA")
            print(f"Device Name: {device_info['device_name']}")
            print(f"Compute Capability: {device_info['device_capability'][0]}.{device_info['device_capability'][1]}")
            print(f"Total Memory: {device_info['device_memory'] / (1024 ** 3):.2f} GB")
        elif device_info['device_type'] == 'cpu':
            print("Device Type: CPU")
            print("Device Name: CPU")
        else:
            print("Unknown device type")
