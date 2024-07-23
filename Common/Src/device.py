import torch

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

def print_device_info(device : torch.device):
    device_info = get_device_info(device)
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
