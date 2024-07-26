import torch.nn as nn

def get_activation_func(activation_func : str):
    activation_func = str.lower(activation_func)
    if activation_func == "relu":
        return nn.ReLU()
    elif activation_func == "sigmoid":
        nn.Sigmoid()
    else:
        assert False