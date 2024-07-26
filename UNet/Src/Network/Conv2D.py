import torch
import torch.nn as nn
from Common.Src.activation_funciton import get_activation_func

class Conv2D(nn.Module):
    def __init__(self, 
                 n_in_channel : int, 
                 n_out_channel : int, 
                 kernel, # int or (int, int)
                 stride, # int or (int, int)
                 padding, # int or (int, int) or str("valid" or "same")
                 n_layer : int = 1, 
                 on_batch_norm : bool = True, 
                 activation_func : str = "relu"):
        assert type(kernel) is int or type(kernel) is (int, int)
        assert type(stride) is int or type(stride) is (int, int)
        assert type(padding) is int or type(padding) is (int, int) or (type(padding) is str and str.lower(padding) in ["valid", "same"])

        super(Conv2D, self).__init__()
        self.n_in_channel = n_in_channel
        self.n_out_channel = n_out_channel
        self.n_layer = n_layer
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.on_batch_norm = on_batch_norm
        self.batch_norms = []
        for _ in range(n_layer):
            self.batch_norms.append(nn.BatchNorm2d(n_out_channel))

        self.conv2ds = [ nn.Conv2d(self.n_in_channel, self.n_out_channel, self.kernel, self.stride, self.padding)]
        for i in range(n_layer - 1):
            self.conv2ds.append(nn.Conv2d(self.n_out_channel, self.n_out_channel, self.kernel, self.stride, self.padding))

        self.activation_funcs = []
        for _ in range(n_layer):
            self.activation_funcs.append(get_activation_func(activation_func))

        self.seq = nn.Sequential()
        for i in range(n_layer):
            self.seq.append(self.conv2ds[i])
            if self.on_batch_norm:
                self.seq.append(self.batch_norms[i])
            self.seq.append(self.activation_funcs[i])

    def forward(self, x : torch.Tensor):
        x = self.seq(x)
        # for i in range(self.n_layer):
        #     x = self.conv2ds[i](x)
        #     if self.on_batch_norm:
        #         x = self.batch_norms[i](x)
        #     x = self.activation_funcs[i](x)
        
        return x