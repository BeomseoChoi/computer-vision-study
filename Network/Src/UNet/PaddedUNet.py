import torch
import torch.nn as nn
from UNet.Src.Network.Conv2D import Conv2D

class PaddedUNet(nn.Module):
    def __init__(self, n_in_channel, n_out_channel):
        super(PaddedUNet, self).__init__()

        self.n_in_channel = n_in_channel
        self.n_out_channel = n_out_channel

        # Downsampling
        self.conv2d_down_0 : Conv2D = Conv2D(self.n_in_channel, 64, 3, 1, "same", 2)
        self.max_pool_0 : nn.MaxPool2d = nn.MaxPool2d(2, 2)

        self.conv2d_down_1 : Conv2D = Conv2D(64, 128, 3, 1, "same", 2)
        self.max_pool_1 : nn.MaxPool2d = nn.MaxPool2d(2, 2)

        self.conv2d_down_2 : Conv2D = Conv2D(128, 256, 3, 1, "same", 2)
        self.max_pool_2 : nn.MaxPool2d = nn.MaxPool2d(2, 2)

        self.conv2d_down_3 : Conv2D = Conv2D(256, 512, 3, 1, "same", 2)
        self.max_pool_3 : nn.MaxPool2d = nn.MaxPool2d(2, 2)

        self.conv2d_down_4 : Conv2D = Conv2D(512, 1024, 3, 1, "same", 2)

        # Upsampling
        self.up_sample_0 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv2d_up_0 : Conv2D = Conv2D(1024, 512, 3, 1, "same", 2)

        self.up_sample_1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv2d_up_1 : Conv2D = Conv2D(512, 256, 3, 1, "same", 2)

        self.up_sample_2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv2d_up_2 : Conv2D = Conv2D(256, 128, 3, 1, "same", 2)

        self.up_sample_3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv2d_up_3 : Conv2D = Conv2D(128, 64, 3, 1, "same", 2)
        self.conv2d_up_4 : Conv2D = Conv2D(64, n_out_channel, 1, 1, "same")

    def forward(self, x : torch.Tensor):
        # Downsampling
        out_conv2d_0 = self.conv2d_down_0(x)
        x = self.max_pool_0(out_conv2d_0)

        out_conv2d_1 = self.conv2d_down_1(x)
        x = self.max_pool_1(out_conv2d_1)

        out_conv2d_2 = self.conv2d_down_2(x)
        x = self.max_pool_2(out_conv2d_2)
        
        out_conv2d_3 = self.conv2d_down_3(x)
        x = self.max_pool_3(out_conv2d_3)
        
        x = self.conv2d_down_4(x)

        # Upsampling
        x = self.up_sample_0(x)
        x = torch.concat([x, out_conv2d_3], dim=1)
        x = self.conv2d_up_0(x)

        x = self.up_sample_1(x)
        x = torch.concat([x, out_conv2d_2], dim=1)
        x = self.conv2d_up_1(x)

        x = self.up_sample_2(x)
        x = torch.concat([x, out_conv2d_1], dim=1)
        x = self.conv2d_up_2(x)

        x = self.up_sample_3(x)
        x = torch.concat([x, out_conv2d_0], dim=1)
        x = self.conv2d_up_3(x)
        x = self.conv2d_up_4(x)

        return x

        
