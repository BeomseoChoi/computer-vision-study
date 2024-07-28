# Vis
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Torch
import torch
from torchvision import transforms
from datetime import datetime
from Common.Src.device import print_device_info
from Common.Src.directory import TensorBoardLogger
from Common.Src.directory import model_save

# Network
from UNet.Src.Network.PaddedUNet_depth_estimation import PaddedUNet_depth_estimation
from UNet.Src.train import train
from UNet.Src.test import test

# Dataset
from Resource.NYUv2Dataset import NYUv2Dataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # Torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device) 

    # Network
    net = PaddedUNet_depth_estimation(3, 1).to(device)
    state_dict = torch.load("./UNet/Model/2024-07-27: 06:14:34.pt")
    net.load_state_dict(state_dict)
    
    t = transforms.Compose([transforms.ToTensor()])
    dataset_test = NYUv2Dataset('./Resource/NYUv2', dataset_x = "rgb", dataset_y = "depth", download=True, is_training = True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

    net.eval()
    for x, y in dataloader_test:
        x = x.to(device)
        pred = net(x)

        plt.subplot(1, 3, 1) # x
        x = x.squeeze(0).permute(1, 2, 0)
        x = x.cpu().detach().numpy()
        plt.imshow(x)
        plt.subplot(1, 3, 2) # pred
        pred = pred.squeeze(0).permute(1, 2, 0)
        pred = pred.cpu().detach().numpy()
        plt.imshow(pred)
        plt.subplot(1, 3, 3) # gt
        y = y.squeeze(0).permute(1, 2, 0)
        y = y.cpu().detach().numpy()
        plt.imshow(y)
        plt.show()

    print("done")