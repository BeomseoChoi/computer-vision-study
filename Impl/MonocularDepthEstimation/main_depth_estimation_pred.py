# Vis
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torchvision import transforms
from datetime import datetime
from Common.Src.DeviceWrapper import DeviceWrapper
from Common.Src.directory import TensorBoardLogger
from Common.Src.directory import model_save

# Network
from Impl.MonocularDepthEstimation.Src.Network.PaddedUNet_depth_estimation import PaddedUNet_depth_estimation
from UNet.Src.main_depth_estimation import DataLoaderWrapper
from UNet.Src.train import train
from UNet.Src.test import test

# Dataset
from Resource.NYUv2Dataset import NYUv2Dataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # Torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DeviceWrapper.print_device_info(device) 

    # Network
    net = PaddedUNet_depth_estimation(3, 1).to(device)
    path : str = "UNet/Model/2024-07-29: 10:34:18"
    state_dict = torch.load(f"{path}/0.pt")
    net.load_state_dict(state_dict) # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/4
    
    t = transforms.Compose([transforms.ToTensor()])
    dataset_test = NYUv2Dataset('./Resource/NYUv2', dataset_x = "rgb", dataset_y = "depth", transform_x = t, transform_y = t, download=True, is_training = False)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

    net.eval()
    for i, (x, y) in enumerate(dataloader_test):
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
        plt.savefig(f"{path}/image/{i}.png")

    print("done")