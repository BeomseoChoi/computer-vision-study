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
    n_epoch : int = 200
    learning_rate : float = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    # t = transforms.Compose([transforms.CenterCrop((240, 320)), transforms.ToTensor()])
    t = transforms.Compose([transforms.ToTensor()])
    dataset_training = NYUv2Dataset('./Resource/NYUv2', dataset_x = "rgb", dataset_y = "depth", transform_x = t, transform_y = t, download=True, is_training = True)
    dataset_test = NYUv2Dataset('./Resource/NYUv2', dataset_x = "rgb", dataset_y = "depth", transform_x = t, transform_y = t, download=True, is_training = False)
    dataloader_training = DataLoader(dataset_training, batch_size=4, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=True)

    datetime_now : str = datetime.now().strftime("%Y-%m-%d: %H:%M:%S")
    logger : TensorBoardLogger = TensorBoardLogger(log_dir="./UNet/Log", log_filename=datetime_now)

    for epoch in range(n_epoch):
        train_loss = train(net, dataloader_training, optimizer, device)
        test_loss = test(net, dataloader_test, device)
        
        logger.writer.add_scalar("Loss/train", train_loss, epoch)
        logger.writer.add_scalar("Loss/test", test_loss, epoch)
        
        model_save(net=net, model_dir=f"./UNet/Model/{datetime_now}", model_filename=f"{epoch}.pt")
        print(f"[LOG] Epoch : {epoch + 1}, Train loss : {train_loss:.08f}, Test loss : {test_loss:.08f}")


    print("done")