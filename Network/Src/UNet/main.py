# Torch
import torch
from datetime import datetime
from Common.Src.DeviceWrapper import print_device_info
from Common.Src.directory import TensorBoardLogger
from Common.Src.directory import model_save

# Network
from UNet.Src.Network.PaddedUNet import PaddedUNet
from UNet.Src.train import train
from UNet.Src.test import test

# Dataset
from Resource.CityscapesDataset import CityscapesDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # Torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device) 

    # Network
    net = PaddedUNet(3, 3).to(device)
    n_epoch : int = 1
    learning_rate : float = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    dataset_training = CityscapesDataset('./Resource/cityscapes/cityscapes', is_training = True)
    dataset_test = CityscapesDataset('./Resource/cityscapes/cityscapes', is_training = False)
    dataloader_training = DataLoader(dataset_training, batch_size=16, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=True)

    datetime_now : str = datetime.now().strftime("%Y-%m-%d: %H:%M:%S")
    logger : TensorBoardLogger = TensorBoardLogger(log_dir="./UNet/Log", log_filename=datetime_now)

    for epoch in range(n_epoch):
        train_loss = train(net, dataloader_training, optimizer, device)
        test_loss = test(net, dataloader_test, device)
        
        logger.writer.add_scalar("Loss/train", train_loss, epoch)
        logger.writer.add_scalar("Loss/test", test_loss, epoch)

        print(f"[LOG] Epoch : {epoch + 1}, Train loss : {train_loss:.08f}, Test loss : {test_loss:.08f}")

    model_save(net=net, model_dir="./UNet/Model", model_filename=datetime_now + ".pt")

    print("done")