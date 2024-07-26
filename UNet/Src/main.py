# Torch
import torch
from Common.Src.device import print_device_info

# Network
from UNet.Src.Network.PaddedUNet import PaddedUNet
from UNet.Src.train import train
from UNet.Src.test import test

# Dataset
from Resource.CityscapesDataset import CityscapesDataset
from torch.utils.data import DataLoader

# Tensor board
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

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

    datetime_now = datetime.now().strftime("%Y-%m-%d: %H:%M:%S")

    log_path : Path = Path(f"./UNet/Log/{datetime_now}")
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)

    for epoch in range(n_epoch):
        train_loss = train(net, dataloader_training, optimizer, device)
        test_loss = test(net, dataloader_test, device)
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)

        print(f"[LOG] Epoch : {epoch + 1}, Train loss : {train_loss:.08f}, Test loss : {test_loss:.08f}")

    model_path : Path = Path(f"./UNet/Models/{datetime_now}.pt")
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), model_path)

    writer.flush()
    writer.close()

    print("done")