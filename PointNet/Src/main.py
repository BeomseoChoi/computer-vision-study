# Torch
import torch
from Common.Src.device import print_device_info

# Network
from PointNet.Src.Network.PointNet import PointNet
from PointNet.Src.train import train
from PointNet.Src.test import test

# Dataset
from Resource.ModelNet40Dataset import ModelNet40Dataset
from torch.utils.data import DataLoader

# Tensor board
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    # classes : list[str] = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle"]
    classes : list[str] = ["airplane", "bathtub"]
    net = PointNet(input_dim=3, n_class=len(classes)).to(device)
    n_sample : int = 4096
    n_epoch : int = 100
    learning_rate : float = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    dataset_training = ModelNet40Dataset("./Resource/ModelNet40", 
                                         n_sample = n_sample,
                                         class_list = classes,
                                         is_training = True)
    dataset_training.normalize()

    dataset_test = ModelNet40Dataset("./Resource/ModelNet40", 
                                     n_sample = n_sample,
                                     class_list = classes,
                                     is_training = False)
    dataset_test.normalize()

    dataloader_training = DataLoader(dataset_training, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)
    
    datetime_now = datetime.now().strftime("%Y-%m-%d: %H:%M:%S")

    log_path : Path = Path(f"./PointNet/Log/{datetime_now}")
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)

    for epoch in range(n_epoch):
        train_loss, train_accuracy = train(net, dataloader_training, optimizer, device)
        test_loss, test_accuracy = test(net, dataloader_test, device)
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_accuracy, epoch)

        print(f"[LOG] Epoch : {epoch + 1}, Train loss : {train_loss:.08f}, Train Accuracy : {train_accuracy:.08f}, Test loss : {test_loss:.08f}, Test Accuracy : {test_accuracy:.08f}")
    
    model_path : Path = Path(f"./PointNet/Models/{datetime_now}.pt")
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), model_path)

    writer.flush()
    writer.close()



    print("done")
    