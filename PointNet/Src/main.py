import torch
import numpy as np
import open3d as o3d
from Common.Src.device import print_device_info
from Resource.ModelNet40Dataset import ModelNet40Dataset
from torch.utils.data import DataLoader
from PointNet.Src.train import train
from PointNet.Src.test import test
from PointNet.Src.Network.PointNet import PointNet
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    classes : list[str] = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle"]
    # classes : list[str] = ["airplane", "bathtub"]
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
    
    log_filename = datetime.now().strftime("%Y-%m-%d: %H:%M:%S")
    writer = SummaryWriter(f"./PointNet/Log/{log_filename}")

    for epoch in range(n_epoch):
        train_loss, train_accuracy = train(net, dataloader_training, optimizer, device)
        test_loss, test_accuracy = test(net, dataloader_test, device)
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_accuracy, epoch)

        print(f"[LOG] Train loss : {train_loss:.08f}, Train Accuracy : {train_accuracy:.08f}, Test loss : {test_loss:.08f}, Test Accuracy : {test_accuracy:.08f}")

    writer.flush()
    writer.close()



    print("done")
    