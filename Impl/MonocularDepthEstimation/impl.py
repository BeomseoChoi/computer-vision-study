# Torch
import torch
import torch.nn as nn
import torch.distributed
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from Common.Src.DeviceWrapper import DeviceWrapper
from Common.Src.directory import TensorBoardLogger
from Common.Src.directory import model_save
from Common.Src.DataLoaderWrapper import DataLoaderWrapper


# Network
from Impl.MonocularDepthEstimation.Src.Network.PaddedUNet_depth_estimation import PaddedUNet_depth_estimation
from Common.Src.Network import Network
from UNet.Src.train import train
from UNet.Src.test import test

# Dataset
from Resource.NYUv2Dataset import NYUv2Dataset

def loss_fn(pred : torch.Tensor, y : torch.Tensor):
    import Loss.Src.basic as basic_loss

    return basic_loss.loss_mse(pred, y)

def impl(device_wrapper : DeviceWrapper):
    # Network
    network = Network(PaddedUNet_depth_estimation(3, 1), device_wrapper, train, test)

    n_epoch : int = 200
    n_batch_per_device : int = 8
    learning_rate : float = 0.0001
    optimizer = None
    if device_wrapper.mode == "multi-gpu":
        optimizer = torch.optim.Adam(network.ddp_net.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(network.net.parameters(), lr=learning_rate)
    
    # t = transforms.Compose([transforms.CenterCrop((240, 320)), transforms.ToTensor()])
    t = transforms.Compose([transforms.ToTensor()])
    dataset_training = NYUv2Dataset('./Resource/NYUv2', dataset_x = "rgb", dataset_y = "depth", transform_x = t, transform_y = t, download=True, is_training = True)
    dataset_test = NYUv2Dataset('./Resource/NYUv2', dataset_x = "rgb", dataset_y = "depth", transform_x = t, transform_y = t, download=True, is_training = False)
    
    dataloader_training = DataLoaderWrapper(dataset_training, device_wrapper, n_batch_per_device)
    dataloader_test = DataLoaderWrapper(dataset_test, device_wrapper, n_batch_per_device)

    datetime_now : str = datetime.now().strftime("%Y-%m-%d: %H:%M:%S")
    logger : TensorBoardLogger = TensorBoardLogger(log_dir="./UNet/Log", log_filename=datetime_now)

    for epoch in range(n_epoch):
        if device_wrapper.mode == "multi-gpu":
            dataloader_training.sampler.set_epoch(epoch)
            dataloader_test.sampler.set_epoch(epoch)
        train_loss = network.train(dataloader=dataloader_training.dataloader, loss_fn=loss_fn, optimizer=optimizer)
        test_loss = network.test(dataloader=dataloader_test.dataloader, loss_fn=loss_fn)

        if device_wrapper.mode == "multi-gpu":
            if device_wrapper.device == 0:
                print(f"[LOG] Epoch : {epoch + 1}, Train loss : {train_loss:.08f}, Test loss : {test_loss:.08f}")
                logger.writer.add_scalar("Loss/train", train_loss, epoch)
                logger.writer.add_scalar("Loss/test", test_loss, epoch)
                model_save(net=network.net, model_dir=f"./UNet/Model/{datetime_now}", model_filename=f"{epoch}.pt")
        else:
            print(f"[LOG] Epoch : {epoch + 1}, Train loss : {train_loss:.08f}, Test loss : {test_loss:.08f}")
            logger.writer.add_scalar("Loss/train", train_loss, epoch)
            logger.writer.add_scalar("Loss/test", test_loss, epoch)
            model_save(net=network.net, model_dir=f"./UNet/Model/{datetime_now}", model_filename=f"{epoch}.pt")

    print("done")
