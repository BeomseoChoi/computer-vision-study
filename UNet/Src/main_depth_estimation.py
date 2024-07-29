# Torch
import torch
import torch.nn as nn
import torch.distributed
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from Common.Src.device import DeviceWrapper
from Common.Src.directory import TensorBoardLogger
from Common.Src.directory import model_save


# Network
from UNet.Src.Network.PaddedUNet_depth_estimation import PaddedUNet_depth_estimation
from UNet.Src.train import train
from UNet.Src.test import test

# Dataset
from Resource.NYUv2Dataset import NYUv2Dataset
from torch.utils.data import DataLoader

# Misc.
import time

class Network():
    def __init__(self, net, device_wrapper : DeviceWrapper, train_fn, test_fn):
        DeviceWrapper.check_if_valid_device_mode(device_wrapper.mode)
        if not callable(train_fn): raise RuntimeError("The argument 'train_fn' must be callable.")
        if not callable(test_fn): raise RuntimeError("The argument 'test_fn' must be callable.")

        self.device_wrapper = device_wrapper
        self.net = net
        self.ddp_net = None
        if device_wrapper.mode == "cpu":
            self.net.to(device_wrapper.device)
        elif device_wrapper.mode == "single-gpu":
            self.net.to(device_wrapper.device)
        elif device_wrapper.mode == "multi-gpu":
            torch.cuda.set_device(device_wrapper.device)
            self.net.to(device_wrapper.device)
            # 여기가 문제. DDP랑 Non-DDP를 구분해야한다. Get property를 만들어야 할 것 같다.
            self.ddp_net = DDP(self.net, device_ids=[device_wrapper.device]) 

        self.train_fn = train_fn
        self.test_fn = test_fn

    def train(self, *args, **kwargs):
        ret = 0
        if self.device_wrapper.mode == "multi-gpu":
            ret = self.train_fn(self.ddp_net, self.device_wrapper, *args, **kwargs)
        else:
            ret = self.train_fn(self.net, self.device_wrapper, *args, **kwargs)

        return ret
    
    def test(self, *args, **kwargs):
        ret = 0
        if self.device_wrapper.mode == "multi-gpu":
            ret = self.test_fn(self.ddp_net, self.device_wrapper, *args, **kwargs)
        else:
            ret = self.test_fn(self.net, self.device_wrapper, *args, **kwargs)

        return ret

class DataLoaderWrapper():
    def __init__(self, dataset, device_wrapper : DeviceWrapper, n_batch_per_device : int):
        DeviceWrapper.check_if_valid_device_mode(device_wrapper.mode)
        self.dataset = dataset
        self.dataloader = None
        self.sampler = None
        if device_wrapper.mode == "multi-gpu":
            self.sampler = DistributedSampler(dataset)
            self.dataloader = DataLoader(dataset, batch_size=n_batch_per_device, num_workers=0, pin_memory=True, shuffle=False, sampler=self.sampler)
        else:
            self.dataloader = DataLoader(dataset, batch_size=n_batch_per_device, shuffle=True)
        
def main_worker(device, mode, n_device):
    # 멀티프로세싱 통신 규약 정의
    if mode == "multi-gpu":
        torch.distributed.init_process_group(
            backend='nccl', 
            init_method='tcp://127.0.0.1:2568', 
            world_size=n_device, 
            rank=device)

    device_wrapper : DeviceWrapper = DeviceWrapper(device, n_device, mode)
    depth_estimation(device_wrapper)

    if mode == "multi-gpu":
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        # 입력 경계의 반사를 사용하여 상/하/좌/우에 입력 텐서를 추가로 채웁니다.
        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        # shape : (xh, xw) -> (xh + 2, xw + 2)
        x = self.refl(x) 
        # shape : (yh, yw) -> (yh + 2, yw + 2)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        # SSIM score
        # return torch.clamp((SSIM_n / SSIM_d) / 2, 0, 1)

        # Loss function
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def loss_ssim(pred : torch.Tensor, y : torch.Tensor):
    import pytorch_ssim

    alpha : float = 0.2
    ssim_term = pytorch_ssim.ssim(pred, y)
    l1_loss = nn.functional.l1_loss(pred, y)
    
    return ssim_term
    # return alpha * ssim_term + (1 - alpha) * l1_loss

def loss_l1(pred, y):
    return nn.functional.l1_loss(pred, y)

def loss_ssim_l2(pred : torch.Tensor, y : torch.Tensor):
    return 0.2 * loss_ssim(pred, y) + 0.8 * loss_mse(pred, y)

def loss_fn(pred : torch.Tensor, y : torch.Tensor):
    return loss_mse(pred, y)

def loss_mse(pred, y):
    return nn.functional.mse_loss(pred, y)

def depth_estimation(device_wrapper : DeviceWrapper):
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

if __name__ == "__main__":
    # Torch
    mode : str = DeviceWrapper.get_device_mode()
    if mode == "cpu":
        main_worker("cpu", mode, -1)
    elif mode == "single-gpu":
        main_worker(torch.device("cuda"), mode, 1)
    else:
        n_gpus = torch.cuda.device_count()
        torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(mode, n_gpus,), join=True)