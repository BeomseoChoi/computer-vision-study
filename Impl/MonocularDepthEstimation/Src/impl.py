# Torch
import torch
import torch.nn as nn
import torch.distributed
from torchvision import transforms
from Common.Src.DeviceWrapper import DeviceWrapper
import Loss.Src.basic as basic_loss
import logging

# Network
from Impl.MonocularDepthEstimation.Src.Network.PaddedUNet_depth_estimation import PaddedUNet_depth_estimation
from Common.Src.NetworkWrapper import NetworkWrapper

# Dataset
from Resource.Src.NYUv2Dataset import NYUv2Dataset
from Common.Src.BaseImpl import BaseImpl

from pathlib import Path

class MonocularDepthEstimationImpl(BaseImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device_wrapper :  DeviceWrapper = self.get_device_wrapper()
        self.network_wrapper : NetworkWrapper = self.wrap_network(PaddedUNet_depth_estimation(3, 1))
        
        self.learning_rate = 0.0001
        self.optimizer = torch.optim.Adam(self.network_wrapper.get().parameters(), lr=self.learning_rate)

        self.n_batch_per_device = 8
        t = transforms.Compose([transforms.ToTensor()])
        self.dataset_training = NYUv2Dataset('./Resource/Data/NYUv2', dataset_x = "rgb", dataset_y = "depth", transform_x = t, transform_y = t, download=True, is_training = True)
        self.dataset_test = NYUv2Dataset('./Resource/Data/NYUv2', dataset_x = "rgb", dataset_y = "depth", transform_x = t, transform_y = t, download=True, is_training = False)
        self.dataloader_training_wrapper = self.wrap_training_dataloader(self.dataset_training, self.n_batch_per_device)
        self.dataloader_test_wrapper = self.wrap_test_dataloader(self.dataset_test, self.n_batch_per_device)

        self.log : str = ""

    def train(self, *args, **kwargs) -> None:
        sum_local_loss : float = 0.0
        self.network_wrapper.train()
        for x, y in self.dataloader_training_wrapper:
            x, y = x.to(self.device_wrapper.get()), y.to(self.device_wrapper.get())
            pred = self.network_wrapper(x)

            loss = basic_loss.loss_mse(pred, y)
            sum_local_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward() # automatically combine gradient when using DDP.
            self.optimizer.step()

        epoch : int = kwargs["epoch"]
        avg_global_loss = basic_loss.calc_avg_loss_from_sum(sum_local_loss, self.dataloader_training_wrapper, self.device_wrapper)
        logging.info(f"[LOG] Epoch : {epoch + 1}, Train loss : {avg_global_loss:.08f}")

    def test(self, *args, **kwargs) -> None:
        sum_local_loss : float = 0.0
        self.network_wrapper.eval()
        with torch.no_grad():
            for x, y in self.dataloader_test_wrapper:
                x, y = x.to(self.device_wrapper.get()), y.to(self.device_wrapper.get())
                pred = self.network_wrapper(x)

                loss = basic_loss.loss_mse(pred, y)
                sum_local_loss += loss.item()
        
        epoch : int = kwargs["epoch"]
        avg_global_loss = basic_loss.calc_avg_loss_from_sum(sum_local_loss, self.dataloader_training_wrapper, self.device_wrapper)
        logging.info(f", Test loss : {avg_global_loss:.08f}\n")

    def check_point(self, *args, **kwargs) -> tuple[Path, str, dict]:
            epoch : int = kwargs["epoch"] + 1

            model_dir : Path = Path("./Impl/MonocularDepthEstimation/Model")
            model_filename : str = f"{epoch}.pt"
            cp : dict = {}
            cp["state_dict"] = self.network_wrapper.get_state_dict()

            return model_dir, model_filename, cp
