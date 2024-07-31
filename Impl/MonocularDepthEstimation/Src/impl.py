# Torch
import torch
import torch.nn as nn
from torchvision import transforms
from Common.Src.DeviceWrapper import DeviceWrapper
from Common.Src.Logger import Logger
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
        seed = kwargs["args"].seed

        self.device_wrapper :  DeviceWrapper = None
        self.network_wrapper : NetworkWrapper = None
        self.optimizer = None
        
        self.learning_rate = 0.0001
        self.n_batch_per_device = 8

        t = transforms.Compose([transforms.CenterCrop((480 - 16, 640 - 16)), transforms.ToTensor()])
        self.dataset_training = NYUv2Dataset('./Resource/Data/NYUv2', dataset_x = "rgb", dataset_y = "depth", transform_x = t, transform_y = t, download=True, is_training = True)
        self.dataset_validation, self.dataset_test = NYUv2Dataset('./Resource/Data/NYUv2', dataset_x = "rgb", dataset_y = "depth", transform_x = t, transform_y = t, download=True, is_training = False).split(test_ratio = 0.5, seed = seed)

        self.dataloader_training_wrapper = None
        self.dataloader_validation_wrapper = None
        self.dataloader_test_wrapper = None

        self.logger : Logger = None
        self.log : str = ""
    
    def initialize(self, *args, **kwargs) -> None:
        self.device_wrapper = self.get_device_wrapper()
        self.logger = Logger(self.device_wrapper)

        if kwargs["args"].type == "train":
            self.network_wrapper = self.wrap_network(PaddedUNet_depth_estimation(3, 1))
            self.optimizer = torch.optim.Adam(self.network_wrapper.get().parameters(), lr=self.learning_rate)
        
            self.dataloader_training_wrapper = self.wrap_training_dataloader(self.dataset_training, self.n_batch_per_device)
            self.dataloader_validation_wrapper = self.wrap_validation_dataloader(self.dataset_validation, self.n_batch_per_device)
        else:
            model_dict : dict = torch.load(kwargs["args"].model_path)
            net = PaddedUNet_depth_estimation(3, 1) # TODO: save model arguments
            net.load_state_dict(model_dict["state_dict"])
            self.network_wrapper = self.wrap_network(net)
            
            self.dataloader_test_wrapper = self.wrap_test_dataloader(self.dataset_test, self.n_batch_per_device)

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
        self.log = f"Epoch : {epoch + 1}, Train loss : {avg_global_loss:.08f}"

    def validate(self, *args, **kwargs) -> None:
        sum_local_loss : float = 0.0
        self.network_wrapper.eval()
        with torch.no_grad():
            for x, y in self.dataloader_validation_wrapper:
                x, y = x.to(self.device_wrapper.get()), y.to(self.device_wrapper.get())
                pred = self.network_wrapper(x)

                loss = basic_loss.loss_mse(pred, y)
                sum_local_loss += loss.item()
        
        avg_global_loss = basic_loss.calc_avg_loss_from_sum(sum_local_loss, self.dataloader_validation_wrapper, self.device_wrapper)
        self.log += f", Validation loss : {avg_global_loss:.08f}"

    def end_epoch(self, *args, **kwargs) -> None:
         self.logger.info(self.log)

    def check_point(self, *args, **kwargs) -> tuple[Path, str, dict]:
            cp : dict = {}
            cp["state_dict"] = self.network_wrapper.get_state_dict()

            return cp
    
    def test(self, *args, **kwargs) -> None:
        sum_local_loss : float = 0.0
        self.network_wrapper.eval()
        with torch.no_grad():
            for x, y in self.dataloader_test_wrapper:
                x, y = x.to(self.device_wrapper.get()), y.to(self.device_wrapper.get())
                pred = self.network_wrapper(x)

                loss = basic_loss.loss_mse(pred, y)
                sum_local_loss += loss.item()
        
        avg_global_loss = basic_loss.calc_avg_loss_from_sum(sum_local_loss, self.dataloader_test_wrapper, self.device_wrapper)
        self.log = f"Test loss : {avg_global_loss:.08f}"
        self.logger.info(self.log)


