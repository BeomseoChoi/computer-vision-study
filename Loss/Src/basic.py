import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from Common.Src.DeviceWrapper import DeviceWrapper
from Common.Src.DataLoaderWrapper import DataLoaderWrapper

def loss_ssim(pred : torch.Tensor, y : torch.Tensor):
    import pytorch_ssim

    ssim_term = pytorch_ssim.ssim(pred, y)
    
    return ssim_term

def loss_l1(pred, y):
    return nn.functional.l1_loss(pred, y)

def loss_ssim_l2(pred : torch.Tensor, y : torch.Tensor):
    return 0.2 * loss_ssim(pred, y) + 0.8 * loss_mse(pred, y)

def loss_fn(pred : torch.Tensor, y : torch.Tensor):
    return loss_mse(pred, y)

def loss_mse(pred, y):
    return nn.functional.mse_loss(pred, y)


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

def calc_avg_loss_from_sum(sum_loss : float, dataloader_wrapper : DataLoaderWrapper, device_wrapper : DeviceWrapper):
    if device_wrapper.is_multi_gpu_mode():
        dist.barrier()

        # sum of losses
        sum_local_loss_tensor = torch.tensor(sum_loss).to(device_wrapper.get())
        dist.reduce(sum_local_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        
        # sum of len of dataloader
        n_mini_batch_tensor = torch.tensor([len(dataloader_wrapper)]).to(device_wrapper.get())
        dist.reduce(n_mini_batch_tensor, dst=0, op=dist.ReduceOp.SUM)

        global_loss = sum_local_loss_tensor / n_mini_batch_tensor
        global_loss = global_loss.item()
    else:
        global_loss = sum_loss / len(dataloader_wrapper)
        
    return global_loss