import torch
import torch.nn as nn
import torch.distributed as dist
from Common.Src.device import DeviceWrapper

def reduce_mean(tensor, nprocs):
    """
    https://github.com/rentainhe/pytorch-distributed-training/blob/master/utils/util.py#L5
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def train(model, device_wrapper : DeviceWrapper, *args, **kwargs) -> float:
    dataloader = kwargs["dataloader"]
    loss_fn = kwargs["loss_fn"]
    optimizer = kwargs["optimizer"]
    sum_local_loss : float = 0.0

    model.train()
    for x, y in dataloader:
        x, y = x.to(device_wrapper.device), y.to(device_wrapper.device)
        pred = model(x)

        loss = loss_fn(pred, y) 
        sum_local_loss += loss.item()

        optimizer.zero_grad()
        loss.backward() # automatically combine gradient when using DDP.
        optimizer.step()
    

    global_loss : float = 0.0
    if device_wrapper.mode == "multi-gpu":
        dist.barrier()

        # sum of losses
        sum_local_loss_tensor = torch.tensor(sum_local_loss).to(device_wrapper.device)
        dist.reduce(sum_local_loss_tensor, dst=0, op=dist.ReduceOp.AVG) # TODO: check what AVG does.

        # sum of len of dataloader
        n_mini_batch_tensor = torch.tensor([len(dataloader)]).to(device_wrapper.device)
        dist.reduce(n_mini_batch_tensor, dst=0, op=dist.ReduceOp.SUM)

        global_loss = sum_local_loss_tensor / n_mini_batch_tensor
        global_loss = global_loss.item()
    else:
        global_loss = sum_local_loss / len(dataloader)
        
    return global_loss

