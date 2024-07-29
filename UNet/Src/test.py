import torch
import torch.nn as nn
import torch.distributed as dist
from Common.Src.device import DeviceWrapper

def test(model, device_wrapper, *args, **kwargs) -> tuple[float, float]:

    dataloader = kwargs["dataloader"]
    loss_fn = kwargs["loss_fn"]
    sum_local_loss : float = 0.0
    global_loss : float = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device_wrapper.device), y.to(device_wrapper.device)
            pred = model(x)

            loss = loss_fn(pred, y) 
            sum_local_loss += loss.item()

    if device_wrapper.mode == "multi-gpu":
        dist.barrier()

        # sum of losses
        sum_local_loss_tensor = torch.tensor(sum_local_loss).to(device_wrapper.device)
        dist.reduce(sum_local_loss_tensor, dst=0, op=dist.ReduceOp.AVG) # TODO: check what AVG does.
        # dist.all_reduce(sum_local_loss_tensor, op=dist.ReduceOp.AVG) # TODO: check what AVG does.

        # sum of len of dataloader
        n_mini_batch_tensor = torch.tensor([len(dataloader)]).to(device_wrapper.device)
        dist.reduce(n_mini_batch_tensor, dst=0, op=dist.ReduceOp.SUM)
        # dist.all_reduce(n_mini_batch_tensor, op=dist.ReduceOp.SUM)

        global_loss = sum_local_loss_tensor / n_mini_batch_tensor
        global_loss = global_loss.item()
    else:
        global_loss = sum_local_loss / len(dataloader)
        
    return global_loss


    """ 
    dataloader = kwargs["dataloader"]
    loss_fn = kwargs["loss_fn"]
    avg_loss : float = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            
            loss = loss_fn(pred, y) 
            avg_loss += loss.item()
    
    avg_loss /= len(dataloader)

    return avg_loss
    """