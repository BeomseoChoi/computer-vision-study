import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class TensorBoardLogger():
    def __init__(self, log_dir : str, log_filename : str):
        self.log_dir : Path = Path(log_dir)
        self.log_filename : str = log_filename
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir / self.log_filename)

    def __del__(self):
        self.writer.flush()
        self.writer.close()
        
def model_save(net : nn.Module, model_dir : str, model_filename : str):
    model_dir : Path = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), model_dir / model_filename)
