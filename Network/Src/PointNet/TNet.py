import torch
import torch.nn as nn

class TNet(nn.Module):
    def __init__(self, k : int):
        super(TNet, self).__init__()

        self.k = k
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.fc_network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, self.k * self.k)
        )
    
    def forward(self, x):
        batch_size : int = x.size()[0]
        x = self.shared_mlp(x)
        x = torch.max(x, 2, keepdim=True)[0] # element-wise maximum. (ref. from Theorem 1. on PointNet paper.)
        x = x.view(-1, 1024)
        x = self.fc_network(x) # (, k * k)
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x
