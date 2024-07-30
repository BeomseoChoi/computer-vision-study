import torch
import torch.nn as nn
from Network.Src.PointNet.TNet import TNet

class PointNet(nn.Module):
    def __init__(self, input_dim : int, n_class : int):
        super(PointNet, self).__init__()

        self.input_dim = input_dim
        self.n_class = n_class

        self.input_tnet = TNet(input_dim)
        self.feature_tnet = TNet(64)

        self.shared_mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.shared_mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, self.n_class),
            nn.BatchNorm1d(n_class),
            nn.ReLU()
        )

        self.log_softmax = nn.LogSoftmax()

    def forward(self, x : torch.Tensor):
        x = x.transpose(1, 2)
        input_transform = self.input_tnet(x)
        x = torch.bmm(x.transpose(1, 2), input_transform).transpose(1, 2)
        x = self.shared_mlp1(x)

        feature_transform = self.feature_tnet(x)
        x = torch.bmm(x.transpose(1, 2), feature_transform).transpose(1, 2)

        x = self.shared_mlp2(x)
        x = torch.max(x, 2, keepdim=False)[0] # element-wise maximum

        x = self.mlp(x)

        x = self.log_softmax(x)
        
        return x, input_transform, feature_transform

def regularizer_feature_transform(feature_transform):
    dim : int = feature_transform.size()[1]
    identity = torch.eye(dim, device=feature_transform.device).unsqueeze(0)
    gram_matrix = torch.bmm(feature_transform, feature_transform.transpose(1, 2))
    reg = torch.mean(torch.norm(identity - gram_matrix, dim=(1, 2)))

    return 0.001 * reg