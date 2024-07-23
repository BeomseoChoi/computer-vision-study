import torch
import torch.nn as nn
from PointNet.Src.Network.PointNet import regularizer_feature_transform

def train(model : nn.modules, train_loader, optimizer, device, tensor_board = None) -> float:
    model.train()
    loss_function = nn.NLLLoss()
    avg_loss : float = 0.0
    correct : float = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()

        pred, _, feature_transform = model(x)
        loss = loss_function(pred, y.argmax(1)) 
        loss += regularizer_feature_transform(feature_transform)
        loss.backward()
        optimizer.step()
        
        pred_class = pred.argmax(dim=1, keepdim=True)
        y_class = y.argmax(dim=1, keepdim=True)
        correct += (pred_class == y_class).type(torch.float).sum().item()
        avg_loss += loss.item()

    avg_loss /= len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)

    return avg_loss, accuracy
