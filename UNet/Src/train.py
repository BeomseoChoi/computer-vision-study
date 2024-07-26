import torch.nn as nn

def train(model : nn.modules, train_loader, optimizer, device, tensor_board = None) -> float:
    model.train()
    loss_function = nn.MSELoss()
    avg_loss : float = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()

        pred = model(x)
        loss = loss_function(pred, y) 
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()

    avg_loss /= len(train_loader)

    return avg_loss
