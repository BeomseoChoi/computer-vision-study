import torch
import torch.nn as nn

def test(model, test_loader, device) -> tuple[float, float]:
    model.eval()
    loss_function = nn.MSELoss()
    avg_loss : float = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            
            loss = loss_function(pred, y) 
            avg_loss += loss.item()
    
    avg_loss /= len(test_loader)

    return avg_loss