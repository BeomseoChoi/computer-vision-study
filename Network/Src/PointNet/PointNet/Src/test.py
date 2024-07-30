import torch
import torch.nn as nn
from PointNet.Src.Network.PointNet import regularizer_feature_transform

def test(model, test_loader, device) -> tuple[float, float]:
    model.eval()  # 모델을 평가 모드로 전환
    loss_function = nn.NLLLoss()
    avg_loss : float = 0.0
    correct : float = 0.0
    with torch.no_grad():  # 평가 시에는 그래디언트 계산을 하지 않음
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            pred, _, feature_transform = model(x)
            
            loss = loss_function(pred, y.argmax(1)) 
            loss += regularizer_feature_transform(feature_transform)

            pred_class = pred.argmax(dim=1, keepdim=True)
            y_class = y.argmax(dim=1, keepdim=True)
            correct += (pred_class == y_class).type(torch.float).sum().item()
            avg_loss += loss.item()
    
    avg_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    return avg_loss, accuracy