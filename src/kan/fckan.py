import torch.nn as nn
from kan import KAN


class FCKAN(nn.Module):
  def __init__(self, n_classes, rnd_state, device):
    
    super(FCKAN, self).__init__()
    self.block1 = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=128, kernel_size=8),  
        nn.BatchNorm1d(128),  
        nn.ReLU() 
    )
    
    self.block2 = nn.Sequential(
        nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5),  
        nn.BatchNorm1d(256),  
        nn.ReLU() 
    )
    
    self.block3 = nn.Sequential(
        nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),  
        nn.BatchNorm1d(128),  
        nn.ReLU() 
    )
    
    self.kan = KAN(width=[1408, 5, n_classes], 
                   grid=5, k=3, seed=rnd_state, device=device)
  

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    self.feature_extractor = self.block3(x).reshape(x.shape[0], -1)
    x = self.kan(self.feature_extractor)
    return x