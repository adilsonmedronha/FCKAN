import torch.nn as nn


class FCN(nn.Module):
  def __init__(self, n_classes):
    
    super(FCN, self).__init__()
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
    
    self.fc = nn.Sequential(
        nn.Flatten(),  
        nn.Linear(1408, n_classes),  
        nn.Softmax(dim=1) 
    )
  

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    self.feature_extractor = self.block3(x)
    x = self.fc(self.feature_extractor)
    return x