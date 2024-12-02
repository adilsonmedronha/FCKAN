import torch.nn as nn
from torch.nn.functional import avg_pool1d

class FCN(nn.Module):
  def __init__(self, n_classes):
    
    super(FCN, self).__init__()
    self.block1 = nn.Sequential(
        nn.Conv1d(in_channels=1, 
                  out_channels=128,
                  kernel_size=8, 
                  padding='same'),  
        nn.BatchNorm1d(128),  
        nn.ReLU() 
    )
    
    self.block2 = nn.Sequential(
        nn.Conv1d(in_channels=128, 
                  out_channels=256, 
                  kernel_size=5,
                  padding='same'),  
        nn.BatchNorm1d(256),  
        nn.ReLU() 
    )
    
    self.block3 = nn.Sequential(
        nn.Conv1d(in_channels=256, 
                  out_channels=128, 
                  kernel_size=3,
                  padding='same'),  
        nn.BatchNorm1d(128),  
        nn.ReLU() 
    )


    
    self.fc = nn.Sequential(
        nn.Flatten(),  
        nn.Linear(128, n_classes),  
        nn.Softmax(dim=1) 
    )
  

  def forward(self, x):
    x = self.block1(x)
    print(f'block1: {x.shape}')
    x = self.block2(x)
    print(f'block2: {x.shape}')
    x = self.block3(x)
    print(f'block3: {x.shape}')
    x = avg_pool1d(x, x.shape[-1])
    print(f'avg_pool1d: {x.shape}')
    x = x.squeeze(-1)
    print(f'squeeze: {x.shape}')
    x = self.fc(x)
    print(f'fc: {x.shape}')
    return x