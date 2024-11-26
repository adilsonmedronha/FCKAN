import torch 
import numpy as np
dtype = torch.get_default_dtype()

def transform(X, y):
  X, y = torch.tensor(X).type(dtype), torch.tensor(np.array(y, dtype=int))
  return X, y

