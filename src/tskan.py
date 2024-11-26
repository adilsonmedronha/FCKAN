from aeon.datasets import load_classification
from aeon.classification.deep_learning import FCNClassifier
from sklearn.model_selection import train_test_split
import aeon
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

rnd_state = 42
np.random.seed(rnd_state)

# 1000 + 370 (train + test) = 1370

def transform(X, y):
  X, y = torch.tensor(X).type(dtype), torch.tensor(np.array(y, dtype=int))
  return X, y

X_train_aeon, y_train_aeon = load_classification(name="ProximalPhalanxOutlineCorrect", split='train')
X_train, y_train = transform(X_train_aeon.squeeze(1), y_train_aeon)

X_test_aeon, y_test_aeon = load_classification(name="ProximalPhalanxOutlineCorrect", split='test')
X_test, y_test = transform(X_test_aeon.squeeze(1), y_test_aeon)

_, input_dim = X_train.shape