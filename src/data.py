
from aeon.datasets import load_classification
import numpy as np
from sklearn.model_selection import StratifiedKFold

rnd_state = 42
np.random.seed(rnd_state)
dataset_name = "ProximalPhalanxOutlineCorrect"

X_train, y_train = load_classification(name=dataset_name, split='train')
X_test, y_test = load_classification(name=dataset_name, split='test')

