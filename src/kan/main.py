from kan import KAN
import torch
from aeon.datasets import load_classification
from kan.utils import create_dataset_from_data
from sklearn.model_selection import StratifiedKFold
from utils import transform
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

dtype = torch.get_default_dtype()



rnd_state = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


X_train_aeon, y_train_aeon = load_classification(name="Chinatown", split='train')
le = LabelEncoder()
y_train_aeon = le.fit_transform(y_train_aeon)
X_train, y_train = transform(X_train_aeon.squeeze(1), y_train_aeon)

X_test_aeon, y_test_aeon = load_classification(name="Chinatown", split='test')
X_test, y_test = transform(X_test_aeon.squeeze(1), y_test_aeon)
y_test_aeon = le.transform(y_test_aeon)
_, input_dim = X_train.shape


dataset = create_dataset_from_data(X_train, y_train, device=device) 
n_inputs, input_dim = dataset['train_input'].shape
n_outputs = len(torch.unique(dataset['train_label']))


def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype))

def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype))


k_folds = 2
skf = StratifiedKFold(n_splits=k_folds, random_state=rnd_state, shuffle=True)
grids = [3,5,10,20,50,100]
k = [3, ]

    
results_list = []

# Loop through the number of models (e.g., 1 to 3)
for n in range(1, 2): 
    print(f'\n ****** {n} ******')
    train_accs = []
    test_accs = []

    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        dataset = create_dataset_from_data(X_train, y_train, device=device)
        model = KAN(width=[input_dim, n, n_outputs], grid=5, k=3, seed=rnd_state, device=device)

        # Fit the model and collect results
        results = model.fit(
            dataset,
            opt="LBFGS",
            steps=50,
            metrics=(train_acc, test_acc),
            loss_fn=torch.nn.CrossEntropyLoss()
        )

        # Append accuracy results to the lists
        train_accs.append(results['train_acc'][-1])
        test_accs.append(results['test_acc'][-1])
        print(f'fold {i+1} - Train Acc: {results["train_acc"][-1]} Test Acc: {results["test_acc"][-1]}')

# Print the DataFrame
print("\nSaved results to 'model_accuracies.csv'.")

with open('grid_search_results.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['n', 'fold', 'train_acc', 'test_acc', 'mean_train_acc', 'std_train_acc', 'mean_test_acc', 'std_test_acc'])
    writer.writeheader()
    writer.writerows(results_list)
print("Results saved to grid_search_results.csv")