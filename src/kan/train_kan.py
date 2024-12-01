import torch
from kan import KAN
import torch.nn as nn
from sklearn.metrics import f1_score
from kan.utils import create_dataset_from_data
from utils.data import DatasetManager
import argparse
import csv


class KANTrainer:
    def __init__(self, n_classes, rnd_state, device):
            self.n_classes = n_classes
            self.rnd_state = rnd_state
            self.device = device

    @staticmethod
    def test_acc(dataset, model):
        return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

    @staticmethod
    def f1(dataset, model):
        y_true = dataset['test_label'].cpu().numpy()
        y_pred = torch.argmax(model(dataset['test_input']), dim=1).cpu().numpy()
        return f1_score(y_true, y_pred, average='weighted')

    def train_model(self, dataset, n, g, k):
        
        input_dim = dataset['train_input'].shape[1]
        n_outputs = len(torch.unique(dataset['train_label']))
        
        model = KAN(width=[input_dim, n, n_outputs], 
                    grid=g, k=k, 
                    seed=self.rnd_state, device=self.device)
        model.fit(
            dataset,
            opt="LBFGS",
            steps=100,
            loss_fn=torch.nn.CrossEntropyLoss()
        )
        test_accuracy = self.test_acc(dataset, model)
        f1_test_score = self.f1(dataset, model)
        return test_accuracy, f1_test_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name.")
    parser.add_argument("--n", type=int, help="#Neurons/nodes.")
    parser.add_argument("--g", type=int, help="Grids sizes.")
    parser.add_argument("--k", type=int, help="Spline order.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnd_state = 42
    dataset_manager = DatasetManager(args.dataset, device)
    print(f"Dataset: {args.dataset} loaded.")
    print(f"Classes: {dataset_manager.get_classes_number()}")
    print(f"Samples: {dataset_manager.dataset['train_input'].shape[0]}")
    print(f"Features: {dataset_manager.dataset['train_input'].shape[1]}")

    
    trainer = KANTrainer(dataset_manager.get_classes_number(), rnd_state, device)
    test_accuracy, f1_test_score = trainer.train_model(dataset_manager.dataset, args.n, args.g, args.k)


    results_file = f"results_after_hsearch/{args.dataset}_trained_with_best_params.csv"

    data = [
        ["Dataset", "Accuracy", "F1 Score"],  
        [dataset_manager.dataset_name, test_accuracy.item(), f1_test_score]  
    ]

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"Results saved to {results_file}")