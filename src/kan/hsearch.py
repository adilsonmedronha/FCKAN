import argparse
import torch
from kan import KAN
from aeon.datasets import load_classification
from kan.utils import create_dataset_from_data
from sklearn.model_selection import StratifiedKFold
from utils import transform
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


class DatasetManager:
    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.label_encoder = LabelEncoder()
        self.load_data()

    def load_data(self):
        X_train_aeon, y_train_aeon = load_classification(name=self.name, split='train')
        X_test_aeon, y_test_aeon = load_classification(name=self.name, split='test')
        
        y_train_encoded = self.label_encoder.fit_transform(y_train_aeon)
        y_test_encoded = self.label_encoder.transform(y_test_aeon)

        self.X_train, self.y_train = transform(X_train_aeon.squeeze(1), y_train_encoded)
        self.X_test, self.y_test = transform(X_test_aeon.squeeze(1), y_test_encoded)
        self.dataset = create_dataset_from_data(self.X_train, self.y_train, device=self.device)

class GridSearch:
    def __init__(self, dataset_manager, device, k_folds=5, random_state=42):
        self.dataset_manager = dataset_manager
        self.device = device
        self.k_folds = k_folds
        self.random_state = random_state
        self.results = []

    def train_model(self, X_train, y_train, X_val, y_val, params):
        dataset_train = create_dataset_from_data(X_train, y_train, device=self.device)
        dataset_val = create_dataset_from_data(X_val, y_val, device=self.device)
        
        input_dim = dataset_train['train_input'].shape[1]
        n_outputs = len(torch.unique(dataset_train['train_label']))
        
        model = KAN(width=[input_dim, params['n'], n_outputs], 
                    grid=params['g'], k=params['k'], 
                    seed=self.random_state, device=self.device)
        model.fit(
            dataset_train,
            opt="LBFGS",
            steps=1,
            loss_fn=torch.nn.CrossEntropyLoss()
        )
        test_accuracy = self.test_acc(dataset_val, model)
        f1_test_score = self.f1(dataset_val, model)
        return test_accuracy, f1_test_score

    @staticmethod
    def test_acc(dataset, model):
        return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

    @staticmethod
    def f1(dataset, model):
        y_true = dataset['test_label'].cpu().numpy()
        y_pred = torch.argmax(model(dataset['test_input']), dim=1).cpu().numpy()
        return f1_score(y_true, y_pred, average='weighted')

    def run(self, N, G, K):
        skf = StratifiedKFold(n_splits=self.k_folds, random_state=self.random_state, shuffle=True)
        X_train, y_train = self.dataset_manager.X_train, self.dataset_manager.y_train

        for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(X_train, y_train)), total=self.k_folds, desc="Cross-Validation"):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            for n in N: # neurons/nodes
                for g in G: # grids sizes
                    for k in K: # spline order
                        params = {'n': n, 'g': g, 'k': k}
                        test_accuracy, f1_test_score = self.train_model(X_train_fold, 
                                                                        y_train_fold, 
                                                                        X_val_fold, y_val_fold, params)
                        self.results.append({
                            'params': params,
                            'fold': fold,
                            'test_accuracy': test_accuracy,
                            'f1_score': f1_test_score
                        })
    
    def save_results(self, filename):
        results_dict = {}
        for result in self.results:
            params = f"N={result['params']['n']}, G={result['params']['g']}, K={result['params']['k']}"
            if params not in results_dict:
                results_dict[params] = {
                    "test_scores": [],
                    "f1_scores": []
                }

            test_score = (
                result["test_accuracy"].cpu().numpy() 
                if isinstance(result["test_accuracy"], torch.Tensor) 
                else result["test_accuracy"]
            )
            f1_score = (
                result["f1_score"].cpu().numpy() 
                if isinstance(result["f1_score"], torch.Tensor) 
                else result["f1_score"]
            )

            results_dict[params]["test_scores"].append(test_score)
            results_dict[params]["f1_scores"].append(f1_score)

        rows = []
        for params, metrics in results_dict.items():
            test_scores = metrics["test_scores"]
            f1_scores = metrics["f1_scores"]
            mean_test_score = np.mean(test_scores)
            std_test_score = np.std(test_scores)
            mean_f1_score = np.mean(f1_scores)
            std_f1_score = np.std(f1_scores)
            rank = None  

            row = {
                "params": params,
                **{f"split{idx}_test_score": test_scores[idx] for idx in range(len(test_scores))},
                **{f"split{idx}_f1_score": f1_scores[idx] for idx in range(len(f1_scores))},
                "mean_test_score": mean_test_score,
                "std_test_score": std_test_score,
                "mean_f1_score": mean_f1_score,
                "std_f1_score": std_f1_score,
                "rank_test_score": rank,
            }
            rows.append(row)

        final_results_df = pd.DataFrame(rows)
        final_results_df = final_results_df.sort_values(by="mean_test_score", ascending=False)
        final_results_df["rank_test_score"] = final_results_df["mean_test_score"].rank(
            ascending=False, method="min"
        )

        final_results_df.to_csv(filename, index=False)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Grid search for KAN models.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Wafer", "Strawberry", "HandOutlines", "TwoPatterns", "DistalPhalanxOutlineCorrect"],
        help="List of datasets to process."
    )
    parser.add_argument("--path2save", type=str, help="save csv(s) results at path.")
    parser.add_argument("--N", nargs="+", type=int, default=list(range(1, 10)), help="#Neurons/nodes.")
    parser.add_argument("--G", nargs="+", type=int, default=[3, 5, 10, 20, 50], help="Grids sizes.")
    parser.add_argument("--K", nargs="+", type=int, default=[3, 5], help="Spline order.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(os.path.exists(args.path2save))
    if not os.path.exists(args.path2save):
        os.makedirs(args.path2save)

    for dataset_name in args.datasets:
        print(f"Processing dataset: {dataset_name}")
        dataset_manager = DatasetManager(dataset_name, device)
        grid_search = GridSearch(dataset_manager, device)
        grid_search.run(N=args.N, G=args.G, K=args.K)
        grid_search.save_results(f"{args.path2save}/kan_{dataset_name}_hsearch_results.csv")