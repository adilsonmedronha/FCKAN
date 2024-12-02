from aeon.classification.deep_learning import FCNClassifier, MLPClassifier, TimeCNNClassifier
from aeon.datasets import load_classification
from sklearn.metrics import accuracy_score, f1_score
from argparse import ArgumentParser
import os
import csv
import numpy as np
import torch
import random 

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

os.environ['AEON_DEPRECATION_WARNING'] = 'False'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CLASSIFIERS = {
    "FCN": FCNClassifier,
    "MLP": MLPClassifier,
    "CNN": TimeCNNClassifier,
}

def train_and_evaluate(classifier_name, dataset_name, X_train, y_train, X_test, y_test):
    print(f"Treinando {classifier_name}...")
    classifier_cls = CLASSIFIERS[classifier_name]
    model = classifier_cls(save_best_model=True, 
                           file_path='best_weights/', 
                           best_file_name=f'{classifier_name}_{dataset_name}', 
                           verbose=True)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{classifier_name} Accuracy: {accuracy:.4f} F1 Score: {f1:.4f}")
    return accuracy, f1

def save_to_csv(filepath, dataset_name, classifier_name, accuracy, f1_score):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Dataset", "Classifier", "Accuracy", 'F1 Score'])
        writer.writerow([dataset_name, classifier_name, accuracy, f1_score])

def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--datasets", type=str, nargs="+", default=["HandOutlines"])
    parser.add_argument(
        "--classifiers",
        type=str,
        nargs="+",
        default=["FCN", "MLP", "CNN"]    
    )
    parser.add_argument("--output_csv", type=str, default="results.csv")
    args = parser.parse_args()
    print(type(args.seed), args.seed)
    set_seed(args.seed) 

    for dataset_name in args.datasets:
        print(f"\nCarregando dataset: {dataset_name}")
        try:
            X_train, y_train = load_classification(name=dataset_name, split='train')
            X_test, y_test = load_classification(name=dataset_name, split='test')
        except Exception as e:
            print(f"Erro ao carregar o dataset {dataset_name}: {e}")
            continue

        for clf_name in args.classifiers:
            accuracy, f1_score = train_and_evaluate(clf_name, dataset_name, X_train, y_train, X_test, y_test)

            save_to_csv(args.output_csv, dataset_name, clf_name, accuracy, f1_score)
    print(f"\nResultados salvos em {args.output_csv}")

if __name__ == "__main__":
    main()
