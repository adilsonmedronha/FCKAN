import torch
from torch import nn
from fckan import FCKAN
from utils.data import DatasetManager
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import argparse
import os
import pickle

def eval(model, criterion, val_loader, device):
    model.eval()
    correct_preds = 0
    total_preds = 0
    running_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
            running_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')  
    return avg_loss, accuracy, f1


def run(model, train_loader, val_loader, criterion, optimizer, num_epochs, path2bestmodel, idx, device):
    model.train()  
    model.to(device)
    tran_loss_per_epoch = []
    val_loss_per_epoch = []
    train_acc_per_epoch = []
    val_acc_per_epoch = []
    val_f1_per_epoch = []
    min_loss = float('inf')

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
        
        avg_val_loss, accuracy_val, f1_val = eval(model, criterion, val_loader, device)
        avg_loss = running_loss / len(train_loader)
        tran_loss_per_epoch.append(running_loss)
        val_loss_per_epoch.append(avg_val_loss)
        train_acc_per_epoch.append(correct_preds / total_preds)
        val_acc_per_epoch.append(accuracy_val)
        val_f1_per_epoch.append(f1_val)

        if avg_val_loss < min_loss:
            min_loss = avg_val_loss
            best_model = model.state_dict() 
            torch.save(best_model, f'{path2bestmodel}/run_{idx}_{dataset_name}_n5_g5_k3_best_model.pth')

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss {avg_val_loss:.4f}")
    
    return {'train_loss': tran_loss_per_epoch, 
            'val_loss': val_loss_per_epoch, 
            'train_acc': train_acc_per_epoch, 
            'val_acc': val_acc_per_epoch, 
            'val_f1': val_f1_per_epoch}
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Chinatown", "ItalyPowerDemand", "ECG200", "ArrowHead", "CricketX", "CricketY", "CricketZ", "Beef"],
        help="List of datasets to process.")
    parser.add_argument("--path2bestmodel", type=str, help="Path to save the best model.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnd_state = 42
    
    RUNS = 5
    for idx in range(RUNS):
        for dataset_name in args.datasets:
            print(f"Processing dataset: {dataset_name}")
            if not os.path.exists(f'{args.path2bestmodel}'): os.makedirs(f'{args.path2bestmodel}')
            datam = DatasetManager(name=dataset_name, device=device)
            n_classes = datam.get_classes_number()
            train_loader, val_loader, test_loader = datam.load_dataloader()

            x, y = next(iter(train_loader))
            print(x.shape, y.shape)
            print(f'Training... device {device}')
            fkan = FCKAN(n_classes, rnd_state, device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(fkan.parameters(), lr=0.001)
            num_epochs = 100
            log = run(fkan, 
                    train_loader, 
                    val_loader, 
                    criterion, 
                    optimizer, 
                    num_epochs, 
                    path2bestmodel=args.path2bestmodel,
                    idx=idx,
                    device=device)
            with open(f'log/run_{idx}_{dataset_name}_n5_g5_k3_optimization.pkl', 'wb') as f:
                pickle.dump(log, f)