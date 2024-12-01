from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder
from kan.utils import create_dataset_from_data
import torch 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class DatasetManager:
    def __init__(self, name, device, batch_size=32):
        self.dtype = torch.get_default_dtype()
        self.dataset_name = name
        self.batch_size = batch_size
        self.device = device
        self.label_encoder = LabelEncoder()
        self.load_data()
    
    def get_classes_number(self):   
        return len(np.unique(self.y_train))

    def transform(self, X, y):
        X = torch.as_tensor(X).type(self.dtype)
        y = torch.as_tensor(np.array(y, dtype=int))
        return X, y

    def process_data(self):
        X_train, y_train = load_classification(name=self.dataset_name, split='train')
        X_test, y_test = load_classification(name=self.dataset_name, split='test')
        
        y_train = self.label_encoder.fit_transform(y_train) 
        y_test = self.label_encoder.transform(y_test)

        X_train, y_train = self.transform(X_train.squeeze(1), y_train)
        X_test, y_test = self.transform(X_test.squeeze(1), y_test)
        return X_train, y_train, X_test, y_test

    def load_data(self):
        self.X_train, self.y_train, self.X_test, self.y_test = self.process_data()
        self.dataset = create_dataset_from_data(self.X_train, self.y_train, device=self.device)

    def split_data(self, val_size=0.2):
        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=val_size, random_state=42)
        return X_train, X_val, y_train, y_val

    
    def load_dataloader(self):
        X_train, X_val, y_train, y_val = self.split_data()
        self.X_test, self.y_test = self.transform(self.X_test, self.y_test)
        X_train, X_val, self.X_test = X_train.unsqueeze(1), X_val.unsqueeze(1), self.X_test.unsqueeze(1)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, self.batch_size, shuffle=False)

        test_dataset = TensorDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False)
        torch.save(test_dataset, f'testset/{self.dataset_name}.pt')
        return train_loader, val_loader, test_loader