from typing import Dict
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class IntrusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_name: str, labels_mapping: Dict[str, int]) -> None:
        self.df = df
        self.target_name = target_name
        self.labels_mapping = labels_mapping
    
    def __getitem__(self, index):
        row = self.df.iloc[index, :]
        features = row.drop(self.target_name)
        label = row[self.target_name]

        label = self.labels_mapping[label]

        return features.to_numpy(dtype=np.float32), label
    
    def __len__(self):
        return len(self.df)

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        return self.X[index, :].astype(np.float32), self.y[index]
    
    def __len__(self):
        return self.y.shape[0]