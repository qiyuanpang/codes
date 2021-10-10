import torch
from torch.utils.data import Dataset

class CRandDataset(Dataset):
    def __init__(self, size_input, size_target):
        self.size_input = size_input
        self.size_target = size_target
        self.data = torch.randn(self.size_input, dtype=torch.cfloat)
        self.labels = torch.randn(self.size_target, dtype=torch.cfloat)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.data[idx]
        label = self.labels[idx]
        return X, label