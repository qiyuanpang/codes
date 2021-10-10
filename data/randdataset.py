import torch
from torch.utils.data import Dataset

class RandDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.rand(self.size)
        self.labels = torch.rand(self.size[0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.data[idx]
        label = self.labels[idx]
        return X, label