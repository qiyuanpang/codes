import numpy as np
import json
import torch
from torch.utils.data import Dataset
import h5py

class ScatteringDataSet():
    def __init__(self, filename):
        self.data = h5py.File(filename, "r")
        self.input = np.array(self.data['Input']) + np.array(self.data['Input2'])*1j
        self.output = np.array(self.data['Output']) + np.array(self.data['Output2'])*1j
        self.adjoint = np.array(self.data['Adjoint']) + np.array(self.data['Adjoint2'])*1j
        # print(self.input.shape, self.output.shape)

    def __len__(self):
        return len(self.output)

    def __getitem__(self, index):
        return self.input[index], self.output[index]

if __name__ == "__main__":
    filename = '../scafull2.h5'
    data = ScatteringDataSet(filename)