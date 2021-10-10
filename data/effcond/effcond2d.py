import numpy as np
import json
import torch
from torch.utils.data import Dataset
import csv
import pandas as pd

class EffCond2D():
    def __init__(self, n):
        self.xi = np.random.rand(2)
        self.xi = self.xi/np.linalg.norm(self.xi)
        self.n = n
        self.h = 1/n
        self.x = np.meshgrid(np.arange(0,1,1/n), np.arange(0,1,1/n))
        self.x = np.array([np.ndarray.flatten(self.x[0]), np.ndarray.flatten(self.x[1])])
        self.construct_a()
        self.construct_L()
        self.u = np.linalg.solve(self.L, -self.b)
        self.effcond = -np.dot(self.u, np.matmul(self.L, self.u)) - 2*np.dot(self.u, self.b) + np.sum(self.a)/self.n**2

    def construct_a(self):
        def construct_P(x, h):
            n = max(x.shape)
            PPT = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    PPT[i,j] = np.exp(-np.linalg.norm(x[:,i]-x[:,j])**2/4/h**2)
            L = np.linalg.cholesky(PPT)
            return L
        b = np.random.uniform(low=0.3, high=5, size=n**2)
        P = construct_P(self.x, self.h)
        self.a = np.matmul(P, b)

    def construct_L(self):
        self.L = np.zeros((self.n**2, self.n**2))
        self.b = np.zeros(self.n**2)
        for i in range(self.n**2):
            PREV = i-1 if i % self.n > 0 else i + self.n-1
            NEXT = i+1 if i % self.n < self.n-1 else i - self.n + 1
            self.L[i, PREV] = self.a[i] + self.a[PREV]
            self.L[i, NEXT] = self.a[i] + self.a[NEXT]
            self.L[i, i] = -(2*self.a[i] + self.a[PREV] + self.a[NEXT])
            self.b[i] += self.xi[0]*(self.a[NEXT] - self.a[PREV])
            PREV = i-self.n if int(i/self.n) > 0 else i + self.n*(self.n-1)
            NEXT = i+self.n if int(i/self.n) < self.n-1 else i - self.n*(self.n-1)
            self.L[i, PREV] = self.a[i] + self.a[PREV]
            self.L[i, NEXT] = self.a[i] + self.a[NEXT]
            self.L[i, i] += -(2*self.a[i] + self.a[PREV] + self.a[NEXT])
            self.b[i] += self.xi[1]*(self.a[NEXT] - self.a[PREV])
        self.L = self.L/2/self.h**2
        self.b = self.b/2/self.h

class EffCond2DDataSet(Dataset):
    def __init__(self, X_file, label_file):
        self.X = np.load(X_file)
        self.labels = np.load(label_file)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.X[index], self.labels[index]


if __name__ == '__main__':
    N = 128
    n = 20
    data = np.zeros((N,2,n,n))
    label = np.zeros(N)
    for i in range(N):
        print('example: ', i)
        d = EffCond2D(n)
        data[i,0,:,0:n] = np.reshape(d.u, (n,n))
        data[i,1,:,0:n] = np.reshape(d.a, (n,n))
        label[i] = d.effcond
    np.save('effcond2d_X.npy', data)
    np.save('effcond2d_label.npy', label)
    # pd.DataFrame(data).to_csv('effcond2d.csv')