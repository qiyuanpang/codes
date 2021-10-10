import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader



class Component1(nn.Module):
    def __init__(self, n, alpha, device):
        super(Component1, self).__init__()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv_four_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=alpha, kernel_size=(5,5), device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=alpha, out_channels=alpha, kernel_size=(1,1), device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=alpha, out_channels=alpha, kernel_size=(1,1), device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=alpha, out_channels=1, kernel_size=(1,1), device=device),
        )
    
    def forward(self, x):
        y = self.conv_four_layers(x)
        return y

class Component2(nn.Module):
    def __init__(self, n, alpha, device):
        super(Component2, self).__init__()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv_six_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=alpha, kernel_size=(5,5), device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=alpha, out_channels=alpha, kernel_size=(1,1), device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=alpha, out_channels=alpha, kernel_size=(1,1), device=device),
            nn.ReLU(),
            nn.AvgPool2d((n,n), divisor_override=1),
            nn.Flatten(1),
            nn.Linear(alpha, alpha, device=device),
            nn.ReLU(),
            nn.Linear(alpha, 1, device=device),
            nn.Flatten(0)
        )

    def forward(self, x):
        y = self.conv_six_layers(x)
        return y

class CNN2D(nn.Module):
    def __init__(self, n, M, alpha):
        super(CNN2D, self).__init__()
        self.M = M
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Component1s = nn.ModuleList()
        self.Component2s = nn.ModuleList()
        self.Component2s.append(Component2(n, alpha, self.device))
        for i in range(M):
            self.Component1s.append(Component1(n, alpha, self.device))

    def forward(self, inputs):
        u = torch.unsqueeze(inputs[:,0,:,:], 1)
        a = torch.unsqueeze(inputs[:,1,:,:], 1)
        u0 = u
        for i in range(self.M):
            # print(u0.size())
            x = torch.cat((u0, a), 1)
            xp = F.pad(x, (0,4,0,4), mode='circular')
            u0 = self.Component1s[i](xp)
        x = torch.cat((u0, a), 1)
        xp = F.pad(x, (0,4,0,4), mode='circular')
        output = self.Component2s[0](xp)
        return output