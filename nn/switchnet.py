import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexReLU
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

class SwitchInverse(nn.Module):
    def __init__(self, t, Pd, Px, N, w, alpha, L, M, device):
        super(SwitchInverse, self).__init__()
        self.t = t
        self.Pd = Pd
        self.Px = Px
        self.N = N
        self.w = w
        self.alpha = alpha
        self.L = L
        self.M = M
        self.fnns_u = nn.ModuleList()
        self.device = device
        self.padding = [int(w[i]/2 - 0.5) for i in range(len(w))] if isinstance(w, list) else int(w/2-0.5)
        for i in range(Pd):
            self.fnns_u.append(ComplexLinear(int(M*M/Pd), t*Px))
        self.fnns_v = nn.ModuleList()
        for i in range(Px):
            self.fnns_v.append(ComplexLinear(t*Pd, int(N*N/Px)))
        self.cnn = nn.ModuleList()
        self.cnn.append(ComplexConv2d(in_channels=1, out_channels=alpha, kernel_size=w, padding=self.padding))
        for i in range(L):
            self.cnn.append(ComplexConv2d(in_channels=alpha, out_channels=alpha, kernel_size=w, padding=self.padding))
        self.cnn.append(ComplexConv2d(in_channels=alpha, out_channels=1, kernel_size=w, padding=self.padding))
        


    def vect(self, z):
        P = self.Pd
        Nsample, n, n = z.shape
        sqrtP = int(np.sqrt(P))
        y = z.reshape((Nsample, sqrtP, int(n/sqrtP), sqrtP, int(n/sqrtP)))
        y = torch.transpose(y, 2, 3)
        y = y.reshape((Nsample, n*n))
        return y

    def square(self, z):
        P = self.Px
        Nsample, n2 = z.shape
        n = int(np.sqrt(n2))
        sqrtP = int(np.sqrt(P))
        y = z.reshape((Nsample, sqrtP, sqrtP, int(n/sqrtP), int(n/sqrtP)))
        y = torch.transpose(y, 2, 3)
        y = y.reshape((Nsample, n,n))
        return y

    def switch(self, z):
        Nsample, _ = z.shape
        cols = int(self.M*self.M/self.Pd)
        Uz = self.fnns_u[0](z[:,:cols])
        cnt = 1
        for i in range(1,self.Pd):
            Uz = torch.cat((Uz, self.fnns_u[i](z[:,cnt*cols:(cnt+1)*cols])), 1)
            cnt += 1
        Uz = Uz.reshape((Nsample, self.Px, self.Pd, self.t))
        Uz = torch.transpose(Uz, 1, 2)
        Uz = Uz.reshape((Nsample, self.Pd*self.Px*self.t))

        cols = self.t*self.Pd
        Vz = self.fnns_v[0](Uz[:,:cols])
        cnt = 1
        for i in range(1,self.Px):
            Vz = torch.cat((Vz, self.fnns_v[i](Uz[:,cnt*cols:(cnt+1)*cols])), 1)
            cnt += 1
        return Vz

    def forward(self, x):
        d = self.vect(x)
        d = self.switch(d)
        d = self.square(d)
        d = torch.unsqueeze(d, 1)
        for i in range(self.L+1):
            d = self.cnn[i](d)
            d = ComplexReLU()(d)
        y = self.cnn[-1](d)
        y = torch.squeeze(y, 1)
        return y


class SwitchForward(nn.Module):
    def __init__(self, t, Pd, Px, N, w, alpha, L, M, device):
        super(SwitchForward, self).__init__()
        self.t = t
        self.Pd = Pd
        self.Px = Px
        self.N = N
        self.w = w
        self.alpha = alpha
        self.L = L
        self.M = M
        self.fnns_u = nn.ModuleList()
        self.device = device
        self.padding = [int((N - M + w[i])/2 - 0.5) for i in range(len(w))] if isinstance(w, list) else int((N- M + w)/2-0.5)
        for i in range(Pd):
            self.fnns_u.append(ComplexLinear(int(M*M/Pd), t*Px))
        self.fnns_v = nn.ModuleList()
        for i in range(Px):
            self.fnns_v.append(ComplexLinear(t*Pd, int(N*N/Px)))
        self.cnn = nn.ModuleList()
        self.cnn.append(ComplexConv2d(in_channels=1, out_channels=alpha, kernel_size=w, padding=self.padding))
        for i in range(L):
            self.cnn.append(ComplexConv2d(in_channels=alpha, out_channels=alpha, kernel_size=w, padding=self.padding))
        self.cnn.append(ComplexConv2d(in_channels=alpha, out_channels=1, kernel_size=w, padding=self.padding))
        


    def vect(self, z):
        P = self.Pd
        Nsample, n, n = z.shape
        sqrtP = int(np.sqrt(P))
        y = z.reshape((Nsample, sqrtP, int(n/sqrtP), sqrtP, int(n/sqrtP)))
        y = torch.transpose(y, 2, 3)
        y = y.reshape((Nsample, n*n))
        return y

    def square(self, z):
        P = self.Px
        Nsample, n2 = z.shape
        n = int(np.sqrt(n2))
        sqrtP = int(np.sqrt(P))
        y = z.reshape((Nsample, sqrtP, sqrtP, int(n/sqrtP), int(n/sqrtP)))
        y = torch.transpose(y, 2, 3)
        y = y.reshape((Nsample, n,n))
        return y

    def switch(self, z):
        Nsample, _ = z.shape
        cols = int(self.M*self.M/self.Pd)
        Uz = self.fnns_u[0](z[:,:cols])
        cnt = 1
        for i in range(1,self.Pd):
            Uz = torch.cat((Uz, self.fnns_u[i](z[:,cnt*cols:(cnt+1)*cols])), 1)
            cnt += 1
        Uz = Uz.reshape((Nsample, self.Px, self.Pd, self.t))
        Uz = torch.transpose(Uz, 1, 2)
        Uz = Uz.reshape((Nsample, self.Pd*self.Px*self.t))

        cols = self.t*self.Pd
        Vz = self.fnns_v[0](Uz[:,:cols])
        cnt = 1
        for i in range(1,self.Px):
            Vz = torch.cat((Vz, self.fnns_v[i](Uz[:,cnt*cols:(cnt+1)*cols])), 1)
            cnt += 1
        return Vz

    def forward(self, x):
        d = torch.unsqueeze(x, 1)
        for i in range(self.L+1):
            d = self.cnn[i](d)
            d = ComplexReLU()(d)
        d = self.cnn[-1](d)
        d = torch.squeeze(d, 1)
        d = self.vect(d)
        d = self.switch(d)
        y = self.square(d)

        return y
        
        



if __name__ == '__main__':
    t = 3
    Pd = 16
    Px = 16
    N = 32
    w = 5
    alpha = 4
    L = 3
    M = 32
    Nsample = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test = SwitchInverse(t, Pd, Px, N, w, alpha, L, M, device).float()
    z = torch.randn((M,M), dtype=torch.cfloat)
    # z = z.reshape((M,M))
    z = torch.unsqueeze(z, 0)
    z = z.repeat(Nsample, 1, 1)
    y = test.forward(z)
    print(y.shape, y)