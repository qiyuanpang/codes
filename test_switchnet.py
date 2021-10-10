import torch
from torch import nn
from torch.utils.data import DataLoader
from nn.switchnet import SwitchForward, SwitchInverse
from loss.customloss import CMSELoss
from data.cranddataset import CRandDataset
from data.scattering.scattering import ScatteringDataSet
from train.train import ctrain_loop, ctest_loop

# N - M + w - 1 should be even
t = 4
Pd = 3**2
Px = 9**2
N = 81
w = 11
alpha = 24
L = 3
M = 81
Nsample = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'

learning_rate = 0.002
batch_size = 64
epoches = 5

filename = 'data/scafull2.h5'
train_data = ScatteringDataSet(filename)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
testfile = 'data/scafull2_test.h5'
test_data = ScatteringDataSet(testfile)
test_dataloader = DataLoader(test_data, batch_size=1)

model = SwitchForward(t, Pd, Px, N, w, alpha, L, M, device).float()
loss_fn = CMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


ctrain_loop(train_dataloader, model, loss_fn, optimizer, epoches=epoches)
ctest_loop(test_dataloader, model, loss_fn)