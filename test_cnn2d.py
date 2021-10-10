import torch
from torch import nn
from torch.utils.data import DataLoader
from nn.cnn2d import CNN2D
# from data.randdataset import RandDataset
from data.effcond.effcond2d import EffCond2DDataSet
from train.train import train_loop, test_loop

N = 128
M = 10
n = 20
alpha = 50

learning_rate = 0.00001
batch_size = 32
epochs = 5

model = CNN2D(n, M, alpha).double()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


train_data = EffCond2DDataSet('./data/effcond/effcond2d_X.npy', './data/effcond/effcond2d_label.npy')
train_dataloader = DataLoader(train_data, batch_size=batch_size)

train_loop(train_dataloader, model, loss_fn, optimizer)
test_loop(train_dataloader, model, loss_fn)
