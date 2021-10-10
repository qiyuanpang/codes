# import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter

def ctrain_loop(dataloader, model, loss_fn, optimizer, epoches=10):
    size = len(dataloader.dataset)
    for epoch in range(epoches):
        print('epoch: ', epoch)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X.cfloat())
            # print(pred.dtype, y.dtype)
            loss = loss_fn(pred, y.cfloat())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def ctest_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, test_loss_rel = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.cfloat())
            test_loss += loss_fn(pred, y).item()
            test_loss_rel += loss_fn(pred, y).item()/loss_fn(torch.zeros(y.shape, dtype=torch.cfloat),y)
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    # correct /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    print(f"Test (rel) Error: \n Avg loss: {test_loss_rel:>8f} \n")



def train_loop(dataloader, model, loss_fn, optimizer, epoches=10):
    size = len(dataloader.dataset)
    for epoch in range(epoches):
        print('epoch: ', epoch)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X.double())
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()/loss_fn(torch.zeros(y.shape),y)
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    # correct /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")