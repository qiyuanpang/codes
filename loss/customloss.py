import torch
import torch.nn as nn
import torch.nn.functional as F

class CMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return F.mse_loss(output.real, target.real) + F.mse_loss(output.imag, target.imag)