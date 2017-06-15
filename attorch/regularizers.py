import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch


def laplace():
    return np.array([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]).astype(np.float32)[None, None, ...]


class Laplace(nn.Module):
    """
    Laplace filter for a stack of data.  
    """
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, bias=False, padding=1)
        self.conv.weight.data.copy_(torch.from_numpy(laplace()))
        self.conv.weight.requires_grad = False


    def forward(self, x):
        return self.conv(x)


class LaplaceL2(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer. 
    """
    def __init__(self):
        super().__init__()
        self.laplace = Laplace()

    def forward(self, x):
        ic, oc, k1, k2 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2)).pow(2).mean() / 2

class LaplaceL1(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """
    def __init__(self):
        super().__init__()
        self.laplace = Laplace()

    def forward(self, x):
        ic, oc, k1, k2 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2)).abs().mean()
