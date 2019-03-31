import numpy as np
import torch
import torch.nn as nn
from itertools import product
from torch.nn import functional as F
#import pytorch_fft.fft as fft


# def laplace():
#     return np.array([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]).astype(np.float32)[None, None, ...]

def laplace():
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)[None, None, ...]


def laplace3d():
    l = np.zeros((3, 3, 3))
    l[1, 1, 1] = -6.
    l[1, 1, 2] = 1.
    l[1, 1, 0] = 1.
    l[1, 0, 1] = 1.
    l[1, 2, 1] = 1.
    l[0, 1, 1] = 1.
    l[2, 1, 1] = 1.
    return l.astype(np.float32)[None, None, ...]


#def fft_smooth(grad, factor=1/4):
#    """
#    Tones down the gradient with (1/f)**(2 * factor) filter in the Fourier domain.
#    Equivalent to low-pass filtering in the spatial domain.
#
#    `grad` is an at least 2D CUDA Tensor, where the last two dimensions are treated
#    as images to apply smoothening transformation.
#
#    `factor` controls the strength of the fall off.
#    """
#    h, w = grad.size()[-2:]
#    tw = np.minimum(np.arange(0, w), np.arange(w, 0, -1), dtype=np.float32)#[-(w+2)//2:]
#    th = np.minimum(np.arange(0, h), np.arange(h, 0, -1), dtype=np.float32)
#    t = 1 / np.maximum(1.0, (tw[None,:] ** 2 + th[:,None] ** 2) ** (factor))
#    F = torch.Tensor(t / t.mean()).cuda()
#    rp, ip = fft.fft2(grad.data, torch.zeros_like(grad.data))
#    return Variable(fft.ifft2(rp * F, ip * F)[0])


class Laplace(nn.Module):
    """
    Laplace filter for a stack of data.
    """

    def __init__(self, padding=0):
        super().__init__()
        self._padding = padding
        self.register_buffer('filter', torch.from_numpy(laplace()))

    def forward(self, x):
        return F.conv2d(x, self.filter, padding=self._padding, bias=None)


class Laplace3d(nn.Module):
    """
    Laplace filter for a stack of data.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('filter', torch.from_numpy(laplace3d()))

    def forward(self, x):
        return F.conv3d(x, self.filter, bias=None)


class LaplaceL2(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer. 
    """

    def __init__(self, padding=0):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x, weights=None):
        ic, oc, k1, k2 = x.size()
        if weights is None:
            weights = 1.0
        return (self.laplace(x.view(ic * oc, 1, k1, k2)).view(ic, oc, k1, k2).pow(2) * weights).mean() / 2


class LaplaceL23d(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace3d()

    def forward(self, x):
        ic, oc, k1, k2, k3 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2, k3)).pow(2).mean() / 2


class FlatLaplaceL23d(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace()

    def forward(self, x):
        ic, oc, k1, k2, k3 = x.size()
        assert k1 == 1, 'time dimension must be one'
        return self.laplace(x.view(ic * oc, 1, k2, k3)).pow(2).mean() / 2


class LaplaceL1(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self, padding=0):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x):
        ic, oc, k1, k2 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2)).abs().mean()
