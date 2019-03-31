import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np

def _assert_no_grad(tensor):
    assert not tensor.requires_grad


class PoissonLoss(nn.Module):
    def __init__(self, bias=1e-12, per_neuron=False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron

    def forward(self, output, target):
        _assert_no_grad(target)
        loss = (output - target * torch.log(output + self.bias))
        if not self.per_neuron:
            return loss.mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)


class PoissonLoss3d(nn.Module):
    def __init__(self, bias=1e-16, per_neuron=False):
        super().__init__()
        self.bias = bias
        self.per_neuron = per_neuron

    def forward(self, output, target):
        _assert_no_grad(target)
        lag = target.size(1) - output.size(1)
        loss =  (output - target[:, lag:, :] * torch.log(output + self.bias))
        if not self.per_neuron:
            return loss.mean()
        else:
            return loss.view(-1, loss.shape[-1]).mean(dim=0)


class L1Loss3d(nn.Module):
    def __init__(self, bias=1e-12):
        super().__init__()
        self.bias = bias

    def forward(self, output, target):
        _assert_no_grad(target)
        lag = target.size(1) - output.size(1)
        return (output - target[:, lag:, :]).abs().mean()


class MSE3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        _assert_no_grad(target)
        lag = target.size(1) - output.size(1)
        return (output - target[:, lag:, :]).pow(2).mean()


class AvgCorr(nn.Module):
    def __init__(self, eps=1e-12):
        self.eps = eps
        super().__init__()

    def forward(self, output, target):
        _assert_no_grad(target)
        delta_out = (output - output.mean(0, keepdim=True).expand_as(output))
        delta_target = (target - target.mean(0, keepdim=True).expand_as(target))

        var_out = delta_out.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)

        corrs = (delta_out * delta_target).mean(0, keepdim=True) / (
                (var_out + self.eps) * (var_target + self.eps)).sqrt()
        return corrs.mean()


class Corr(nn.Module):
    def __init__(self, eps=1e-12):
        self.eps = eps
        super().__init__()

    def forward(self, output, target):
        _assert_no_grad(target)
        delta_out = (output - output.mean(0, keepdim=True).expand_as(output))
        delta_target = (target - target.mean(0, keepdim=True).expand_as(target))

        var_out = delta_out.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)

        corrs = (delta_out * delta_target).mean(0, keepdim=True) / (
                (var_out + self.eps) * (var_target + self.eps)).sqrt()
        return corrs


class UnnormalizedCorr(nn.Module):
    def __init__(self, eps=1e-12):
        self.eps = eps
        super().__init__()

    def forward(self, output, target):
        _assert_no_grad(target)
        delta_out = (output - output.mean(0, keepdim=True).expand_as(output))
        delta_target = (target - target.mean(0, keepdim=True).expand_as(target))

        var_out = delta_out.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)

        corrs = (delta_out * delta_target).sum(0, keepdim=True) / (
                (var_out + self.eps) * (var_target + self.eps)).sqrt()
        return corrs, delta_out.size(0)
