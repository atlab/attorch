import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import _assert_no_grad
import torch
import numpy as np

class PoissonLoss(nn.Module):
    def __init__(self, bias=1e-12):
        super().__init__()
        self.bias = bias

    def forward(self, output, target):
        _assert_no_grad(target)
        return (output - target * torch.log(output + self.bias)).mean()


class AvgCorr(nn.Module):
    def __init__(self, eps=1e-12):
        self.eps = eps
        super().__init__()

    def forward(self, output, target):
        _assert_no_grad(target)
        delta_out = (output - output.mean(0).expand_as(output))
        delta_target = (target - target.mean(0).expand_as(target))

        var_out = delta_out.pow(2).mean(0)
        var_target = delta_target.pow(2).mean(0)

        corrs = (delta_out * delta_target).mean(0) / ((var_out + self.eps) * (var_target + self.eps)).sqrt()
        return corrs.mean()


class Corr(nn.Module):
    def __init__(self, eps=1e-12):
        self.eps = eps
        super().__init__()

    def forward(self, output, target):
        _assert_no_grad(target)
        delta_out = (output - output.mean(0).expand_as(output))
        delta_target = (target - target.mean(0).expand_as(target))

        var_out = delta_out.pow(2).mean(0)
        var_target = delta_target.pow(2).mean(0)

        corrs = (delta_out * delta_target).mean(0) / ((var_out + self.eps) * (var_target + self.eps)).sqrt()
        return corrs

class UnnormalizedCorr(nn.Module):
    def __init__(self, eps=1e-12):
        self.eps = eps
        super().__init__()

    def forward(self, output, target):
        _assert_no_grad(target)
        delta_out = (output - output.mean(0).expand_as(output))
        delta_target = (target - target.mean(0).expand_as(target))

        var_out = delta_out.pow(2).mean(0)
        var_target = delta_target.pow(2).mean(0)

        corrs = (delta_out * delta_target).sum(0) / ((var_out + self.eps) * (var_target + self.eps)).sqrt()
        return corrs, delta_out.size(0)