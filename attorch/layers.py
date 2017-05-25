import torch
from .constraints import positive
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
# from .module import Module
from torch.nn import Parameter


class Offset(nn.Module):
    def __init__(self, offset=1, **kwargs):
        super().__init__()
        self.offset = offset

    def forward(self, x):
        return x + self.offset


class Elu1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.elu(x) + 1.


class Conv2dPad(nn.Conv2d):
    """
    Padded Conv2d layer. Pads with reflect by default.
    """

    def __init__(self, in_channels, out_channels, kernel_size, pad=None, mode='reflect', **kwargs):
        assert 'padding' not in kwargs, 'You need to use `pad` not `padding`'
        self.padding = _pair(0)
        super().__init__(in_channels, out_channels, kernel_size, padding=0, **kwargs)
        self.mode = mode

        if isinstance(pad, tuple) or pad is None:
            self.pad = pad
        else:
            self.pad = 4 * (pad,)

    def _pad(self, input):
        if self.pad is not None and self.pad != 4 * (0,):
            input = F.pad(input, mode=self.mode, pad=self.pad)
        return input

    def forward(self, input):
        input = self._pad(input)
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class BiasBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, features, **kwargs):
        super().__init__(features, affine=False, **kwargs)
        bias = nn.Parameter(torch.FloatTensor(1, features, 1, 1))
        self.register_parameter('bias', bias)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(input, self.running_mean, self.running_var, weight=None, bias=None,
                            training=self.training, momentum=self.momentum, eps=self.eps)
        N, _, w, h = input.size()
        return input + self.bias.repeat(N, 1, w, h)


class SpatialXFeatureLinear(nn.Module):
    """
    Factorized fully connected layer. Weights are a sum of outer products between a spatial filter and a feature vector.  
    """

    def __init__(self, in_shape, outdims, components=1, bias=True, normalize=True, positive=False):
        super().__init__()
        self.in_shape = in_shape
        c, w, h = self.in_shape
        self.outdims = outdims
        self.components = components
        self.normalize = normalize
        self.positive = positive

        spatial = Parameter(torch.Tensor(self.outdims, 1, w, h, components))
        features = Parameter(torch.Tensor(self.outdims, c, 1, 1, components))
        self.register_parameter('spatial', spatial)
        self.register_parameter('features', features)

        if bias:
            bias = Parameter(torch.Tensor(self.outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.initialize()

    @property
    def normalized_spatial(self):
        c, w, h = self.in_shape
        if self.positive:
            positive(self.spatial)
        if self.normalize:
            weight = self.spatial / (self.spatial.pow(2).sum(2).sum(3).sqrt().repeat(1, 1, w, h, 1) + 1e-6)
        else:
            weight = self.spatial
        return weight

    @property
    def weight(self):
        c, w, h = self.in_shape
        weight = self.normalized_spatial.repeat(1, c, 1, 1, 1) \
                 * self.features.repeat(1, 1, w, h, 1)
        weight = weight.sum(4).view(self.outdims, -1)
        return weight

    def initialize(self, init_noise=1e-3):
        self.spatial.data.normal_(0, init_noise)
        self.features.data.normal_(0, init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)


    @property
    def basis(self):
        c, w, h = self.in_shape
        return self.weight.view(-1, c, w, h).data.cpu().numpy()

    def forward(self, x):
        N = x.size(0)
        y = x.view(N, -1) @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias.expand_as(y)
        return y

    def __repr__(self):
        return ('spatial positive ' if self.positive else '') + \
               ('normalized ' if self.normalize else '') + \
               self.__class__.__name__ + \
               ' (' + '{} x {} x {}'.format(*self.in_shape) + ' -> ' + str(
            self.outdims) + ') rank {}'.format(self.components)


class WidthXHeightXFeatureLinear(nn.Module):
    """
    Factorized fully connected layer. Weights are a sum of outer products between three vectors over width,
    height and spatial.  
    """

    def __init__(self, in_shape, outdims, components=1, bias=True, normalize=True, positive=False):
        super().__init__()
        self.in_shape = in_shape

        c, w, h = self.in_shape
        self.outdims = outdims
        self.normalize = normalize
        self.positive = positive
        self.components = components

        width = Parameter(torch.Tensor(self.outdims, 1, w, 1, components))
        height = Parameter(torch.Tensor(self.outdims, 1, 1, h, components))
        features = Parameter(torch.Tensor(self.outdims, c, 1, 1, components))
        self.register_parameter('width', width)
        self.register_parameter('height', height)
        self.register_parameter('features', features)

        if bias:
            bias = Parameter(torch.Tensor(self.outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.initialize()

    def initialize(self, init_noise=1e-3):
        self.width.data.normal_(0, init_noise)
        self.height.data.normal_(0, init_noise)
        self.features.data.normal_(0, init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

    @property
    def normalized_width(self):
        c, w, h = self.in_shape
        if self.positive:
            positive(self.width)
        if self.normalize:
            return self.width / (self.width.pow(2).sum(2).sqrt().repeat(1, 1, w, 1, 1) + 1e-6)
        else:
            return self.width

    @property
    def normalized_height(self):
        c, w, h = self.in_shape
        if self.positive:
            positive(self.height)
        if self.normalize:
            return self.height / (self.height.pow(2).sum(3).sqrt().repeat(1, 1, 1, h, 1) + 1e-6)
        else:
            return self.height

    @property
    def spatial(self):
        c, w, h = self.in_shape
        weight = self.normalized_width.repeat(1, 1, 1, h, 1) \
                 * self.normalized_height.repeat(1, 1, w, 1, 1)
        return weight

    @property
    def weight(self):
        c, w, h = self.in_shape

        weight = self.normalized_width.repeat(1, c, 1, h, 1) \
                 * self.normalized_height.repeat(1, c, w, 1, 1) \
                 * self.features.repeat(1, 1, w, h, 1)
        weight = weight.sum(4).view(self.outdims, -1)
        return weight

    @property
    def basis(self):
        c, w, h = self.in_shape
        return self.weight.view(-1, c, w, h).data.cpu().numpy()

    def forward(self, x):

        N = x.size(0)
        y = x.view(N, -1) @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias.expand_as(y)
        return y

    def __repr__(self):
        return ('spatial positive ' if self.positive else '') + \
               ('normalized ' if self.normalize else '') + \
               self.__class__.__name__ + \
               ' (' + '{} x {} x {}'.format(*self.in_shape) + ' -> ' + str(self.outdims) + ') rank {}'.format(
            self.components)
