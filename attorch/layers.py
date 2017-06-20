import torch
from .constraints import positive
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import numpy as np
from math import ceil
# from .module import Module
from torch.nn import Parameter


class Offset(nn.Module):
    def __init__(self, offset=1):
        super().__init__()
        self.offset = offset

    def forward(self, x):
        return x + self.offset


class Elu1(nn.Module):
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


class SpatialXFeatureLinear(nn.Module):
    """
    Factorized fully connected layer. Weights are a sum of outer products between a spatial filter and a feature vector.  
    """

    def __init__(self, in_shape, outdims, bias=True, normalize=True, positive=False, spatial=None):
        super().__init__()
        self.in_shape = in_shape
        c, w, h = self.in_shape
        self.outdims = outdims
        self.normalize = normalize
        self.positive = positive

        self.spatial = Parameter(torch.Tensor(self.outdims, 1, w, h)) if spatial is None else spatial
        self.features = Parameter(torch.Tensor(self.outdims, c, 1, 1))

        if bias:
            bias = Parameter(torch.Tensor(self.outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.initialize()

    @property
    def normalized_spatial(self):
        if self.positive:
            positive(self.spatial)
        if self.normalize:
            weight = self.spatial / (self.spatial.pow(2).sum(2).sum(3).sqrt().expand_as(self.spatial) + 1e-6)
        else:
            weight = self.spatial
        return weight

    @property
    def weight(self):
        c, w, h = self.in_shape
        n = self.outdims
        weight = self.noormalized_spatial.expand(n, c, w, h) * self.features.expand(n, c, w, h)
        weight = weight.view(self.outdims, -1)
        return weight

    def initialize(self, init_noise=1e-3):
        self.spatial.data.normal_(0, init_noise)
        self.features.data.normal_(0, init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

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
            self.outdims) + ')'


class WidthXHeightXFeatureLinear(nn.Module):
    """
    Factorized fully connected layer. Weights are a sum of outer products between three vectors over width,
    height and spatial.  
    """

    def __init__(self, in_shape, outdims, components=1, bias=True, normalize=True, positive=False, width=None,
                 height=None, eps=1e-6):
        super().__init__()
        self.in_shape = in_shape
        self.eps = eps
        c, w, h = self.in_shape
        self.outdims = outdims
        self.normalize = normalize
        self.positive = positive
        self.components = components

        self.width = Parameter(torch.Tensor(self.outdims, 1, w, 1, components)) if width is None else width
        self.height = Parameter(torch.Tensor(self.outdims, 1, 1, h, components)) if height is None else height
        self.features = Parameter(torch.Tensor(self.outdims, c, 1, 1))
        assert self.width.size(4) == self.height.size(4), 'The number of components in width and height do not agree'
        self.components = self.width.size(4)
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
        if self.positive:
            positive(self.width)
        if self.normalize:
            return self.width / (self.width.pow(2).sum(2).sqrt().expand_as(self.width) + self.eps)
        else:
            return self.width

    @property
    def normalized_height(self):
        c, w, h = self.in_shape
        if self.positive:
            positive(self.height)
        if self.normalize:
            return self.height / (self.height.pow(2).sum(3).sqrt().expand_as(self.height) + self.eps)
        else:
            return self.height

    @property
    def spatial(self):
        c, w, h = self.in_shape
        n, comp = self.outdims, self.components
        weight = self.normalized_width.expand(n, 1, w, h, comp) \
                 * self.normalized_height.expand(n, 1, w, h, comp)
        weight = weight.sum(4).view(n, 1, w, h)
        return weight

    @property
    def weight(self):
        c, w, h = self.in_shape
        n, comp = self.outdims, self.components
        weight = self.spatial.expand(n, c, w, h) * self.features.expand(n, c, w, h)
        weight = weight.view(self.outdims, -1)
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
               ' (' + '{} x {} x {}'.format(*self.in_shape) + ' -> ' + str(self.outdims) + ') spatial rank {}'.format(
            self.components)


class BiasBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, features, **kwargs):
        kwargs['affine'] = False
        super().__init__(features, **kwargs)
        self.bias = nn.Parameter(torch.Tensor(features))
        self.initialize()

    def initialize(self):
        self.bias.data.fill_(0.)


class ExtendedConv2d(nn.Conv2d):
    """
    Extended 2D convolution module with fancier padding options.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, in_shape=None, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        if padding == 'SAME':
            assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1, "kernel must be odd sized"
            if stride[0] == 1 and stride[1] == 1:
                padding = (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2
            else:
                assert in_shape is not None, 'Input shape must be provided for stride that is not 1'
                h = in_shape[-2]
                w = in_shape[-1]

                padding = ceil((h * (stride[0] - 1) + kernel_size[0] - 1) / 2), \
                          ceil((w * (stride[1] - 1) + kernel_size[1] - 1) / 2)

        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, groups=groups, bias=bias)


class ConstrainedConv2d(ExtendedConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, in_shape=None, groups=1, bias=True, constrain=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, in_shape=in_shape, groups=groups, bias=bias)
        self.constrain_fn = constrain
        self.constrain_cache = None

    def constrain(self):
        if self.constrain_fn is not None:
            self.constrain_cache = self.constrain_fn(self.weight, cache=self.constrain_cache)

    def forward(self, *args, **kwargs):
        self.constrain()
        return super().forward(*args, **kwargs)


class ConstrainedConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, constrain=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.constrain_fn = constrain
        self.constrain_cache = None

    def constrain(self):
        if self.constrain_fn is not None:
            self.constrain_cache = self.constrain_fn(self.weight, cache=self.constrain_cache)

    def forward(self, *args, **kwargs):
        self.constrain()
        return super().forward(*args, **kwargs)


def conv2d_config(in_shape, out_shape, kernel_size, stride=None):
    """
    Given desired input and output tensor shapes and convolution kernel size,
    returns configurations that can be used to construct an appropriate 2D
    convolution operation satisfying the desired properties.

    Args:
        in_shape: shape of the input tensor. May be either [batch, channel, height, width]
                  or [channel, height, width]
        out_shape: shape of the output tensor. May be either [batch, channel, height, width]
                   or [channel, height, width]
        kernel_size: shape of the kernel. May be an integer or a pair tuple
        stride: (OPTIONAL) desired stride to be used. If not provided, optimal stride size
                will be computed and returned to minimize the necesssary amount of padding
                or stripping.

    Returns:
        A tuple (stride, padding, output_padding, padded_shape, conv_type, padding_type).
        stride: optimial stride size to be used. If stride was passed in, no change is made.
        padding: padding to be applied to each edge
        output_padding: if operation is transpose convolution, supplies output_padding that's
            necessary. Otherwise, this is None.
        conv_type: the required type of convolution. It is either "NORMAL" or "TRANSPOSE"
        padding_type: string to indicate the type of padding. Either "VALID" or "SAME".

    """
    in_shape = np.array(in_shape[-3:])
    out_shape = np.array(out_shape[-3:])
    kern_shape = np.array(kernel_size)

    # determine the kind of convolution to use
    if np.all(in_shape[-2:] >= out_shape[-2:]):
        conv_type = "NORMAL"
    elif np.all(in_shape[-2:] <= out_shape[-2:]):
        conv_type = "TRANSPOSE"
        in_shape, out_shape = out_shape, in_shape
    else:
        raise ValueError('Input shape dimensions must be both >= OR <= the output shape dimensions')

    if stride is None:
        stride = np.ceil((in_shape[-2:] - kern_shape + 1) / (out_shape[-2:] - 1)).astype(np.int)
    else:
        stride = np.array(_pair(stride))
    stride[stride <= 0] = 1
    padding = (out_shape[-2:] - 1) * stride + kern_shape - in_shape[-2:]

    if np.all(np.ceil(in_shape[-2:] / stride) == out_shape[-2:]):
        padding_type = 'SAME'
    else:
        padding_type = 'VALID'

    # get padded input shape
    in_shape[-2:] = in_shape[-2:] + padding.astype(np.int)
    padded_shape = tuple(in_shape.tolist())
    if conv_type == "TRANSPOSE":
        output_padding = tuple((padding % 2 != 0).astype(np.int).tolist())
    else:
        output_padding = None

    padding = tuple(np.ceil(padding / 2).astype(np.int).tolist())
    stride = tuple(stride.tolist())

    return stride, padding, output_padding, \
           padded_shape, conv_type, padding_type


def get_conv(in_shape, out_shape, kernel_size, stride=None, constrain=None, **kwargs):
    """
    Given desired input and output tensor shapes and convolution kernel size,
    returns a convolution operation satisfying the desired properties.

    Args:
        in_shape: shape of the input tensor. May be either [batch, channel, height, width]
                  or [channel, height, width]
        out_shape: shape of the output tensor. May be either [batch, channel, height, width]
                   or [channel, height, width]
        kernel_size: shape of the kernel. May be an integer or a pair tuple
        stride: (OPTIONAL) desired stride to be used. If not provided, optimal stride size
                will be computed and returned to minimize the necesssary amount of padding
                or stripping.
        constrain: (OPTIONAL) constrain function to be applied to the convolution filter weights
        **kwargs: additional arguments that are passed into the underlying convolution operation

    Returns:
        A convolution module (either a nn.Conv2d subclass or nn.ConvTranspose2d subclass)

    """
    in_channels, out_channels = in_shape[-3], out_shape[-3]
    stride, padding, output_padding, padded_shape, conv_type, padding_type = conv2d_config(in_shape, out_shape,
                                                                                           kernel_size, stride)

    if conv_type == "NORMAL":
        return ConstrainedConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 constrain=constrain, **kwargs)
    else:
        return ConstrainedConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                          constrain=constrain, output_padding=output_padding, **kwargs)
