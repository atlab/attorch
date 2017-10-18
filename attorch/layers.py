import torch
from .constraints import positive
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.autograd import Variable
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


def elu1(x):
    return F.elu(x, inplace=True) + 1.


class Elu1(nn.Module):
    """
    Elu activation function shifted by 1 to ensure that the
    output stays positive. That is:

    Elu1(x) = Elu(x) + 1
    """

    def forward(self, x):
        return elu1(x)




def log1exp(x):
    return torch.log(1. + torch.exp(x))


class Log1Exp(nn.Module):
    def forward(self, x):
        return log1exp(x)


class AdjustedElu(nn.Module):
    """
    Elu activation function that's adjusted to:
    1) ensure that all outputs are positive and
    2) f(x) = x for x >= 1
    """

    def forward(self, x):
        return F.elu(x - 1.) + 1.


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


class SpatialXFeatureLinear3D(nn.Module):
    def __init__(self, in_shape, outdims, bias=True, normalize=False, positive=True, spatial=None):
        super().__init__()
        self.in_shape = in_shape
        self.outdims = outdims
        self.normalize = normalize
        self.positive = positive
        c, t, w, h = in_shape
        self.spatial = Parameter(torch.Tensor(self.outdims, 1, 1, w, h)) if spatial is None else spatial
        self.features = Parameter(torch.Tensor(self.outdims, c, 1, 1, 1))
        if bias:
            bias = Parameter(torch.Tensor(self.outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.initialize()

    def l1(self, average=True):
        n = self.outdims
        c, _, w, h = self.in_shape
        ret = (self.spatial.view(self.outdims, -1).abs().sum(1)
               * self.features.view(self.outdims, -1).abs().sum(1)).sum()
        if average:
            ret = ret / (n * c * w * h)
        return ret

    @property
    def normalized_spatial(self):
        if self.positive:
            positive(self.spatial)
        if self.normalize:
            weight = self.spatial / (self.spatial.pow(2).sum(2).sum(3).sum(4).sqrt().expand(self.spatial) + 1e-6)
        else:
            weight = self.spatial
        return weight

    @property
    def constrained_features(self):
        if self.positive:
            positive(self.features)
        return self.features

    @property
    def weight(self):
        n = self.outdims
        c, _, w, h = self.in_shape
        weight = self.normalized_spatial.expand(n, c, 1, w, h) * self.constrained_features.expand(n, c, 1, w, h)
        return weight

    def initialize(self, init_noise=1e-3):
        self.spatial.data.normal_(0, init_noise)
        self.features.data.normal_(0, init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x):
        N, c, t, w, h = x.size()
        # tmp2 = x.transpose(2, 1).contiguous()
        # tmp2 = tmp2.view(-1, w * h) @ self.normalized_spatial.view(self.outdims, -1).t()
        # tmp2 = (tmp2.view(N*t,c,self.outdims) \
        #         * self.constrained_features.transpose(0,1).contiguous().view(c, self.outdims).expand(N* t, c, self.outdims)).sum(1)

        tmp = x.transpose(2, 1).contiguous().view(-1, c * w * h) @ self.weight.view(self.outdims, -1).t()
        if self.bias is not None:
            tmp = tmp + self.bias.expand_as(tmp)
            # tmp2 = tmp2 + self.bias.expand_as(tmp2)
        return tmp.view(N, t, self.outdims)
        # return tmp2.view(N, t, self.outdims)

    def __repr__(self):
        c, t, w, h = self.in_shape
        return ('positive ' if self.positive else '') + \
               ('spatially normalized ' if self.normalize else '') + \
               self.__class__.__name__ + \
               ' (' + '{} x {} x {}'.format(c, w, h) + ' -> ' + str(self.outdims) + ')'


class GaussianSpatialXFeatureLinear(nn.Module):
    """
    Factorized readout layer from convolution activations. For each feature layer, the readout weights are
    Gaussian over spatial dimensions.
    """

    def __init__(self, in_shape, outdims, bias=True, sigma_scale=1.0, sigma_eps=1e-3):
        super().__init__()
        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims
        self.sigma_eps = sigma_eps
        self.sigma_scale = sigma_scale
        self.cx = Parameter(torch.Tensor(outdims, 1, 1, 1))
        self.cy = Parameter(torch.Tensor(outdims, 1, 1, 1))
        self.sigma = Parameter(torch.Tensor(outdims, 1, 1, 1))
        self.features = Parameter(torch.Tensor(outdims, c, 1, 1))

        w_edge = (w - 1) / 2.0
        h_edge = (h - 1) / 2.0

        grid_x = torch.linspace(-w_edge, w_edge, w).view(1, 1, -1, 1)
        grid_y = torch.linspace(-h_edge, h_edge, h).view(1, 1, 1, -1)
        # grids are non-parameters but needs to be maintained as persistent state
        self.register_buffer('grid_x', grid_x)
        self.register_buffer('grid_y', grid_y)

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)

        self.initialize()

    def constrain_sigma(self):
        pos = self.sigma.data.ge(0).float()
        self.sigma.data *= pos
        self.sigma.data += (1 - pos) * self.sigma_eps

    @property
    def spatial(self):
        self.constrain_sigma()

        n = self.outdims
        c, w, h = self.in_shape
        grid_x = Variable(self.grid_x)
        grid_y = Variable(self.grid_y)
        d = (self.cx.expand(n, 1, w, h) - grid_x.expand(n, 1, w, h)).pow(2) + \
            (self.cy.expand(n, 1, w, h) - grid_y.expand(n, 1, w, h)).pow(2)
        return torch.exp(-d / self.sigma.expand(n, 1, w, h).pow(2))

    @property
    def raw_weight(self):
        n = self.outdims
        c, w, h = self.in_shape
        return self.spatial.expand(n, c, w, h) * self.features.expand(n, c, w, h)

    @property
    def weight(self):
        return self.raw_weight.view(self.outdims, -1)

    def initialize(self, init_noise=1e-3):
        c, w, h = self.in_shape

        x = np.linspace(0, w, 100)
        y = np.linspace(0, h, 100)
        xv, yv = np.meshgrid(x, y)
        xf = xv.flatten()
        yf = yv.flatten()

        # numerically approximate the median distance between two randomly chosen points in a rectangle
        sigma = self.sigma_scale * np.median(np.sqrt((xf - xf[:, np.newaxis]) ** 2 + (yf - yf[:, np.newaxis]) ** 2))

        # randomly pick centers within the spatial map
        self.cx.data.uniform_(-w / 2.0, w / 2.0)
        self.cy.data.uniform_(-h / 2.0, h / 2.0)

        self.sigma.data.fill_(sigma)

        self.features.data.normal_(0, init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def sigma_l1(self):
        return (self.sigma - self.sigma_eps).abs().mean()
        
    def sigma_l2(self):
        return (self.sigma - self.sigma_eps).pow(2).mean()

    def feature_l1(self, average=True):
        if average:
            return self.features.abs().mean()
        else:
            return self.features.abs().sum()
             
    def weight_l1(self, average=True):
        if average:
            return self.weight.abs().mean()
        else:
            return self.weight.abs().sum()

    def forward(self, x):
        N = x.size(0)
        y = x.view(N, -1) @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias.expand_as(y)
        return y

    def __repr__(self):
        r = self.__class__.__name__ + \
            ' (' + '{} x {} x {}'.format(*self.in_shape) + ' -> ' + str(self.outdims) + ')'
        r += ' sigma scale={}'.format(self.sigma_scale)
        if self.bias is not None:
            r += ' with bias'


        return r


class GaussianSpatialXFeatureLinear3d(GaussianSpatialXFeatureLinear):
    """
    Factorized readout layer from convolution activations. For each feature layer, the readout weights are
    Gaussian over spatial dimensions.
    """

    def __init__(self, in_shape, outdims, bias=True, sigma_scale=1.0):
        super().__init__(in_shape[:1] + in_shape[2:], outdims, bias=bias, sigma_scale=sigma_scale)

    def forward(self, x):
        N, c, t, w, h = x.size()
        tmp = x.transpose(2, 1).contiguous().view(-1, c * w * h) @ self.raw_weight.view(self.outdims, -1).t()
        if self.bias is not None:
            tmp = tmp + self.bias.expand_as(tmp)
        return tmp.view(N, t, self.outdims)


class FullLinear(nn.Module):
    """
    Fully connected linear readout from image-like input with c x w x h into a vector output
    """

    def __init__(self, in_shape, outdims, bias=True):
        super().__init__()
        self.in_shape = in_shape
        self.outdims = outdims

        c, w, h = in_shape

        self.raw_weight = Parameter(torch.Tensor(self.outdims, c, w, h))

        if bias:
            self.bias = Parameter(torch.Tensor(self.outdims))
        else:
            self.register_parameter('bias', None)

        self.initialize()

    def initialize(self, init_noise=1e-3):
        self.raw_weight.data.normal_(0, init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

    @property
    def weight(self):
        return self.raw_weight.view(self.outdims, -1)

    def weight_l1(self, average=True):
        if average:
            return self.weight.abs().mean()
        else:
            return self.weight.abs().sum()

    def weight_l2(self, average=True):
        if average:
            return self.weight.pow(2).mean()
        else:
            return self.weight.pow(2).sum()

    def forward(self, x):
        N = x.size(0)
        y = x.view(N, -1) @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias.expand_as(y)
        return y

    def __repr__(self):
        r = self.__class__.__name__ + \
            ' (' + '{} x {} x {}'.format(*self.in_shape) + ' -> ' + str(self.outdims) + ')'
        if self.bias is not None:
            r += ' with bias'
        return r



class SpatialXFeatureLinear(nn.Module):
    """
    Factorized fully connected layer. Weights are a sum of outer products between a spatial filter and a feature vector.  
    """

    def __init__(self, in_shape, outdims, bias=True, normalize=True, positive=False, spatial=None):
        super().__init__()
        self.in_shape = in_shape
        self.outdims = outdims
        self.normalize = normalize
        self.positive = positive
        c, w, h = in_shape
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
        n = self.outdims
        c, w, h = self.in_shape
        weight = self.normalized_spatial.expand(n, c, w, h) * self.features.expand(n, c, w, h)
        weight = weight.view(self.outdims, -1)
        return weight

    def l1(self, average=True):
        n = self.outdims
        c, w, h = self.in_shape
        ret = (self.normalized_spatial.view(self.outdims, -1).abs().sum(1)
               * self.features.view(self.outdims, -1).abs().sum(1)).sum()
        if average:
            ret = ret / (n * c * w * h)
        return ret

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
            return self.width / (self.width.pow(2).sum(2) + self.eps).sqrt().expand_as(self.width)
        else:
            return self.width

    @property
    def normalized_height(self):
        c, w, h = self.in_shape
        if self.positive:
            positive(self.height)
        if self.normalize:
            return self.height / (self.height.pow(2).sum(3) + self.eps).sqrt().expand_as(self.height)
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


class BiasBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, features, **kwargs):
        kwargs['affine'] = False
        super().__init__(features, **kwargs)
        self.bias = nn.Parameter(torch.Tensor(features))
        self.initialize()

    def initialize(self):
        self.bias.data.fill_(0.)


class DivNorm3d(nn.Module):
    def __init__(self, num_features, sigma=0.1, bias=False, scale=False):
        super().__init__()
        self.sigma = sigma
        self.num_features = num_features

        if bias:
            b = Parameter(torch.zero(num_features))
            self.register_parameter('bias', b)
        else:
            self.register_parameter('bias', None)

        if scale:
            s = Parameter(torch.ones(num_features))
            self.register_parameter('scale', s)
        else:
            self.register_parameter('scale', None)

    def __repr__(self):
        return ('{name}({num_features}, sigma={sigma})'.format(name=self.__class__.__name__, **self.__dict__))

    def forward(self, x):
        mu = x.mean(1).mean(2).mean(3).mean(4)
        y = x - mu.expand_as(x)
        y = y / torch.sqrt(self.sigma + y.pow(2).mean().expand_as(y))

        if self.bias is not None:
            y = y + self.bias.expand_as(y)
        if self.scale is not None:
            y = y * self.scale.expand_as(y)
        return y


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
