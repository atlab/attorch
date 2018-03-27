import torch
import scipy.signal
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constraints import positive
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.autograd import Variable
import numpy as np
from math import ceil
# from .module import Module
from torch.nn import Parameter
from torch.nn.init import xavier_normal


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


# TODO if that's not needed, we should replace it by a padding and a convlayer in the future
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

    def l1(self, average=True):
        if average:
            return self.weight.abs().mean()
        else:
            return self.weight.abs().sum()

    def l2(self, average=True):
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
            return self.width / (self.width.pow(2).sum(2, keepdim=True) + self.eps).sqrt().expand_as(self.width)
        else:
            return self.width

    @property
    def normalized_height(self):
        c, w, h = self.in_shape
        if self.positive:
            positive(self.height)
        if self.normalize:
            return self.height / (self.height.pow(2).sum(3, keepdim=True) + self.eps).sqrt().expand_as(self.height)
        else:
            return self.height

    @property
    def spatial(self):
        c, w, h = self.in_shape
        n, comp = self.outdims, self.components
        weight = self.normalized_width.expand(n, 1, w, h, comp) \
                 * self.normalized_height.expand(n, 1, w, h, comp)
        weight = weight.sum(4, keepdim=True).view(n, 1, w, h)
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


class SpatialXFeatureLinear(nn.Module):
    """
    Factorized fully connected layer. Weights are a sum of outer products between a spatial filter and a feature vector.
    """

    def __init__(self, in_shape, outdims, bias=True, normalize=True, positive=True, spatial=None):
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
        positive(self.spatial)
        if self.normalize:
            weight = self.spatial / (
                    self.spatial.pow(2).sum(2, keepdim=True).sum(3, keepdim=True).sqrt().expand_as(self.spatial) + 1e-6)
        else:
            weight = self.spatial
        return weight

    @property
    def weight(self):
        if self.positive:
            positive(self.features)
        n = self.outdims
        c, w, h = self.in_shape
        weight = self.normalized_spatial.expand(n, c, w, h) * self.features.expand(n, c, w, h)
        weight = weight.view(self.outdims, -1)
        return weight

    def l1(self, average=True):
        n = self.outdims
        c, w, h = self.in_shape
        ret = (self.normalized_spatial.view(self.outdims, -1).abs().sum(1, keepdim=True)
               * self.features.view(self.outdims, -1).abs().sum(1)).sum()
        if average:
            ret = ret / (n * c * w * h)
        return ret

    def initialize(self, init_noise=1e-3):
        self.spatial.data.normal_(0, init_noise)
        self.features.data.normal_(0, init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, shift=None):
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


class SpatialTransformerPyramid2d(nn.Module):
    def __init__(self, in_shape, outdims, scale_n=4, positive=False, bias=True,
                 init_range=.1, downsample=True, _skip_upsampling=False, type=None):
        super().__init__()
        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims
        self.positive = positive
        self.gauss_pyramid = Pyramid(scale_n=scale_n, downsample=downsample, _skip_upsampling=_skip_upsampling, type=type)
        self.grid = Parameter(torch.Tensor(1, outdims, 1, 2))
        self.features = Parameter(torch.Tensor(1, c * (scale_n + 1), 1, outdims))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.init_range = init_range
        self.initialize()

    def initialize(self):
        self.grid.data.uniform_(-self.init_range, self.init_range)
        self.features.data.fill_(1 / self.in_shape[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def group_sparsity(self, group_size):
        f = self.features.size(1)
        n = f // group_size
        ret = 0
        for chunk in range(0, f, group_size):
            ret = ret + (self.features[:, chunk:chunk + group_size, ...].pow(2).mean(1) + 1e-12).sqrt().mean() / n
        return ret

    def feature_l1(self, average=True):
        if average:
            return self.features.abs().mean()
        else:
            return self.features.abs().sum()

    def neuron_layer_power(self, x, neuron_id):
        if self.positive:
            positive(self.features)
        self.grid.data = torch.clamp(self.grid.data, -1, 1)
        N, c, w, h = x.size()
        m = self.gauss_pyramid.scale_n + 1
        feat = self.features.view(1, m * c, self.outdims)

        y = torch.cat(self.gauss_pyramid(x), dim=1)
        y = (y * feat[:, :, neuron_id, None, None]).sum(1)
        return y.pow(2).mean()

    def forward(self, x, shift=None):
        if self.positive:
            positive(self.features)
        self.grid.data = torch.clamp(self.grid.data, -1, 1)
        N, c, w, h = x.size()
        m = self.gauss_pyramid.scale_n + 1
        feat = self.features.view(1, m * c, self.outdims)

        if shift is None:
            grid = self.grid.expand(N, self.outdims, 1, 2)
        else:
            grid = self.grid.expand(N, self.outdims, 1, 2) + shift[:, None, None, :]

        pools = [F.grid_sample(xx, grid) for xx in self.gauss_pyramid(x)]
        y = torch.cat(pools, dim=1).squeeze(-1)
        y = (y * feat).sum(1).view(N, self.outdims)

        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = self.__class__.__name__ + \
            ' (' + '{} x {} x {}'.format(c, w, h) + ' -> ' + str(self.outdims) + ')'
        if self.bias is not None:
            r += ' with bias'

        for ch in self.children():
            r += '  -> ' + ch.__repr__() + '\n'
        return r

class FactorizedSpatialTransformerPyramid2d(SpatialTransformerPyramid2d):
    def __init__(self, in_shape, outdims, scale_n=4, positive=False, bias=True,
                 init_range=.1, downsample=True, type=None):
        super(SpatialTransformerPyramid2d, self).__init__()
        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims
        self.positive = positive
        self.gauss_pyramid = Pyramid(scale_n=scale_n, downsample=downsample, type=type)
        self.grid = Parameter(torch.Tensor(1, outdims, 1, 2))
        self.feature_scales = Parameter(torch.Tensor(1, scale_n + 1, 1, outdims))
        self.feature_channels = Parameter(torch.Tensor(1, 1, c, outdims))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.init_range = init_range
        self.initialize()

    @property
    def features(self):
        return (self.feature_scales * self.feature_channels).view(1, -1, 1, self.outdims)

    def scale_l1(self, average=True):
        if average:
            return self.feature_scales.abs().mean()
        else:
            return self.feature_scales.abs().sum()

    def channel_l1(self, average=True):
        if average:
            return self.feature_channels.abs().mean()
        else:
            return self.feature_channels.abs().sum()

    def initialize(self):
        self.grid.data.uniform_(-self.init_range, self.init_range)
        self.feature_scales.data.fill_(1 / np.sqrt(self.in_shape[0]))
        self.feature_channels.data.fill_(1 / np.sqrt(self.in_shape[0]))

        if self.bias is not None:
            self.bias.data.fill_(0)



class SpatialTransformerPooled2d(nn.Module):
    def __init__(self, in_shape, outdims, pool_steps=1, positive=False, bias=True,
                 pool_kern=2, init_range=.1):
        super().__init__()
        self.pool_steps = pool_steps
        self.in_shape = in_shape
        c, w, h = in_shape
        self.outdims = outdims
        self.positive = positive
        self.grid = Parameter(torch.Tensor(1, outdims, 1, 2))
        self.features = Parameter(torch.Tensor(1, c * (self.pool_steps + 1), 1, outdims))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)

        self.pool_kern = pool_kern
        self.avg = nn.AvgPool2d((pool_kern, pool_kern), stride=pool_kern, count_include_pad=False)
        self.init_range = init_range
        self.initialize()

    def initialize(self):
        self.grid.data.uniform_(-self.init_range, self.init_range)
        self.features.data.fill_(1 / self.in_shape[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def feature_l1(self, average=True):
        if average:
            return self.features.abs().mean()
        else:
            return self.features.abs().sum()

    def group_sparsity(self, group_size):
        f = self.features.size(1)
        n = f // group_size
        ret = 0
        for chunk in range(0, f, group_size):
            ret = ret + (self.features[:, chunk:chunk + group_size, ...].pow(2).mean(1) + 1e-12).sqrt().mean() / n
        return ret

    def forward(self, x, shift=None):
        if self.positive:
            positive(self.features)
        self.grid.data = torch.clamp(self.grid.data, -1, 1)
        N, c, w, h = x.size()
        m = self.pool_steps + 1
        feat = self.features.view(1, m * c, self.outdims)

        if shift is None:
            grid = self.grid.expand(N, self.outdims, 1, 2)
        else:
            grid = self.grid.expand(N, self.outdims, 1, 2) + shift[:, None, None, :]

        pools = [F.grid_sample(x, grid)]
        for _ in range(self.pool_steps):
            x = self.avg(x)
            pools.append(F.grid_sample(x, grid))
        y = torch.cat(pools, dim=1)
        y = (y.squeeze(-1) * feat).sum(1).view(N, self.outdims)

        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = self.__class__.__name__ + \
            ' (' + '{} x {} x {}'.format(c, w, h) + ' -> ' + str(self.outdims) + ')'
        if self.bias is not None:
            r += ' with bias'
        r += ' and pooling for {} steps\n'.format(self.pool_steps)
        for ch in self.children():
            r += '  -> ' + ch.__repr__() + '\n'
        return r


class SpatialXFeatureLinear3d(nn.Module):
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
        ret = (self.spatial.view(self.outdims, -1).abs().sum(1, keepdim=True)
               * self.features.view(self.outdims, -1).abs().sum(1, keepdim=True)).sum()
        if average:
            ret = ret / (n * c * w * h)
        return ret

    @property
    def normalized_spatial(self):
        if self.positive:
            positive(self.spatial)
        if self.normalize:
            weight = self.spatial / (
                    self.spatial.pow(2).sum(2, keepdim=True).sum(3, keepdim=True).sum(4, keepdim=True).sqrt().expand(
                        self.spatial) + 1e-6)
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


class SpatialTransformerPyramid3d(nn.Module):
    def __init__(self, in_shape, outdims, scale_n=4, positive=True, bias=True, init_range=.05, downsample=True,
                 _skip_upsampling=False, type=None):
        super().__init__()
        self.in_shape = in_shape
        c, _, w, h = in_shape
        self.outdims = outdims
        self.positive = positive
        self.gauss = Pyramid(scale_n=scale_n, downsample=downsample, _skip_upsampling=_skip_upsampling, type=type)

        self.grid = Parameter(torch.Tensor(1, outdims, 1, 2))
        self.features = Parameter(torch.Tensor(1, c * (scale_n + 1), 1, outdims))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)
        self.init_range = init_range
        self.initialize()

    def initialize(self):
        self.grid.data.uniform_(-self.init_range, self.init_range)
        self.features.data.fill_(1 / self.in_shape[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def feature_l1(self, average=True, subs_idx=None):
        if subs_idx is not None: raise NotImplemented('Subsample is not implemented.')

        if average:
            return self.features.abs().mean()
        else:
            return self.features.abs().sum()

    def forward(self, x, shift=None, subs_idx=None):
        if subs_idx is not None: raise NotImplemented('Subsample is not implemented.')

        if self.positive:
            positive(self.features)
        self.grid.data = torch.clamp(self.grid.data, -1, 1)
        N, c, t, w, h = x.size()
        m = self.gauss.scale_n + 1
        feat = self.features.view(1, m * c, self.outdims)

        if shift is None:
            grid = self.grid.expand(N * t, self.outdims, 1, 2)
        else:
            grid = self.grid.expand(N, self.outdims, 1, 2)
            grid = torch.stack([grid + shift[:, i, :][:, None, None, :] for i in range(t)], 1)
            grid = grid.contiguous().view(-1, self.outdims, 1, 2)

        z = x.contiguous().transpose(2, 1).contiguous().view(-1, c, w, h)
        pools = [F.grid_sample(x, grid) for x in self.gauss(z)]
        y = torch.cat(pools, dim=1).squeeze(-1)
        y = (y * feat).sum(1).view(N, t, self.outdims)

        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        c, t, w, h = self.in_shape
        r = self.__class__.__name__ + \
            ' (' + '{} x {} x {}'.format(c, w, h) + ' -> ' + str(self.outdims) + ')'
        if self.bias is not None:
            r += ' with bias'

        for ch in self.children():
            r += '\n  -> ' + ch.__repr__()
        return r


class SpatialTransformerPooled3d(nn.Module):
    """
    Factorized readout layer from convolution activations. For each feature layer, the readout weights are
    Gaussian over spatial dimensions.
    """

    def __init__(self, in_shape, outdims, pool_steps=1, positive=False, bias=True, init_range=.05):
        super().__init__()
        self.pool_steps = pool_steps
        self.in_shape = in_shape
        c, t, w, h = in_shape
        self.outdims = outdims
        self.positive = positive
        self.grid = Parameter(torch.Tensor(1, outdims, 1, 2))
        self.features = Parameter(torch.Tensor(1, c * (self.pool_steps + 1), 1, outdims))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter('bias', bias)
        else:
            self.register_parameter('bias', None)

        self.avg = nn.AvgPool2d((2, 2), stride=(2, 2), count_include_pad=False)
        self.init_range = init_range
        self.initialize()

    def initialize(self, init_noise=1e-3):
        # randomly pick centers within the spatial map
        self.grid.data.uniform_(-self.init_range, self.init_range)
        self.features.data.fill_(1 / self.in_shape[0])

        if self.bias is not None:
            self.bias.data.fill_(0)

    def feature_l1(self, average=True, subs_idx=None):
        subs_idx = subs_idx if subs_idx is not None else slice(None)
        if average:
            return self.features[..., subs_idx].abs().mean()
        else:
            return self.features[..., subs_idx].abs().sum()

    def forward(self, x, shift=None, subs_idx=None):
        if self.positive:
            positive(self.features)
        self.grid.data = torch.clamp(self.grid.data, -1, 1)

        N, c, t, w, h = x.size()
        m = self.pool_steps + 1
        if subs_idx is not None:
            feat = self.features[..., subs_idx].contiguous()
            outdims = feat.size(-1)
            feat = feat.view(1, m * c, outdims)
            grid = self.grid[:, subs_idx, ...]
        else:
            grid = self.grid
            feat = self.features.view(1, m * c, self.outdims)
            outdims = self.outdims

        if shift is None:
            grid = grid.expand(N * t, outdims, 1, 2)
        else:
            grid = grid.expand(N, outdims, 1, 2)
            grid = torch.stack([grid + shift[:, i, :][:, None, None, :] for i in range(t)], 1)
            grid = grid.contiguous().view(-1, outdims, 1, 2)
        z = x.contiguous().transpose(2, 1).contiguous().view(-1, c, w, h)
        pools = [F.grid_sample(z, grid)]
        for i in range(self.pool_steps):
            z = self.avg(z)
            pools.append(F.grid_sample(z, grid))
        y = torch.cat(pools, dim=1)
        y = (y.squeeze(-1) * feat).sum(1).view(N, t, outdims)

        if self.bias is not None:
            if subs_idx is None:
                y = y + self.bias
            else:
                y = y + self.bias[subs_idx]

        return y

    def __repr__(self):
        c, _, w, h = self.in_shape
        r = self.__class__.__name__ + \
            ' (' + '{} x {} x {}'.format(c, w, h) + ' -> ' + str(self.outdims) + ')'
        if self.bias is not None:
            r += ' with bias\n'
        for ch in self.children():
            r += '  -> ' + ch.__repr__() + '\n'
        return r

class BiasBatchNorm2d(nn.Module):
    def __init__(self, features, **kwargs):
        kwargs['affine'] = False
        super().__init__()
        self.bn = nn.BatchNorm2d(features, **kwargs)
        self.bias = nn.Parameter(torch.Tensor(1, features, 1, 1))
        self.initialize()

    def initialize(self):
        self.bn.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x):
        return self.bn(x) + self.bias

#
# class BiasBatchNorm2d(nn.BatchNorm2d):
#     def __init__(self, features, **kwargs):
#         kwargs['affine'] = False
#         super().__init__(features, **kwargs)
#         self.offset = nn.Parameter(torch.Tensor(1, features, 1, 1))
#         self.initialize()
#
#     def initialize(self):
#         self.reset_parameters()
#         self.offset.data.zero_()
#
#     def forward(self, x):
#         x = super().forward(x)
#         return x + self.offset


class BiasBatchNorm3d(nn.BatchNorm3d):
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
                will be computed and returned to minimize the necessary amount of padding
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


class Pyramid(nn.Module):
    _filter_dict = {
        'gauss5x5': np.float32([
            [0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
            [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
            [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
            [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
            [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]]),
        'gauss3x3': np.float32([
            [1 / 16, 1 / 8, 1 / 16],
            [1 / 8, 1 / 4, 1 / 8],
            [1 / 16, 1 / 8, 1 / 16]]
        ),
        'laplace5x5': np.outer(np.float32([1, 4, 6, 4, 1]), np.float32([1, 4, 6, 4, 1])) / 256,

    }

    def __init__(self, scale_n=4, type='gauss5x5', downsample=True, _skip_upsampling=False):
        """
        Setup Laplace image pyramid
        Args:
            scale_n: number of Laplace pyramid layers to construct
            type: type of Gaussian filter used in pyramid construction. Valid options are: 'gauss5x5', 'gauss3x3', and 'laplace5x5'
            downsample: whether to downsample the image in each layer. Defaults to True
            _skip_upsampling: Present for legacy reasons. Set to False (default) to get correct behavior.
        """
        super().__init__()
        self.type = type
        self.downsample = downsample
        self._skip_upsampling = _skip_upsampling
        h = self._filter_dict[type]
        self.register_buffer('filter', torch.from_numpy(h))
        self.scale_n = scale_n
        self._kern = h.shape[0]
        self._pad = self._kern // 2
        self._filter_cache = None

    def lap_split(self, img):
        N, c, h, w = img.size()
        if self._filter_cache is not None and self._filter_cache.size(0) == c:
            filter = self._filter_cache
        else:
            filter = Variable(self.filter.expand(c, 1, self._kern, self._kern)).contiguous()
            self._filter_cache = filter

        # the necessary output padding depends on even/odd of the dimension
        output_padding = (h + 1) % 2, (w + 1) % 2

        smooth = F.conv2d(img, filter, padding=self._pad, groups=c)
        if self.downsample:
            lo = smooth[:, :, ::2, ::2]
            if self._skip_upsampling:
                lo2 = smooth
            else:
                lo2 = 4 * F.conv_transpose2d(lo, filter, stride=2, padding=self._pad, output_padding=output_padding, groups=c)
        else:
            lo = lo2 = smooth

        hi = img - lo2

        return lo, hi


    def forward(self, img):
        levels = []
        for i in range(self.scale_n):
            img, hi = self.lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels

    def __repr__(self):
        return "Pyramid(scale_n={scale_n}, padding={_pad}, downsample={downsample}, _skip_upsampling={_skip_upsampling}, type={type})".format(
            **self.__dict__)
