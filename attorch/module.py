from collections import OrderedDict

from torch import nn


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.__dict__['_hyper'] = OrderedDict()

    def __getattr__(self, item):
        if '_hyper' in self.__dict__:
            _hyper = self.__dict__['_hyper']
            if item in _hyper:
                return _hyper[item]
        return nn.Module.__getattr__(self, item)

    def __setattr__(self, item, value):
        if '_hyper' in self.__dict__:
            _hyper = self.__dict__['_hyper']
            if item in _hyper:
                _hyper[item] = value
        return nn.Module.__setattr__(self, item, value)

    def hyper_parameters(self):
        return self.__dict__['_hyper']
