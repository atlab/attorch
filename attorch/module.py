from collections import OrderedDict

from torch import nn


# class Module(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.__dict__['_hyper'] = OrderedDict()
#
#     def __getattr__(self, item):
#         if '_hyper' in self.__dict__:
#             _hyper = self.__dict__['_hyper']
#             if item in _hyper:
#                 return _hyper[item]
#         return nn.Module.__getattr__(self, item)
#
#     def __setattr__(self, item, value):
#         if '_hyper' in self.__dict__:
#             _hyper = self.__dict__['_hyper']
#             if item in _hyper:
#                 _hyper[item] = value
#         return nn.Module.__setattr__(self, item, value)
#
#     def hyper_parameters(self):
#         return self.__dict__['_hyper']

class ModuleDict(nn.Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for k, v in modules.items():
                if k not in self._modules:
                    self.add_module(str(k), v)

    def __getitem__(self, idx):
        return self._modules[str(idx)]

    def __setitem__(self, idx, module):
        return setattr(self, str(idx), module)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())


