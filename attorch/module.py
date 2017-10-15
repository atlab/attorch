from collections import OrderedDict

from torch import nn


class ModuleDict(nn.Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for k, v in modules.items():
                if k not in self._modules:
                    self.add_module(str(k), v)
                else:
                    raise KeyError('{} is already in modules. Please choose a different name'.format(str(k)))

    def __getitem__(self, idx):
        return self._modules[str(idx)]

    def __setitem__(self, idx, module):
        return setattr(self, str(idx), module)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def values(self):
        return self._modules.values()

