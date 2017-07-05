import torch
from torch.utils.data import Dataset
import numpy as np
from torch.autograd import Variable


class MultiTensorDataset(Dataset):
    """Dataset wrapping data and target tensors.
    Each sample will be retrieved by indexing both tensors along the first
    dimension and converted into a the according dtype.
    
    Arguments:
        *data (numpy arrays): datasets
        data_dtype               : dtype of the tensors (for cuda conversion)
    """

    def __init__(self, *data, data_dtype=None):
        self.data_dtype = data_dtype
        for d in data:
            assert d.shape[0] == data[0].shape[0], 'datasets must have same first dimension'
        self.data = tuple(torch.from_numpy(d.astype(np.float32)) for d in data)

    def __getitem__(self, index):
        return tuple(d[index] for d in self.data)

    def mean(self, axis=None):
        if axis is None:
            return tuple(d.mean() for d in self.data)
        else:
            return tuple(d.mean(axis) for d in self.data)

    def __len__(self):
        return self.data[0].size(0)

    def __repr__(self):
        return '\n'.join(['Tensor {}: {}'.format(i, str(t.size())) for i, t in enumerate(self.data)])


class NumpyDataset:
    """
    Arguments:
        *data (numpy arrays): datasets
    """

    def __init__(self, *data):
        for d in data:
            assert d.shape[0] == data[0].shape[0], 'datasets must have same first dimension'
        self.data = data

    def __getitem__(self, index):
        return tuple(d[index] for d in self.data)

    def mean(self, axis=None):
        if axis is None:
            return tuple(d.mean() for d in self.data)
        else:
            return tuple(d.mean(axis) for d in self.data)

    def __len__(self):
        return self.data[0].shape(0)

    def __repr__(self):
        return '\n'.join(['Array {}: {}'.format(i, str(t.size())) for i, t in enumerate(self.data)])

def to_variable(iter, cuda=True, **kwargs):
    """
    Converts output of iter into Variables.
    
    Args:
        iter:       iterator that returns tuples of tensors 
        **kwargs:   keyword arguments for the Variable constructor
    """
    for elem in iter:
        if cuda:
            yield tuple(Variable(e.cuda(), **kwargs) for e in elem)
        else:
            yield tuple(Variable(e, **kwargs) for e in elem)


