import torch
from torch.utils.data import Dataset
import numpy as np
from torch.autograd import Variable


# class Dataset:
#     def __init__(self, inputs, outputs):
#         self.inputs = inputs
#         self.outputs = outputs
#
#     def __repr__(self):
#         s = ['Inputs:']
#         for k, v in self.inputs.items():
#             s.append('\t{}:\t{}'.format(k, ' x '.join(map(str, v.shape))))
#
#         s = ['Outputs:']
#         for k, v in self.outputs.items():
#             s.append('\t{}:\t{}'.format(k, ' x '.join(map(str, v.shape))))
#         return '\n'.join(s)


class NumpyDataset(Dataset):
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
        return tuple(d[index].type(self.data_dtype) for d in self.data)

    def mean(self, axis=None):
        if axis is None:
            return tuple(d.mean().type(self.data_dtype) for d in self.data)
        else:
            return tuple(d.mean(axis).type(self.data_dtype) for d in self.data)

    def __len__(self):
        return self.data[0].size(0)

def to_variable(iter, **kwargs):
    """
    Converts output of iter into Variables.
    
    Args:
        iter:       iterator that returns tuples of tensors 
        **kwargs:   keyword arguments for the Variable constructor
    """
    for elem in iter:
        yield tuple(Variable(e, **kwargs) for e in elem)
