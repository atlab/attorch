from collections import defaultdict, namedtuple, Mapping

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.autograd import Variable


class DataTransform:
    def initialize(self, dataset):
        pass

    def __repr__(self):
        return self.__class__.__name__


class SubsampleNeurons(DataTransform):
    def __init__(self, datakey, idx, axis):
        super().__init__()
        self.idx = idx
        self.datakey = datakey
        self._subsamp = None
        self.axis = axis

    def initialize(self, dataset):
        self._subsamp = []
        for d in dataset.data_keys:
            if d == self.datakey:
                self._subsamp.append([slice(None) for _ in range(self.axis - 1)] + [self.idx, ...])
            else:
                self._subsamp.append(...)

    def __call__(self, item):
        return tuple(it[sub] for sub, it in zip(self._subsamp, item))


class Neurons2Behavior(DataTransform):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def __call__(self, item):
        return tuple((item[0], np.hstack((item[1], item[3][~self.idx])), item[2], item[3][self.idx]))


class TransformFromFuncs(DataTransform):
    def __init__(self):
        super().__init__()

    def initialize(self, dataset):
        self._transforms = []
        self.transformees = []
        for d in dataset.data_keys:
            if hasattr(dataset, d + '_transform'):
                self._transforms.append(getattr(dataset, d + '_transform'))
                self.transformees.append(d)
            else:
                self._transforms.append(lambda x: x)

    def __call__(self, item):
        return tuple(tr(it) for tr, it in zip(self._transforms, item))

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ', '.join(self.transformees))


class ToTensor(DataTransform):
    def __call__(self, item):
        return tuple(torch.from_numpy(it) for it in item)


class Chain(DataTransform):
    def __init__(self, *transforms):
        self.transforms = transforms

    def initialize(self, dataset):
        for tr in self.transforms:
            tr.initialize(dataset)

    def __call__(self, item):
        for tr in self.transforms:
            item = tr(item)
        return item

    def __add__(self, other):
        return Chain(*self.transforms, other)

    def __iadd__(self, other):
        self.transforms = self.transforms + (other,)
        return self

    def __repr__(self):
        return "{}[{}]".format(self.__class__.__name__, ' -> '.join(map(repr, self.transforms)))


class H5Dataset(Dataset):
    def __init__(self, filename, *data_keys, info_name=None, transform=None):
        self.fid = h5py.File(filename, 'r')
        m = None
        for key in data_keys:
            assert key in self.fid, 'Could not find {} in file'.format(key)
            if m is None:
                m = len(self.fid[key])
            else:
                assert m == len(self.fid[key]), 'Length of datasets do not match'
        self._len = m
        self.data_keys = data_keys

        if info_name is not None:
            self.info = self.fid[info_name]

        if transform is None:
            self.transform = Chain(TransformFromFuncs(), ToTensor())
        else:
            self.transform = transform

        self.transform.initialize(self)

    def __getitem__(self, item):
        return self.transform(tuple(self.fid[d][item] for d in self.data_keys))

    def __iter__(self):
        yield from map(self.__getitem__, range(len(self)))

    def __len__(self):
        return self._len

    def __repr__(self):
        return '\n'.join(['Tensor {}: {} '.format(key, self.fid[key].shape)
                          for key in self.data_keys] + ['Transforms: ' + repr(self.transform)])


class H5SequenceSet(Dataset):
    def __init__(self, filename, *data_groups, transforms=None):
        self._fid = h5py.File(filename, 'r')

        m = None
        for key in data_groups:
            assert key in self._fid, 'Could not find {} in file'.format(key)
            l = len(self._fid[key])
            if m is not None and l != m:
                raise ValueError('groups have different length')
            m = l
        self._len = m

        self.data_groups = data_groups

        self.transforms = transforms or []

        self.data_point = namedtuple('DataPoint', data_groups)

    def transform(self, x, exclude=None):
        for tr in self.transforms:
            if exclude is None or not isinstance(tr, exclude):
                x = tr(x)
        return x

    def __getitem__(self, item):
        x = self.data_point(*(np.array(self._fid[g][str(item)]) for g in self.data_groups))
        for tr in self.transforms:
            x = tr(x)
        return x

    def __iter__(self):
        yield from map(self.__getitem__, range(len(self)))

    def __len__(self):
        return self._len

    def __repr__(self):
        return 'H5SequenceSet m={}:\n\t({})'.format(len(self), ', '.join(self.data_groups)) \
             + '\n\t[Transforms: ' + '->'.join([repr(tr) for tr in self.transforms]) +']'

    def __getattr__(self, item):
        if item in self._fid:
            item = self._fid[item]
            if isinstance(item, h5py._hl.dataset.Dataset):
                item = item.value
                if item.dtype.char == 'S':  # convert bytes to univcode
                    item = item.astype(str)
                return item
            return item
        else:
            raise AttributeError('Item {} not found in {}'.format(item, self.__class__.__name__))


class MultiTensorDataset(Dataset):
    """Dataset wrapping data and target tensors.
    Each sample will be retrieved by indexing both tensors along the first
    dimension and converted into a the according dtype.
    
    Arguments:
        *data (numpy arrays): datasets
        data_dtype               : dtype of the tensors (for cuda conversion)
    """

    def __init__(self, *data, transform=None):
        for d in data:
            assert d.shape[0] == data[0].shape[0], 'datasets must have same first dimension'
        self.data = tuple(torch.from_numpy(d.astype(np.float32)) for d in data)
        if transform is None:
            transform = lambda x: x
        self.transform = transform

    def __getitem__(self, index):
        ret = tuple(d[index] for d in self.data)
        return self.transform(ret)

    def mean(self, axis=None):
        if axis is None:
            return tuple(d.mean() for d in self.data)
        else:
            return tuple(d.mean(axis, keepdim=True) for d in self.data)

    def __len__(self):
        return self.data[0].size(0)

    def as_variables(self, **kwargs):
        return tuple(Variable(v, **kwargs) for v in self.data)

    def cuda_(self):
        self.data = tuple(v.cuda() for v in self.data)

    def cpu_(self):
        self.data = tuple(v.cpu() for v in self.data)

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
        return '\n'.join(['Array {}: {}'.format(i, str(t.shape)) for i, t in enumerate(self.data)])


def to_variable(iter, cuda=True, filter=None, **kwargs):
    """
    Converts output of iter into Variables.
    
    Args:
        iter:       iterator that returns tuples of tensors
        cuda:       whether the elements should be loaded onto the GPU
        filter:     tuple of bools as long as the number of returned elements by iter. If filter[i] is False,
                    the element is not converted.
        **kwargs:   keyword arguments for the Variable constructor
    """
    for elem in iter:
        if filter is None:
            filter = (True,) if not isinstance(elem, (tuple, list)) else len(elem) * (True,)
        if cuda:
            yield tuple(Variable(e.cuda(), **kwargs) if f else e for f, e in zip(filter, elem))
        else:
            yield tuple(Variable(e, **kwargs) if f else e for f, e in zip(filter, elem))


class ListDataset(Dataset):
    """
    Arguments:
        *data (indexable): datasets
    """

    def __init__(self, *data, transform=None):
        self.transform = transform
        for d in data:
            assert len(d) == len(data[0]), 'datasets must have same first dimension'
        self.data = data

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(tuple(d[index] for d in self.data))
        else:
            return tuple(d[index] for d in self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return '\n'.join(['List  {}: {}'.format(i, str(len(t))) for i, t in enumerate(self.data)])
