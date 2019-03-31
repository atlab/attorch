from collections import OrderedDict, namedtuple, Mapping
from itertools import cycle
import contextlib

import numpy as np
from graphviz import Digraph
from scipy import signal
from scipy.interpolate import InterpolatedUnivariateSpline
from torch.autograd import Variable
import torch
import time

import sys

class silence:
    """
    Context manager to temporarily suppress stdout prints

    Examples:
        with silent():
          code_that_would_print_stuff()
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def namedtuple_with_defaults(typename, field_names, default_values=()):
    """
    From https://stackoverflow.com/questions/11351032/namedtuple-and-optional-keyword-arguments
    """
    T = namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T

def make_dot(var):
    """from https://github.com/szagoruyko/functional-zoo/blob/master/visualize.py"""
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '(' + (', ').join(['%d' % v for v in var.size()]) + ')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):

                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])

    add_nodes(var.creator)
    return dot


def get_static_nonlinearity(y_hat, y):
    """
    Compute static nonlinearity by computing a linear spline from y_hat[:,i].sort to y[:,i].sort 
    independently for each column i in y_hat and y.
    
    Args:
        y_hat: example x dimension array 
        y:     example x dimension array

    Returns:
        A function that computes the nonlinearity d dimensional arrays where d == y_hat.shape[1] 
    """
    fs = [InterpolatedUnivariateSpline(np.sort(y1), np.sort(y2), k=1, ext=1)
          for y1, y2 in zip(y_hat.T, y.T)]

    def nl(y_test):
        return np.vstack([f(yt) for f, yt in zip(fs, y_test.T)]).T

    return nl


def downsample(images, downsample_by=4):
    """
    Downsamples images in in images by a factor of downsample. Filters with a hamming filter beforehand.

    Args:
        images: an iterable containing images
        downsample_by: downsampling factor

    Returns:
        stacked downsampled images

    """
    h = np.hamming(2 * downsample_by + 1)
    h /= h.sum()
    H = h[:, np.newaxis] * h[np.newaxis, :]

    down = lambda img: signal.convolve2d(img.astype(np.float32), H, mode='same',
                                         boundary='symm')[downsample_by // 2::downsample_by,
                       downsample_by // 2::downsample_by]
    return np.stack([down(img) for img in images], axis=0)

@contextlib.contextmanager
def timing(name):
    torch.cuda.synchronize()
    start_time = time.time()
    yield
    torch.cuda.synchronize()
    end_time = time.time()
    print('{} {:6.3f} seconds'.format(name, end_time-start_time))


def alternate(*args):
    """
    Given multiple iterators, returns a generator that alternatively visit one element from each iterator at a time.
    Examples:
        >>> list(alternate(['a', 'b', 'c'], [1, 2, 3], ['Mon', 'Tue', 'Wed']))
        ['a', 1, 'Mon', 'b', 2, 'Tue', 'c', 3, 'Wed']
    Args:
        *args: one or more iterables (e.g. tuples, list, iterators) separated by commas
    Returns:
        A generator that alternatively visits one element at a time from the list of iterables
    """
    for row in zip(*args):
        yield from row


def cycle_datasets(loaders):
    """
    Cycles through datasets of dataloaders.
    Args:
        loaders: OrderedDict with loaders as values
    Yields:
        readout key, dataset item
    """
    assert isinstance(
        loaders, OrderedDict), 'loaders must be an ordered dict'
    for readout_key, outputs in zip(
            cycle(loaders.keys()), alternate(*loaders.values())):
        yield readout_key, outputs


def n_batches(n, loaders):
    """
    Cycles through datasets of dataloaders until n batches are reached
    Args:
        n (int): n number of batches
        loaders: OrderedDict with loaders as values
    Yields:
        readout key, dataset item
    """
    i = 0
    while True:
        for d in cycle_datasets(loaders):
            if i == n:
                return
            yield d
            i += 1
