from collections.__init__ import OrderedDict
from itertools import cycle


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