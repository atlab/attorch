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


def cycle_datasets(trainloaders, **kwargs):
    """
    Cycles through datasets of train loaders.

    Args:
        trainloaders: OrderedDict with trainloaders as values
        **kwargs: those arguments will be passed to `attorch.dataset.to_variable`

    Yields:
        readout key, input, targets

    """
    assert isinstance(trainloaders, OrderedDict), 'trainloaders must be an ordered dict'
    for readout_key, outputs in zip(cycle(trainloaders.keys()), alternate(*trainloaders.values(), **kwargs)):
        yield readout_key, outputs