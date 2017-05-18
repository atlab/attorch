import numpy as np

from graphviz import Digraph
from scipy.interpolate import InterpolatedUnivariateSpline
from torch.autograd import Variable


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

