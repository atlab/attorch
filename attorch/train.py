from collections import OrderedDict


def copy_state(model):
    """
    Given PyTorch module `model`, makes a copy of the state onto CPU.
    Args:
        model: PyTorch module to copy state dict of

    Returns:
        A copy of state dict with all tensors allocated on the CPU
    """
    copy_dict = OrderedDict()
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        copy_dict[k] = v.cpu() if v.is_cuda else v.clone()

    return copy_dict


def early_stopping(model, objective, interval=5, patience=20, start=0, max_iter=1000, maximize=True, tolerance=1e-5,
                   switch_mode=True):
    """
    Early stopping iterator. When it stops, it restores the best previous state of the model.  
    
    Args:
        model:     model that is being optimized 
        objective: objective function that is used for early stopping. Must be of the form objective(model)
        interval:  interval at which objective is evaluated to consider early stopping
        patience:  number of times the objective is allow to not become better before the iterator terminates
        start:     start value for iteration (used to check against `max_iter`)
        max_iter:  maximum number of iterations before the iterator terminated
        maximize:  whether the objective is maximized of minimized
        tolerance: margin by which the new objective score must improve to be considered as an update in best score
        switch_mode: whether to switch model's train mode into eval prior to objective evaluation. If True (default),
                     the model is switched to eval mode before objective evaluation and restored to its previous mode
                     after the evaluation.

    """
    training_status = model.training

    def _objective(mod):
        if switch_mode:
            mod.eval()
        ret = objective(mod)
        if switch_mode:
            mod.train(training_status)
        return ret

    epoch = start
    maximize = float(maximize)
    best_objective = current_objective = _objective(model)
    best_state_dict = copy_state(model)
    patience_counter = 0
    while patience_counter < patience and epoch < max_iter:
        for _ in range(interval):
            epoch += 1
            yield epoch, current_objective

        current_objective = _objective(model)

        if current_objective * (-1) ** maximize < best_objective * (-1) ** maximize - tolerance:
            print('[{:03d}|{:02d}/{:02d}] ---> {}'.format(epoch, patience_counter, patience, current_objective),
                  flush=True)
            best_state_dict = copy_state(model)
            best_objective = current_objective
            patience_counter = 0
        else:
            patience_counter += 1
            print('[{:03d}|{:02d}/{:02d}] -/-> {}'.format(epoch, patience_counter, patience, current_objective),
                  flush=True)
    old_objective = _objective(model)
    model.load_state_dict(best_state_dict)
    print('Restoring best model! {:.6f} ---> {:.6f}'.format(old_objective, _objective(model)))

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