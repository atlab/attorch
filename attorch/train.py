from copy import deepcopy


def early_stopping(model, objective, interval=5, patience=20, start=0, max_iter=1000, maximize=True):
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
    """
    epoch = start
    maximize = float(maximize)
    best_objective = current_objective = objective(model)
    best_state_dict = deepcopy(model.state_dict())
    patience_counter = 0
    while patience_counter < patience and epoch < max_iter:
        for _ in range(interval):
            epoch += 1
            yield epoch, current_objective

        current_objective = objective(model)

        if current_objective * (-1)**maximize < best_objective * (-1)**maximize:
            print('[{:03d}|{:02d}/{:02d}] -> {}'.format(epoch, patience_counter, patience, current_objective), flush=True)
            best_state_dict = deepcopy(model.state_dict())
            best_objective = current_objective
            patience_counter = 0
        else:
            patience_counter += 1
            print('[{:03d}|{:02d}/{:02d}]'.format(epoch, patience_counter, patience), flush=True)
    old_objective = objective(model)
    model.load_state_dict(best_state_dict)
    print('Restoring best model! {:.6f} -> {:.6f}'.format(old_objective, objective(model)))