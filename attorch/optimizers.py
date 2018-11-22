import math
import torch
from torch import optim


class ActiveSGD(optim.SGD):
    def __init__(self, params, lr,
                 momentum=0, dampening=0, weight_decay=0, nesterov=False):
        params = list(params)
        assert not isinstance(
            params[0], dict), 'Only a single param group is supported'
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def step(self, active_params=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            active_params (iterable | None):
                An iterable containing parameters to be updated by 
                this optimization step
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        lr = self.param_groups[0]['lr']
        weight_decay = self.param_groups[0]['weight_decay']
        momentum = self.param_groups[0]['momentum']
        dampening = self.param_groups[0]['dampening']
        nesterov = self.param_groups[0]['nesterov']

        params = active_params if active_params is not None \
            else self.param_groups[0]['params']

        for p in params:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(
                        p.data)
                    buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf

            p.data.add_(-lr, d_p)

        return loss


def cosine_schedule(max_value, min_value, period_init=10, period_mult=2, n=1000):
    """ Generator that produces cosine learning rate schedule,
        as defined in Loshchilov & Hutter, 2017, https://arxiv.org/abs/1608.03983
    Arguments:
        max_value (float): maximum value
        min_value (float): minimum value
        period_init (int): intial learning rate restart period
        period_mult (int): period multiplier that is applied at each restart
        n (int): number of iterations
    Yield:
        learning rate
    """
    i = 0
    epoch = 0
    period = period_init
    while i < n:
        lr = min_value + (max_value - min_value) * \
            (1 + math.cos(math.pi * epoch / period)) / 2
        yield lr
        i += 1
        epoch += 1
        if epoch % period == 0:
            period *= period_mult
            epoch = 0
