import copy
import torch
import numpy as np
import torch.nn as nn

from torch.optim.optimizer import Optimizer, required

# code from https://github.com/lxcnju/FedRepo/blob/main/algorithms/fednova.py
class NovaOptimizer(Optimizer):
    """ gmf: global momentum
        prox_mu: mu of proximal term
        ratio: client weight
    """

    def __init__(
        self, params, lr, ratio, gmf, prox_mu=0,
        momentum=0, dampening=0, weight_decay=0, nesterov=False, variance=0
    ):
        self.gmf = gmf
        self.ratio = ratio
        self.prox_mu = prox_mu
        self.momentum = momentum
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0

        if lr < 0.0:
            raise ValueError("Invalid lr: {}".format(lr))

        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov, variance=variance
        )
        super(NovaOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NovaOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                # weight_decay
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # save the first parameter w0
                param_state = self.state[p]
                if "old_init" not in param_state:
                    param_state["old_init"] = torch.clone(p.data).detach()

                # momentum:
                # v_{t+1} = rho * v_t + g_t
                # g_t = v_{t+1}
                # rho = momentum
                local_lr = group["lr"]
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = torch.clone(d_p).detach()
                        param_state["momentum_buffer"] = buf
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)

                        # update momentum buffer !!!
                        param_state["momentum_buffer"] = buf

                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # add proximal updates: g_t = g_t + prox_mu * (w - w0)
                if self.prox_mu != 0:
                    d_p.add_(self.prox_mu, p.data - param_state["old_init"])

                # updata accumulated local updates
                # sum(g_0, g_1, ..., g_t)
                if "cum_grad" not in param_state:
                    param_state["cum_grad"] = torch.clone(d_p).detach()
                    param_state["cum_grad"].mul_(local_lr)
                else:
                    param_state["cum_grad"].add_(local_lr, d_p)

                # update: w_{t+1} = w_t - lr * g_t
                p.data.add_(-1.0 * local_lr, d_p)

        # compute local normalizing vec, a_i
        # For momentum: a_i = [(1 - rho)^{tau_i - 1}/(1 - rho), ..., 1]
        # 1, 1 + rho, 1 + rho + rho^2, ...
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter

        # proximal: a_i = [(1 - eta * mu)^{\tau_i - 1}, ..., 1]
        # 1, 1 - eta * mu, (1 - eta * mu)^2 + 1, ...
        self.etamu = local_lr * self.prox_mu
        if self.etamu != 0:
            self.local_normalizing_vec *= (1 - self.etamu)
            self.local_normalizing_vec += 1

        # FedAvg: no momentum, no proximal, [1, 1, 1, ...]
        if self.momentum == 0 and self.etamu == 0:
            self.local_normalizing_vec += 1

        self.local_steps += 1
        return 


class ScaffoldOptimizer(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        lr, 
        momentum=0, 
        dampening=0, 
        weight_decay=0
    ):
        defaults = dict(
            lr=lr, 
            weight_decay=weight_decay, 
            momentum=momentum, 
            dampening=dampening,
            nesterov=False
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(ScaffoldOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, server_control, client_control, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        ng = len(self.param_groups[0]["params"])
        names = list(server_control.keys())

        # BatchNorm: running_mean/std, num_batches_tracked
        names = [name for name in names if "running" not in name]
        names = [name for name in names if "num_batch" not in name]

        t = 0
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                param_state = self.state[p]
                
                # weight_decay
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = torch.clone(d_p).detach()
                        param_state["momentum_buffer"] = buf
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)

                        # update momentum buffer !!!
                        param_state["momentum_buffer"] = buf
                    d_p = buf

                # d_p = p.grad.data
                c = server_control[names[t]]
                ci = client_control[names[t]]

                # print(names[t], p.shape, c.shape, ci.shape)
                d_p = d_p + c.data - ci.data

                p.data = p.data - d_p.data * group["lr"]
                t += 1
        assert t == ng
        return loss

# implementation from: https://github.com/JYWa/FedNova/blob/master/distoptim/FedProx.py
class FedProxOptimizer(Optimizer):
    r"""Implements FedAvg and FedProx. Local Solver can have momentum.
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        ratio (float): relative sample size of client
        gmf (float): global/server/slow momentum factor
        mu (float): parameter for proximal local SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(
        self, 
        params, 
        lr=required, 
        momentum=0, 
        dampening=0,
        weight_decay=0, 
        nesterov=False, 
        variance=0, 
        mu=0
    ):
        
        self.itr = 0
        self.a_sum = 0
        self.mu = mu


        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedProxOptimizer, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(FedProxOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                
                param_state = self.state[p]
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # apply proximal update
                d_p.add_(self.mu, p.data - param_state['old_init'])
                p.data.add_(-group['lr'], d_p)

        return loss