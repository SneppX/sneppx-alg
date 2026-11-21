"""Optimizer Module — AdamW, SGD, Lion, LAMB."""

from typing import Callable, List, Optional, Iterator
from .core import Tensor
from .nn import Module
import numpy as np


class Optimizer:
    def __init__(self, params: Iterator[Tensor], lr: float, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.state = [{} for _ in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.numpy() + self.weight_decay * p.numpy()
            if self.momentum > 0:
                if 'momentum_buf' not in self.state[i]:
                    self.state[i]['momentum_buf'] = np.zeros_like(g)
                buf = self.state[i]['momentum_buf']
                buf = self.momentum * buf + g
                self.state[i]['momentum_buf'] = buf
                g = buf
            p_new = p.numpy() - self.lr * g
            self.params[i] = Tensor(p_new, dtype=p.dtype)


class AdamW(Optimizer):
    def __init__(self, params, lr: float = 0.001, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, lr, weight_decay)
        self.betas = betas
        self.eps = eps
        self._step = 0

    def step(self):
        self._step += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.numpy()
            if 'exp_avg' not in self.state[i]:
                self.state[i]['exp_avg'] = np.zeros_like(g)
                self.state[i]['exp_avg_sq'] = np.zeros_like(g)
            m = self.state[i]['exp_avg']
            v = self.state[i]['exp_avg_sq']
            m = self.betas[0] * m + (1 - self.betas[0]) * g
            v = self.betas[1] * v + (1 - self.betas[1]) * g ** 2
            self.state[i]['exp_avg'] = m
            self.state[i]['exp_avg_sq'] = v
            m_hat = m / (1 - self.betas[0] ** self._step)
            v_hat = v / (1 - self.betas[1] ** self._step)
            p_arr = p.numpy()
            p_arr = p_arr - self.lr * self.weight_decay * p_arr
            p_arr = p_arr - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.params[i] = Tensor(p_arr, dtype=p.dtype)


class Lion(Optimizer):
    def __init__(self, params, lr: float = 0.0001, betas=(0.9, 0.99),
                 weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.betas = betas

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.numpy()
            if 'momentum' not in self.state[i]:
                self.state[i]['momentum'] = np.zeros_like(g)
            m = self.state[i]['momentum']
            update = self.betas[0] * m + (1 - self.betas[0]) * g
            m = self.betas[1] * m + (1 - self.betas[1]) * g
            self.state[i]['momentum'] = m
            p_arr = p.numpy()
            p_arr = p_arr - self.lr * self.weight_decay * p_arr
            p_arr = p_arr - self.lr * np.sign(update)
            self.params[i] = Tensor(p_arr, dtype=p.dtype)


class LAMB(Optimizer):
    def __init__(self, params, lr: float = 0.001, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.betas = betas
        self.eps = eps
        self._step = 0

    def step(self):
        self._step += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.numpy()
            if 'exp_avg' not in self.state[i]:
                self.state[i]['exp_avg'] = np.zeros_like(g)
                self.state[i]['exp_avg_sq'] = np.zeros_like(g)
            m = self.state[i]['exp_avg']
            v = self.state[i]['exp_avg_sq']
            m = self.betas[0] * m + (1 - self.betas[0]) * g
            v = self.betas[1] * v + (1 - self.betas[1]) * g ** 2
            self.state[i]['exp_avg'] = m
            self.state[i]['exp_avg_sq'] = v
            m_hat = m / (1 - self.betas[0] ** self._step)
            v_hat = v / (1 - self.betas[1] ** self._step)
            update = m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.numpy()
            p_norm = np.linalg.norm(p.numpy())
            g_norm = np.linalg.norm(update)
            trust = 1.0
            if p_norm > 0 and g_norm > 0:
                trust = p_norm / g_norm
            p_arr = p.numpy() - self.lr * trust * update
            self.params[i] = Tensor(p_arr, dtype=p.dtype)


class CosineAnnealingLR:
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self._step = 0

    def step(self):
        self._step += 1
        cos = (1 + np.cos(np.pi * self._step / self.T_max)) / 2
        lr = self.eta_min + (self.optimizer.lr - self.eta_min) * cos
        self.optimizer.lr = lr