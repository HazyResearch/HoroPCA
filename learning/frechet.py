"""Frechet data statistics."""

import torch

import geom.poincare as poincare


class Frechet:
    """Class to compute Frechet statiscs (mean and variance)."""

    def __init__(self, lr=1e-1, eps=1e-5, max_steps=5000, max_lr_try=3):
        self.lr = lr
        self.eps = eps
        self.max_steps = max_steps
        self.max_lr_try = max_lr_try
        self.lr_values = [self.lr]
        self.lr_values = [self.lr * (2 ** (i + 1)) for i in range(self.max_lr_try)]
        self.lr_values += [self.lr * (0.5 * (i + 1)) for i in range(self.max_lr_try)]

    def mean(self, x, return_converged=False):
        """Compute the Frechet mean with gradient descent steps."""
        n = x.shape[0]
        mu_init = torch.mean(x, dim=0, keepdim=True)
        has_converged = False
        for i, lr in enumerate(self.lr_values):
            mu = mu_init
            for i in range(self.max_steps):
                log_x = torch.sum(poincare.logmap(mu, x), dim=0, keepdim=True)
                delta_mu = lr / n * log_x
                mu = poincare.expmap(mu, delta_mu)
                if delta_mu.norm(dim=-1, p=2, keepdim=False) < self.eps:
                    has_converged = True
                    break
            if has_converged:
                break
        if not has_converged:
            mu = mu_init
        if return_converged:
            return mu, has_converged
        else:
            return mu

    def variance(self, x, return_converged=False):
        """Compute the Frechet variance."""
        mu, has_converged = self.mean(x, return_converged=True)
        distances = poincare.distance(x, mu.unsqueeze(0)) ** 2
        var = torch.mean(distances)
        if return_converged:
            return var, has_converged
        else:
            return var
