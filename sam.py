"""Sharpness-Aware Minimization (SAM) wrapper for an arbitrary base optimizer.

The LR scheduler should be attached to ``sam.base_optimizer``. This keeps
PyTorch scheduler behavior explicit while SAM exposes the base parameter groups.
"""
from __future__ import annotations

import torch


class SAM:
    """Two-step SAM optimizer wrapper.

    Usage::
        sam = SAM(params, torch.optim.AdamW, rho=0.05, ...)
        loss.backward()
        sam.first_step(zero_grad=True)
        second_loss.backward()
        sam.second_step(zero_grad=True)

    ``params`` may be a list of parameter groups with per-group learning rates.
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        if rho <= 0:
            raise ValueError("SAM rho must be positive.")
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults
        self.rho = float(rho)
        self._perturbations: dict[torch.nn.Parameter, torch.Tensor] = {}

    @property
    def state(self):
        return self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _grad_norm(self):
        norms = [
            p.grad.detach().norm(p=2).to(p.device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            raise RuntimeError("SAM first_step called without gradients.")
        return torch.norm(torch.stack([n.to(norms[0].device) for n in norms]), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        if self._perturbations:
            raise RuntimeError("SAM first_step called before the previous step was completed.")
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p.device, dtype=p.dtype)
                p.add_(e_w)
                self._perturbations[p] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        if not self._perturbations:
            raise RuntimeError("SAM second_step called before first_step.")
        for p, e_w in self._perturbations.items():
            p.sub_(e_w)
        self._perturbations.clear()
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def cancel_step(self, zero_grad: bool = True):
        """Undo a first-step perturbation if the second forward/backward fails."""
        for p, e_w in self._perturbations.items():
            p.sub_(e_w)
        self._perturbations.clear()
        if zero_grad:
            self.zero_grad(set_to_none=True)
