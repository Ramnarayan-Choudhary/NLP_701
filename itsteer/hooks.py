from __future__ import annotations
from typing import Optional, Callable
import torch
from torch import nn

class SteeringHook:
    """Forward hook that adds alpha * v to a module's output at selected token positions."""
    def __init__(self, v: torch.Tensor, alpha: float, token_mask_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        # v: (hidden,)
        self.v = v
        self.alpha = alpha
        self.token_mask_fn = token_mask_fn
        self.handle = None

    def __call__(self, module: nn.Module, inputs, output):
        # output: Tensor or tuple; we expect Tensor (B, T, H) for decoder block
        if isinstance(output, tuple):
            hs = output[0]
            rest = list(output[1:])
        else:
            hs = output
            rest = None

        if self.token_mask_fn is None:
            mask = None
        else:
            # token_mask over sequence length; shape (B, T) -> (B, T, 1)
            mask = self.token_mask_fn(hs)
            if mask is not None and mask.dim() == 2:
                mask = mask.unsqueeze(-1)

        add = self.alpha * self.v
        while add.dim() < hs.dim():
            add = add.unsqueeze(0)
        if mask is not None:
            hs = hs + mask * add
        else:
            hs = hs + add

        if rest is not None:
            return (hs, *rest)
        return hs

class SteeringContext:
    """Context manager to register/unregister steering hook on a module."""
    def __init__(self, module: nn.Module, v: torch.Tensor, alpha: float, token_mask_fn: Optional[Callable]=None):
        self.module = module
        self.hook = SteeringHook(v, alpha, token_mask_fn)

    def __enter__(self):
        self.hook.handle = self.module.register_forward_hook(self.hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.hook.handle is not None:
            self.hook.handle.remove()
            self.hook.handle = None
        return False

def make_last_n_token_masker(last_n: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def mask_fn(hs: torch.Tensor) -> torch.Tensor:
        # hs: (B, T, H)
        B, T, _ = hs.shape
        n = min(last_n, T)
        mask = torch.zeros(B, T, device=hs.device, dtype=hs.dtype)
        mask[:, T-n:] = 1.0
        return mask
    return mask_fn
