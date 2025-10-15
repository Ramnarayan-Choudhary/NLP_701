from __future__ import annotations
import torch
from typing import Optional, Callable
from .hooks import SteeringContext, make_last_n_token_masker

class Steerer:
    def __init__(self, model, layer_idx: int, v: torch.Tensor, alpha: float, last_n: int = 32):
        self.model = model
        self.layer_idx = layer_idx
        self.v = v.to(model.device)
        self.alpha = alpha
        self.last_n = last_n

    def resolve_layer(self):
        # Works for Llama/Qwen-like models: model.model.layers[i]
        base = getattr(self.model, "model", None) or getattr(self.model, "transformer", None)
        if base is None:
            raise ValueError("Unsupported model; expected .model or .transformer attribute")
        layers = getattr(base, "layers", None) or getattr(base, "h", None)
        if layers is None:
            raise ValueError("Unsupported model; expected decoder layers in .layers or .h")
        if self.layer_idx < 0 or self.layer_idx >= len(layers):
            raise IndexError(f"layer_idx {self.layer_idx} out of range 0..{len(layers)-1}")
        return layers[self.layer_idx]

    def context(self) -> SteeringContext:
        layer = self.resolve_layer()
        token_mask_fn = make_last_n_token_masker(self.last_n)
        return SteeringContext(layer, self.v, self.alpha, token_mask_fn=token_mask_fn)
