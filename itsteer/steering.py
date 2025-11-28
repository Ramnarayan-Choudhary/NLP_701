from __future__ import annotations

from typing import Optional

import torch

from .hooks import SteeringContext, make_last_n_token_masker


class Steerer:
    def __init__(self, model, layer_idx: int, v: torch.Tensor, alpha: float, last_n: int = 32):
        self.model = model
        self.layer_idx = layer_idx
        self.v = v.to(model.device)
        self.alpha = alpha
        self.last_n = last_n
        self._token_mask_fn = make_last_n_token_masker(last_n)

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

    def context(self, alpha: Optional[float] = None) -> SteeringContext:
        layer = self.resolve_layer()
        effective_alpha = self.alpha if alpha is None else alpha
        return SteeringContext(layer, self.v, effective_alpha, token_mask_fn=self._token_mask_fn)
