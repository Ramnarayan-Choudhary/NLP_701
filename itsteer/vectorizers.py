from __future__ import annotations
from typing import List, Dict, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

@torch.no_grad()
def collect_hidden_states(model, tokenizer, texts: list[str], layer_idx: int, pos_strategy: str = "last", batch_size: int = 4, max_length: int = 2048):
    """Return (N, H) tensor of hidden states at a given layer & position strategy.
    pos_strategy: 'last' or 'last_n_mean:K' to average last K tokens.
    """
    device = model.device
    hs_list = []
    if pos_strategy.startswith("last_n_mean"):
        K = int(pos_strategy.split(":")[1])
    else:
        K = 1

    for i in tqdm(range(0, len(texts), batch_size), desc=f"collect@L{layer_idx}"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        # hidden_states: tuple(len=L+1); each (B, T, H)
        layer_hs = out.hidden_states[layer_idx]  # (B, T, H)
        if pos_strategy == "last":
            pick = (enc.input_ids != tokenizer.pad_token_id).sum(dim=1) - 1  # (B,)
            picked = layer_hs[torch.arange(layer_hs.size(0), device=device), pick]  # (B, H)
        elif pos_strategy.startswith("last_n_mean"):
            seq_lens = (enc.input_ids != tokenizer.pad_token_id).sum(dim=1)  # (B,)
            picked = []
            for b in range(layer_hs.size(0)):
                l = int(seq_lens[b].item())
                s = max(0, l-K)
                picked.append(layer_hs[b, s:l].mean(dim=0))
            picked = torch.stack(picked, dim=0)
        else:
            raise ValueError(f"Unknown pos_strategy: {pos_strategy}")
        hs_list.append(picked.detach())
    return torch.cat(hs_list, dim=0)  # (N, H)

def build_contrastive_direction(hs_pos: torch.Tensor, hs_neg: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Return direction v = mean(pos) - mean(neg)."""
    v = hs_pos.mean(dim=0) - hs_neg.mean(dim=0)
    if normalize:
        v = v / (v.norm(p=2) + 1e-8)
    return v
