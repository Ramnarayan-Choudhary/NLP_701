from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import CLASS_LABELS, build_prompt
from .utils import get_device
from .eval_metrics import accuracy, content_effect_metrics
from .steering import Steerer


def load_model_and_tokenizer(model_name: str):
    device = get_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    return model, tok


def score_candidates(
    model,
    tokenizer,
    prompt_text: str,
    candidates: list[str] = CLASS_LABELS,
    max_length: int = 2048,
    steerer: Optional[Steerer] = None,
    alpha: Optional[float] = None,
) -> list[float]:
    """Return log-prob for each candidate string by teacher-forcing the candidate after prompt."""
    device = model.device
    enc_prompt = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    B, T = 1, enc_prompt.input_ids.size(1)
    # Build batch of prompt+candidate
    all_input_ids = []
    all_labels = []
    for cand in candidates:
        cand_ids = tokenizer(cand, add_special_tokens=False).input_ids
        ids = enc_prompt.input_ids[0].tolist() + cand_ids
        all_input_ids.append(ids)
        # Labels: -100 for prompt, cand token ids for candidate portion
        labels = [-100] * T + cand_ids
        all_labels.append(labels)
    # Pad
    max_len = max(len(x) for x in all_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(candidates), max_len), pad_id, dtype=torch.long, device=device)
    labels_t = torch.full((len(candidates), max_len), -100, dtype=torch.long, device=device)
    for i, (ids, labs) in enumerate(zip(all_input_ids, all_labels)):
        input_ids[i, : len(ids)] = torch.tensor(ids, device=device)
        labels_t[i, : len(labs)] = torch.tensor(labs, device=device)
    attn_mask = (input_ids != pad_id).long()

    ctx = steerer.context(alpha=alpha) if steerer is not None else nullcontext()
    with ctx:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels_t, use_cache=False)
            logits = out.logits  # (B, L, V)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels_t[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            # (B, L-1)
            per_pos_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
                shift_labels.size()
            )
            # mask valid positions
            mask = (shift_labels != -100).float()
            per_row_neglogp = (per_pos_loss * mask).sum(dim=1)
            logps = (-per_row_neglogp).detach().cpu().tolist()
    return logps


def batch_predict(
    model,
    tokenizer,
    syllogisms: list[str],
    steerer: Optional[Steerer] = None,
    alphas: Optional[Sequence[float]] = None,
    return_logps: bool = False,
) -> tuple[list[bool], list[list[float]]] | list[bool]:
    preds: list[bool] = []
    logps_all: list[list[float]] = []
    for idx, s in enumerate(syllogisms):
        prompt = build_prompt(s)
        alpha = None
        if alphas is not None:
            alpha = alphas[idx]
        logps = score_candidates(
            model,
            tokenizer,
            prompt.text,
            candidates=prompt.labels,
            steerer=steerer,
            alpha=alpha,
        )
        pred = True if logps[0] > logps[1] else False  # VALID vs INVALID
        preds.append(pred)
        if return_logps:
            logps_all.append(logps)
    if return_logps:
        return preds, logps_all
    return preds


def evaluate_model(
    model_name: str,
    dataset,
    vector_conf: Optional[dict] = None,
    alpha_schedule: Optional[Sequence[float]] = None,
    return_preds: bool = False,
    model=None,
    tokenizer=None,
):
    owns_model = False
    if model is None or tokenizer is None:
        model, tok = load_model_and_tokenizer(model_name)
        owns_model = True
    else:
        tok = tokenizer

    steerer: Optional[Steerer] = None
    if vector_conf is not None:
        steerer = Steerer(
            model,
            layer_idx=vector_conf["layer"],
            v=vector_conf["v"],
            alpha=vector_conf["alpha"],
            last_n=vector_conf.get("last_n", 32),
        )

    syllogisms = [ex.syllogism for ex in dataset]
    y_true = [ex.validity for ex in dataset]
    plaus = [ex.plausibility for ex in dataset]
    preds = batch_predict(
        model,
        tok,
        syllogisms,
        steerer=steerer,
        alphas=alpha_schedule,
    )
    acc = accuracy(y_true, preds)
    ce = content_effect_metrics(y_true, preds, plaus)
    metrics = {"accuracy": acc, **ce}
    if return_preds:
        return metrics, y_true, preds, plaus
    return metrics
