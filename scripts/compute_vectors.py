import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from itsteer.data import SemevalDataset
from itsteer.prompts import build_prompt
from itsteer.utils import get_device
from itsteer.vectorizers import build_contrastive_direction, collect_hidden_states
from itsteer.tracking import args_to_config, start_wandb_run, log_artifact, wandb


def _truncate_text(text: str, limit: int = 320) -> str:
    snippet = text.strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--data_dir",
        help="Directory containing train.json(l) (legacy). Ignored if --data-file is set.",
    )
    ap.add_argument(
        "--data-file",
        help="Path to a specific JSON/JSONL file to build the vector from (preferred to avoid data leakage).",
    )
    ap.add_argument("--layer", type=int, required=True, help="Layer index to extract from (0-based)")
    ap.add_argument("--last_n", type=int, default=32, help="Average the last N tokens of the prompt")
    ap.add_argument("--contrast", choices=["plausibility", "validity"], default="plausibility")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument(
        "--mean-center",
        action="store_true",
        help="Subtract the global activation mean before computing the contrastive difference (reduces entanglement).",
    )
    ap.add_argument("--wandb-project", type=str, default=None, help="W&B project (required when --wandb-mode != disabled)")
    ap.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity/account")
    ap.add_argument("--wandb-run-name", type=str, default=None, help="Optional run name visible in the W&B UI")
    ap.add_argument("--wandb-group", type=str, default=None, help="Optional group for multi-run sweeps")
    ap.add_argument("--wandb-tags", nargs="*", default=None, help="Space-separated tags for W&B filtering")
    ap.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="disabled",
        help="Use 'online' for live logging or 'offline' to sync later",
    )
    ap.add_argument("--wandb-notes", type=str, default=None, help="Optional W&B notes/description")
    ap.add_argument(
        "--wandb-log-prompts",
        type=int,
        default=0,
        help="Log a preview of the first N prompts per class (0 disables)",
    )
    ap.add_argument(
        "--wandb-upload-vector",
        action="store_true",
        help="Upload the resulting .pt file as a W&B 'vector' artifact",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.data_file and not args.data_dir:
        raise ValueError("Provide --data-file or --data_dir to load training data for vector building.")
    data_source = args.data_file or args.data_dir
    device = get_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    ds = SemevalDataset(data_source)
    stats = {
        "total_examples": len(ds.examples),
        "skipped": 0,
        "pos_used": 0,
        "neg_used": 0,
    }
    preview_rows: list[tuple[str, str, str]] = []
    texts_pos, texts_neg = [], []
    contrast_attr = "plausibility" if args.contrast == "plausibility" else "validity"

    for ex in ds.examples:
        label_value = getattr(ex, contrast_attr)
        if label_value is None:
            stats["skipped"] += 1
            continue
        prompt = build_prompt(ex.syllogism).text
        target_bucket = texts_pos if label_value else texts_neg
        target_bucket.append(prompt)
        key = "pos_used" if label_value else "neg_used"
        stats[key] += 1
        if len(preview_rows) < args.wandb_log_prompts:
            preview_rows.append((ex.id, f"{args.contrast}:{label_value}", _truncate_text(prompt)))

    wandb_settings = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "run_name": args.wandb_run_name,
        "group": args.wandb_group,
        "tags": args.wandb_tags,
        "mode": args.wandb_mode,
        "notes": args.wandb_notes,
    }
    drop_keys = {
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "wandb_group",
        "wandb_tags",
        "wandb_mode",
        "wandb_notes",
        "wandb_log_prompts",
        "wandb_upload_vector",
        "data_file",
    }
    wandb_run = start_wandb_run(
        wandb_settings,
        config={**args_to_config(args, drop_keys=drop_keys), **stats},
    )

    try:
        pos_hs = collect_hidden_states(
            model,
            tok,
            texts_pos,
            layer_idx=args.layer,
            pos_strategy=f"last_n_mean:{args.last_n}",
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        neg_hs = collect_hidden_states(
            model,
            tok,
            texts_neg,
            layer_idx=args.layer,
            pos_strategy=f"last_n_mean:{args.last_n}",
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        # Optional mean-centering to remove global drift (helps reduce unintended directions)
        if args.mean_center:
            all_hs = torch.cat([pos_hs, neg_hs], dim=0)
            global_mean = all_hs.mean(dim=0, keepdim=True)
            pos_hs = pos_hs - global_mean
            neg_hs = neg_hs - global_mean

        v = build_contrastive_direction(pos_hs, neg_hs, normalize=True).cpu()
        suffix = "mc" if args.mean_center else None
        filename = f"{args.contrast}_layer{args.layer}_last{args.last_n}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".pt"
        out_path = os.path.join(args.out_dir, filename)
        payload = {"vector": v, "meta": vars(args), "dims": int(v.numel()), "mean_center": bool(args.mean_center)}
        torch.save(payload, out_path)
        print(f"Saved vector -> {out_path} (dim={v.numel()})")

        if wandb_run:
            wandb_run.summary["vector_dim"] = int(v.numel())
            wandb_run.summary["vector_path"] = out_path
            wandb_run.summary["contrast_attr"] = args.contrast
            table = wandb.Table(columns=["class", "count"])
            table.add_data("positive", stats["pos_used"])
            table.add_data("negative", stats["neg_used"])
            wandb_run.log({"vectors/class_balance": table})
            if preview_rows:
                preview = wandb.Table(columns=["example_id", "label", "prompt_snippet"])
                for row in preview_rows:
                    preview.add_data(*row)
                wandb_run.log({"vectors/prompt_preview": preview})
            if args.wandb_upload_vector:
                artifact_name = args.wandb_run_name or Path(out_path).stem
                log_artifact(
                    wandb_run,
                    name=f"{artifact_name}-vector".replace(" ", "-"),
                    artifact_type="vector",
                    path=out_path,
                    description=f"{args.contrast} steering vector (layer {args.layer}, last {args.last_n})",
                )
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
