import argparse

import numpy as np
import torch

from itsteer.data import SemevalDataset
from itsteer.eval_runner import evaluate_model
from itsteer.tracking import args_to_config, start_wandb_run, wandb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--data_dir",
        help="Legacy: directory with train.json(l) to be split internally. Use --train-file/--val-file for explicit splits.",
    )
    ap.add_argument(
        "--train-file",
        help="Explicit training JSON/JSONL file (used for bookkeeping; alpha is tuned on --val-file).",
    )
    ap.add_argument(
        "--val-file",
        help="Explicit validation JSON/JSONL file for alpha search (preferred to avoid leakage).",
    )
    ap.add_argument("--vector_path", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--last_n", type=int, default=32)
    ap.add_argument("--alpha_min", type=float, default=-2.0)
    ap.add_argument("--alpha_max", type=float, default=2.0)
    ap.add_argument("--alpha_steps", type=int, default=21)
    ap.add_argument("--valid_frac", type=float, default=0.2)
    ap.add_argument(
        "--objective",
        choices=["ratio", "acc_minus_ce", "accuracy"],
        default="ratio",
        help="ratio: accuracy / total_content_effect; acc_minus_ce: accuracy - total_content_effect; accuracy: max accuracy",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", type=str, default=None, help="W&B project (required when --wandb-mode != disabled)")
    ap.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity/account")
    ap.add_argument("--wandb-run-name", type=str, default=None, help="Optional run name")
    ap.add_argument("--wandb-group", type=str, default=None, help="Optional run group")
    ap.add_argument("--wandb-tags", nargs="*", default=None, help="Space-separated tags")
    ap.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="disabled",
        help="Use 'online' for live logging or 'offline' for later sync",
    )
    ap.add_argument("--wandb-notes", type=str, default=None, help="Optional W&B notes field")
    ap.add_argument(
        "--wandb-log-table",
        type=int,
        default=0,
        help="Log at most N alpha rows to W&B (0 logs the full sweep, negative disables the table)",
    )
    args = ap.parse_args()

    # Safe vector load; fall back if torch version doesn't support weights_only
    try:
        v = torch.load(args.vector_path, weights_only=True)["vector"]
    except TypeError:
        v = torch.load(args.vector_path)["vector"]

    if args.val_file:
        valid_ds = SemevalDataset(args.val_file)
        valid = valid_ds.examples
        train_examples = 0
        if args.train_file:
            train_ds = SemevalDataset(args.train_file)
            train_examples = len(train_ds.examples)
    else:
        if not args.data_dir:
            raise ValueError("Provide --data_dir for internal split or --val-file for an explicit validation set.")
        ds = SemevalDataset(args.data_dir)
        train, valid = ds.train_valid_split(valid_frac=args.valid_frac, seed=args.seed)
        train_examples = len(train)

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
        "wandb_log_table",
        "train_file",
        "val_file",
        "data_dir",
    }
    wandb_run = start_wandb_run(
        wandb_settings,
        config={
            **args_to_config(args, drop_keys=drop_keys),
            "train_examples": train_examples,
            "valid_examples": len(valid),
        },
    )

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    best = None  # tuple: (score, alpha_float, metrics_dict)
    sweep_rows = []

    try:
        for step_idx, alpha in enumerate(alphas):
            a = float(alpha)
            vec_cfg = {"layer": args.layer, "last_n": args.last_n, "v": v, "alpha": a}
            metrics = evaluate_model(args.model, valid, vector_conf=vec_cfg)
            acc = metrics["accuracy"]
            ce  = metrics["total_content_effect"]
            if args.objective == "ratio":
                score = (acc / (ce + 1e-6))
            elif args.objective == "acc_minus_ce":
                score = (acc - ce)
            else:  # accuracy
                score = acc
            print(f"alpha={a:.3f} -> acc={acc:.4f} totalCE={ce:.4f} score={score:.4f}")
            sweep_rows.append({"alpha": a, "score": score, **metrics})
            if wandb_run:
                wandb_run.log(
                    {
                        "search/alpha": a,
                        "search/score": score,
                        "metrics/accuracy": acc,
                        "metrics/total_content_effect": ce,
                        "metrics/positive_content_effect": metrics.get("positive_content_effect"),
                        "metrics/negative_content_effect": metrics.get("negative_content_effect"),
                    },
                    step=step_idx,
                )
            if best is None or score > best[0]:
                best = (score, a, metrics)

        # Human-friendly and machine-friendly lines
        print("Best:", best)
        # This exact line is parsed by your shell to capture alpha:
        print("BEST_ALPHA", best[1])

        if wandb_run and best is not None:
            wandb_run.summary["best_alpha"] = best[1]
            for key, value in best[2].items():
                wandb_run.summary[f"best/{key}"] = value
            if args.wandb_log_table >= 0:
                limit = len(sweep_rows) if args.wandb_log_table == 0 else min(args.wandb_log_table, len(sweep_rows))
                if limit > 0:
                    table = wandb.Table(columns=["alpha", "score", "accuracy", "total_content_effect"])
                    for row in sweep_rows[:limit]:
                        table.add_data(row["alpha"], row["score"], row["accuracy"], row["total_content_effect"])
                    wandb_run.log({"search/alpha_sweep": table})
    finally:
        if wandb_run:
            wandb_run.finish()

if __name__ == "__main__":
    main()
