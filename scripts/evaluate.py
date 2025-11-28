import argparse
from statistics import mean

import torch

from itsteer.data import SemevalDataset
from itsteer.eval_metrics import accuracy as metric_accuracy, content_effect_metrics
from itsteer.eval_runner import batch_predict, evaluate_model, load_model_and_tokenizer
from itsteer.tracking import args_to_config, start_wandb_run, wandb


def _label_name(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "VALID" if value else "INVALID"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--data_dir",
        help="Legacy: directory with train.json(l) to be split. If --eval-file is set, this is ignored.",
    )
    ap.add_argument(
        "--eval-file",
        help="Explicit JSON/JSONL file to evaluate on (preferred for held-out test). Skips internal split.",
    )
    ap.add_argument("--vector_path", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--last_n", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=None)

    # NEW: replicate the same split so we can exclude validation
    ap.add_argument("--valid_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_on", choices=["all", "valid", "rest"], default="rest")
    ap.add_argument(
        "--alpha-mode",
        choices=["constant", "dynamic_margin"],
        default="constant",
        help="constant: use --alpha everywhere; dynamic_margin: rescale alpha per example based on baseline logit margins",
    )
    ap.add_argument("--dynamic-target-margin", type=float, default=1.5, help="Target margin for scaling dynamic alpha")
    ap.add_argument("--dynamic-min-alpha", type=float, default=0.1, help="Minimum alpha when scheduling dynamically")
    ap.add_argument("--dynamic-max-alpha", type=float, default=3.0, help="Maximum alpha when scheduling dynamically")
    ap.add_argument("--dynamic-margin-eps", type=float, default=1e-3, help="Stability constant for margin division")
    ap.add_argument("--wandb-project", type=str, default=None, help="W&B project (required when --wandb-mode != disabled)")
    ap.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity/account")
    ap.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name")
    ap.add_argument("--wandb-group", type=str, default=None, help="Optional run group")
    ap.add_argument("--wandb-tags", nargs="*", default=None, help="Space-separated W&B tags")
    ap.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="disabled",
        help="Use 'online' for live logging or 'offline' to sync later",
    )
    ap.add_argument("--wandb-notes", type=str, default=None, help="Optional W&B notes")
    ap.add_argument(
        "--wandb-log-predictions",
        type=int,
        default=0,
        help="Log the first N predictions (id, gold, pred, plausibility) to W&B",
    )
    args = ap.parse_args()

    if args.eval_file:
        ds = SemevalDataset(args.eval_file)
        eval_data = list(ds.examples)
        train_rest = valid = None
    else:
        if not args.data_dir:
            raise ValueError("Provide --data_dir for internal split or --eval-file for explicit eval set.")
        ds = SemevalDataset(args.data_dir)
        # Recreate the split deterministically
        train_rest, valid = ds.train_valid_split(valid_frac=args.valid_frac, seed=args.seed)

        if args.eval_on == "all":
            eval_data = list(ds.examples)
        elif args.eval_on == "valid":
            eval_data = list(valid)
        else:  # "rest" -> holdout that was NOT used in alpha search
            eval_data = list(train_rest)

    vec_cfg = None
    vector_dims = None
    if args.vector_path is not None:
        assert args.layer is not None and args.alpha is not None, "Need --layer and --alpha with --vector_path"
        payload = torch.load(args.vector_path)
        v = payload["vector"] if isinstance(payload, dict) and "vector" in payload else payload
        vector_dims = int(v.numel()) if hasattr(v, "numel") else None
        vec_cfg = {"layer": args.layer, "last_n": args.last_n, "v": v, "alpha": args.alpha}

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
        "wandb_log_predictions",
        "alpha_mode",
        "dynamic_target_margin",
        "dynamic_min_alpha",
        "dynamic_max_alpha",
        "dynamic_margin_eps",
        "eval_file",
        "data_dir",
    }
    wandb_run = start_wandb_run(
        wandb_settings,
        config={
            **args_to_config(args, drop_keys=drop_keys),
            "vector_dim": vector_dims,
            "eval_examples": len(eval_data),
            "total_examples": len(ds.examples),
        },
    )

    model_for_eval = None
    tokenizer_for_eval = None
    alpha_schedule = None
    baseline_metrics = None
    alpha_stats = None

    def compute_dynamic_alpha(margin: float) -> float:
        # Preserve the sign of the base alpha while clipping the magnitude to the allowed window.
        scaled = args.alpha * (args.dynamic_target_margin / max(margin, args.dynamic_margin_eps))
        sign = 1.0 if scaled >= 0 else -1.0
        mag = abs(scaled)
        mag = max(args.dynamic_min_alpha, min(args.dynamic_max_alpha, mag))
        return float(sign * mag)

    try:
        if vec_cfg is not None and args.alpha_mode == "dynamic_margin":
            model_for_eval, tokenizer_for_eval = load_model_and_tokenizer(args.model)
            syllogisms = [ex.syllogism for ex in eval_data]
            base_preds, base_logps = batch_predict(
                model_for_eval,
                tokenizer_for_eval,
                syllogisms,
                return_logps=True,
            )
            margins = [abs(lp[0] - lp[1]) for lp in base_logps]
            alpha_schedule = [compute_dynamic_alpha(m) for m in margins]
            y_truth = [ex.validity for ex in eval_data]
            plaus_list = [ex.plausibility for ex in eval_data]
            baseline_metrics = {
                "accuracy": metric_accuracy(y_truth, base_preds),
                **content_effect_metrics(y_truth, base_preds, plaus_list),
            }
            alpha_stats = {
                "min": min(alpha_schedule),
                "max": max(alpha_schedule),
                "mean": mean(alpha_schedule),
            }
        metrics, y_true, preds, plaus = evaluate_model(
            args.model,
            eval_data,
            vector_conf=vec_cfg,
            alpha_schedule=alpha_schedule,
            return_preds=True,
            model=model_for_eval,
            tokenizer=tokenizer_for_eval,
        )
        print(metrics)

        if wandb_run:
            log_payload = {f"metrics/{k}": v for k, v in metrics.items()}
            wandb_run.log(log_payload)
            wandb_run.summary.update(log_payload)
            # Confusion matrix for entries with gold labels
            paired = [(gt, pr) for gt, pr in zip(y_true, preds) if gt is not None]
            if paired:
                class_names = ["INVALID", "VALID"]
                y_true_cm = [1 if gt else 0 for gt, _ in paired]
                preds_cm = [1 if pr else 0 for _, pr in paired]
                wandb_run.log(
                    {
                        "eval/confusion_matrix": wandb.plot.confusion_matrix(
                            y_true=y_true_cm,
                            preds=preds_cm,
                            class_names=class_names,
                        )
                    }
                )
            if args.wandb_log_predictions > 0:
                limit = min(args.wandb_log_predictions, len(eval_data))
                if limit > 0:
                    table = wandb.Table(columns=["example_id", "gold_validity", "pred_validity", "plausibility"])
                    for ex, gt, pr, pl in zip(eval_data[:limit], y_true[:limit], preds[:limit], plaus[:limit]):
                        plaus_label = "unknown" if pl is None else ("PLAUSIBLE" if pl else "IMPLAUSIBLE")
                        table.add_data(ex.id, _label_name(gt), _label_name(pr), plaus_label)
                    wandb_run.log({"eval/predictions_head": table})
            if baseline_metrics:
                wandb_run.log({f"baseline/{k}": v for k, v in baseline_metrics.items()})
            if alpha_schedule:
                wandb_run.log(
                    {
                        "dynamic/alpha_hist": wandb.Histogram(alpha_schedule),
                        "dynamic/alpha_min": alpha_stats["min"],
                        "dynamic/alpha_max": alpha_stats["max"],
                        "dynamic/alpha_mean": alpha_stats["mean"],
                    }
                )
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
