"""
End-to-end helper to split data, build a steering vector, search alpha, and evaluate.

Expected inputs (explicit, to avoid leakage):
- --train-file  : labeled JSON/JSONL used to build v (train half)
- --pilot-file  : labeled JSON/JSONL used as validation for alpha search
- --test-file   : labeled JSON/JSONL held out for final eval

This script shells out to existing CLIs:
- scripts/compute_vectors.py
- scripts/search_alpha.py
- scripts/evaluate.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable

from itsteer.data import Example, SemevalDataset, stratified_split_examples


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command, echo stdout/stderr, and fail fast on nonzero exit."""
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def save_examples(path: Path, examples: Iterable[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "id": ex.id,
            "syllogism": ex.syllogism,
            "validity": ex.validity,
            "plausibility": ex.plausibility,
        }
        for ex in examples
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_best_alpha(stdout: str) -> float | None:
    for line in stdout.splitlines():
        if line.startswith("BEST_ALPHA"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    return None


def append_wandb_args(cmd: list[str], args, run_suffix: str) -> list[str]:
    """Optionally add W&B flags to a command."""
    if args.wandb_mode != "disabled" and args.wandb_project:
        cmd.extend(["--wandb-mode", args.wandb_mode])
        cmd.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            cmd.extend(["--wandb-entity", args.wandb_entity])
        if args.wandb_group:
            cmd.extend(["--wandb-group", args.wandb_group])
        run_name = f"{args.wandb_prefix}-{run_suffix}"
        cmd.extend(["--wandb-run-name", run_name])
        if args.wandb_tags:
            cmd.extend(["--wandb-tags", *args.wandb_tags])
        if args.wandb_notes:
            cmd.extend(["--wandb-notes", args.wandb_notes])
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name for steering.")
    ap.add_argument("--train-file", required=True, help="Train JSON/JSONL (used to build v).")
    ap.add_argument("--pilot-file", help="Pilot/validation JSON/JSONL for alpha search (optional if using --alpha-val-fraction).")
    ap.add_argument("--test-file", required=True, help="Held-out test JSON/JSONL for final eval (may be split if --alpha-val-fraction>0).")
    ap.add_argument("--split-ratio", type=float, default=None, help="If set, first split train-file before building v.")
    ap.add_argument("--layer", type=int, default=16, help="Single layer to use if --layers is not set.")
    ap.add_argument("--layers", type=str, default=None, help="Comma-separated list of layers to sweep (e.g., 14,16,18).")
    ap.add_argument("--last-n", type=int, default=32, help="Single last_n to use if --last-n-list is not set.")
    ap.add_argument("--last-n-list", type=str, default=None, help="Comma-separated list of last_n values to sweep (e.g., 16,32).")
    ap.add_argument("--contrast", choices=["plausibility", "validity"], default="plausibility")
    ap.add_argument("--alpha-min", type=float, default=-2.0)
    ap.add_argument("--alpha-max", type=float, default=2.0)
    ap.add_argument("--alpha-steps", type=int, default=21)
    ap.add_argument("--alpha-objective", choices=["ratio", "acc_minus_ce", "accuracy"], default="ratio")
    ap.add_argument("--out-dir", default="pipeline_runs/auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha-val-fraction", type=float, default=0.0, help="If >0, carve this fraction of test-file (stratified) as alpha validation, leave rest for final eval.")
    ap.add_argument("--run-dynamic-eval", action="store_true", help="Also run dynamic alpha eval.")
    ap.add_argument("--dynamic-target-margin", type=float, default=1.5)
    ap.add_argument("--dynamic-min-alpha", type=float, default=0.1)
    ap.add_argument("--dynamic-max-alpha", type=float, default=3.0)
    ap.add_argument("--mean-center", action="store_true", help="Use mean-centering when building v.")
    ap.add_argument("--wandb-project", type=str, default=None, help="W&B project for logging (set to enable).")
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-group", type=str, default=None)
    ap.add_argument("--wandb-tags", nargs="*", default=None)
    ap.add_argument("--wandb-notes", type=str, default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    ap.add_argument("--wandb-prefix", type=str, default="pipeline", help="Prefix for run names on W&B.")
    args = ap.parse_args()

    out_base = Path(args.out_dir)
    splits_dir = out_base / "splits"
    artifacts_dir = out_base / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_source = Path(args.train_file)
    test_file = Path(args.test_file)

    # Optional: split the provided train-file to mimic 50/50 train/test
    if args.split_ratio:
        ds = SemevalDataset(str(train_source))
        train_part, rest_part = stratified_split_examples(ds.examples, train_ratio=args.split_ratio, seed=args.seed)
        train_split_path = splits_dir / f"{train_source.stem}_train.json"
        rest_split_path = splits_dir / f"{train_source.stem}_rest.json"
        save_examples(train_split_path, train_part)
        save_examples(rest_split_path, rest_part)
        train_for_vector = train_split_path
        print(f"Saved stratified split -> train: {train_split_path} ({len(train_part)}), rest: {rest_split_path} ({len(rest_part)})")
    else:
        train_for_vector = train_source

    # Optional: carve a val split from the provided test file if requested (distribution-matched alpha tuning)
    if args.alpha_val_fraction and args.alpha_val_fraction > 0:
        ds_test = SemevalDataset(str(test_file))
        val_alpha, test_final = stratified_split_examples(ds_test.examples, train_ratio=args.alpha_val_fraction, seed=args.seed)
        val_alpha_path = splits_dir / f"{test_file.stem}_alpha_val.json"
        test_final_path = splits_dir / f"{test_file.stem}_final_eval.json"
        save_examples(val_alpha_path, val_alpha)
        save_examples(test_final_path, test_final)
        pilot_file = val_alpha_path
        test_file = test_final_path
        print(f"Alpha val (from test) -> {val_alpha_path} ({len(val_alpha)}) | Final eval -> {test_final_path} ({len(test_final)})")
    else:
        if not args.pilot_file:
            raise ValueError("Provide --pilot-file or set --alpha-val-fraction>0 to derive a val split.")
        pilot_file = Path(args.pilot_file)

    # Prepare layer/last_n sweep
    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        layers = [args.layer]
    if args.last_n_list:
        last_ns = [int(x) for x in args.last_n_list.split(",") if x.strip()]
    else:
        last_ns = [args.last_n]

    best_combo = None  # (metric, layer, last_n, alpha, vector_path)

    for lyr in layers:
        for ln in last_ns:
            combo_tag = f"L{lyr}_N{ln}"
            vector_out_dir = artifacts_dir / combo_tag
            vector_out_dir.mkdir(parents=True, exist_ok=True)

            # 1) Build vector
            compute_cmd = [
                "python",
                "scripts/compute_vectors.py",
                "--model",
                args.model,
                "--data-file",
                str(train_for_vector),
                "--layer",
                str(lyr),
                "--last_n",
                str(ln),
                "--contrast",
                args.contrast,
                "--out_dir",
                str(vector_out_dir),
            ]
            if args.mean_center:
                compute_cmd.append("--mean-center")
            compute_cmd = append_wandb_args(compute_cmd, args, run_suffix=f"compute-{combo_tag}")
            run_cmd(compute_cmd)

            vector_path = None
            for path in sorted(vector_out_dir.glob(f"{args.contrast}_layer{lyr}_last{ln}*.pt")):
                vector_path = path
            if not vector_path:
                raise RuntimeError(f"Could not locate saved vector for {combo_tag}; check compute_vectors output.")

            # 2) Alpha search on pilot/val
            search_cmd = [
                "python",
                "scripts/search_alpha.py",
                "--model",
                args.model,
                "--val-file",
                str(pilot_file),
                "--vector_path",
                str(vector_path),
                "--layer",
                str(lyr),
                "--last_n",
                str(ln),
                "--alpha_min",
                str(args.alpha_min),
                "--alpha_max",
                str(args.alpha_max),
                "--alpha_steps",
                str(args.alpha_steps),
                "--objective",
                args.alpha_objective,
            ]
            search_cmd = append_wandb_args(search_cmd, args, run_suffix=f"alpha-{combo_tag}")
            search_proc = run_cmd(search_cmd)
            best_alpha = parse_best_alpha(search_proc.stdout or "")
            if best_alpha is None:
                raise RuntimeError(f"Failed to parse BEST_ALPHA for {combo_tag}")
            print(f"[{combo_tag}] Selected alpha: {best_alpha}")

            # 2b) Evaluate on val to pick best combo (static only)
            eval_val_cmd = [
                "python",
                "scripts/evaluate.py",
                "--model",
                args.model,
                "--eval-file",
                str(pilot_file),
                "--vector_path",
                str(vector_path),
                "--layer",
                str(lyr),
                "--last_n",
                str(ln),
                "--alpha",
                str(best_alpha),
            ]
            eval_val_cmd = append_wandb_args(eval_val_cmd, args, run_suffix=f"eval-val-{combo_tag}")
            val_proc = run_cmd(eval_val_cmd)
            # crude parse of accuracy from last line if present
            val_acc = None
            for line in (val_proc.stdout or "").splitlines()[::-1]:
                if '"accuracy"' in line:
                    try:
                        val_acc = float(line.split(":")[1].strip().strip(",").strip("}"))
                        break
                    except Exception:
                        continue
            if val_acc is None:
                # fallback: try to eval on test file? skip
                pass

            if val_acc is not None:
                if best_combo is None or val_acc > best_combo[0]:
                    best_combo = (val_acc, lyr, ln, best_alpha, vector_path)

    if best_combo is None:
        raise RuntimeError("No valid combo found during sweep.")

    _, best_layer, best_ln, best_alpha, best_vector = best_combo
    print(f"Best combo -> layer {best_layer}, last_n {best_ln}, alpha {best_alpha}")

    # 3) Evaluate on held-out test with best combo
    eval_cmd = [
        "python",
        "scripts/evaluate.py",
        "--model",
        args.model,
        "--eval-file",
        str(test_file),
        "--vector_path",
        str(best_vector),
        "--layer",
        str(best_layer),
        "--last_n",
        str(best_ln),
        "--alpha",
        str(best_alpha),
    ]
    eval_cmd = append_wandb_args(eval_cmd, args, run_suffix="eval-static")
    run_cmd(eval_cmd)

    if args.run_dynamic_eval:
        eval_dyn_cmd = [
            "python",
            "scripts/evaluate.py",
            "--model",
            args.model,
            "--eval-file",
            str(test_file),
            "--vector_path",
            str(best_vector),
            "--layer",
            str(best_layer),
            "--last_n",
            str(best_ln),
            "--alpha",
            str(best_alpha),
            "--alpha-mode",
            "dynamic_margin",
            "--dynamic-target-margin",
            str(args.dynamic_target_margin),
            "--dynamic-min-alpha",
            str(args.dynamic_min_alpha),
            "--dynamic-max-alpha",
            str(args.dynamic_max_alpha),
        ]
        eval_dyn_cmd = append_wandb_args(eval_dyn_cmd, args, run_suffix="eval-dynamic")
        run_cmd(eval_dyn_cmd)


if __name__ == "__main__":
    main()
