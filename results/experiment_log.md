# Experiment Log

This document promotes the midterm notes into a structured record so future experiments can be reproduced quickly. Every block below ties together the command that was run, the vector that was used, and the metrics that were observed.

## Shared Setup

- **Data**: unless otherwise noted, runs consumed `data/train.jsonl` (downloaded via `scripts/download_data.py`). Some ad-hoc notes reference `data/train.json`, which is the JSON version of the same contents.
- **Vector files**: stored in `artifacts/` as `plausibility_layer{L}_last{N}.pt`. Always record the exact layer/last_n pair when logging a run.
- **Scripts**: `scripts/compute_vectors.py`, `scripts/search_alpha.py`, `scripts/evaluate.py`. All commands are launched from the repository root.
- **Hardware warnings**: runs on the campus cluster emit `spack`/`libc` errors before Hugging Face logs. Metrics still appear at the bottom of each log; ignore the noisy preamble.
- **PyTorch warnings**: upgrade paths are noted (use `dtype=` instead of `torch_dtype`, pass `weights_only=True` to `torch.load`). No behaviour change yet, but keep this in mind when revisiting the scripts.

When adding a new experiment, append a section that records:

1. The command(s) that were executed.
2. The vector configuration (layer, last_n, contrast type).
3. Baseline vs steered metrics (accuracy + total content effect at minimum).
4. The log file path under `logs/`.

## Model-Specific Notes

### Qwen/Qwen3-0.6B

- **Vector**: `artifacts/plausibility_layer14_last16.pt` (contrast = plausibility, built with `scripts/compute_vectors.py --model Qwen/Qwen3-0.6B --layer 14 --last_n 16`).
- **Alpha search (focused sweep)**:
  ```bash
  python scripts/search_alpha.py \
    --model Qwen/Qwen3-0.6B \
    --data_dir data/train.json \
    --vector_path artifacts/plausibility_layer14_last16.pt \
    --layer 14 --last_n 16 \
    --alpha_min 1 --alpha_max 2 --alpha_steps 2 --valid_frac 0.1
  ```
  - Result: `Best: ... alpha=1.0`, metrics = `acc 0.5313`, `totalCE 0.5000`, score `1.0625`.
  - Notes: both probed alphas tied; chose the smaller magnitude for stability.
- **Alpha search (full sweep, logs/qwen3_0.6b_alpha_search.log)**: 21-point sweep from `-3` to `3` plateaued at `acc 0.4896`, `totalCE 0.5` for every alpha. Serves as a sanity check that the vector has little effect without tuned scaling.
- **Baseline evaluation**:
  ```bash
  python scripts/evaluate.py \
    --model Qwen/Qwen3-0.6B \
    --data_dir data/train.json
  ```
  - Metrics: `accuracy 0.5000`, `total_content_effect 0.5000` (`logs/qwen3_0.6b_eval_baseline.log`).
- **Steered evaluation**:
  ```bash
  python scripts/evaluate.py \
    --model Qwen/Qwen3-0.6B \
    --data_dir data/train.json \
    --vector_path artifacts/plausibility_layer14_last16.pt \
    --layer 14 --last_n 16 --alpha 1.0
  ```
  - Metrics: `accuracy 0.5000`, `total_content_effect 0.5000` (`logs/qwen3_0.6b_eval_steered.log`). Steering neither helped nor hurt for this configuration.

### Qwen/Qwen3-1.7B

- **Status**: baseline evaluation was captured; alpha search and steered evaluation logs are currently empty (see `logs/qwen3_1p7b_alpha.log` and `logs/qwen3_1p7b_eval_steered.log`, both `0` bytes). Re-run when time allows.
- **Baseline evaluation**:
  ```bash
  python scripts/evaluate.py \
    --model Qwen/Qwen3-1.5B \
    --data_dir data/train.jsonl
  ```
  - Metrics recorded in `logs/qwen3_1p7b_eval_baseline.log`: `accuracy 0.5586`, `total_content_effect 0.4668` (with large intra-plausibility gap). Use this as the reference point for the re-run.
- **TODO**: recompute a vector (likely `layer 20-24, last_n 32`), run `scripts/search_alpha.py`, and capture both baseline and steered evaluations with the restored logging.

### Qwen/Qwen3-4B

- **Vector**: `artifacts/plausibility_layer24_last32.pt`.
- **Alpha search**:
  ```bash
  python scripts/search_alpha.py \
    --model Qwen/Qwen3-4B --layer 24 --last_n 32 \
    --vector_path artifacts/plausibility_layer24_last32.pt \
    --data_dir data/ \
    --alpha_min -3.0 --alpha_max 3.0 --alpha_steps 8 \
    --valid_frac 0.2 --seed 42 | tee logs/qwen3_4b_alpha.log
  ```
  - Highlights from `logs/qwen3_4b_alpha.log`: best score at `alpha = -3.0` with `accuracy 0.7396`, `totalCE 0.1851`.
- **Evaluation (rest split)**:
  ```bash
  ALPHA=$(awk '/^BEST_ALPHA /{print $2}' logs/qwen3_4b_alpha.log)
  python scripts/evaluate.py \
    --model Qwen/Qwen3-4B \
    --data_dir data/ \
    --vector_path artifacts/plausibility_layer24_last32.pt \
    --alpha "$ALPHA" --layer 24 --last_n 32 \
    --eval_on rest
  ```
  - Baseline (`logs/qwen3_4b_eval_baseline.log`): `accuracy 0.7214`, `total_content_effect 0.3029`.
  - Steered (`logs/qwen3_4b_eval_steered.log` & console snippet): `accuracy 0.7174`, `total_content_effect 0.3199`. Accuracy dipped slightly while CE worsened—worth re-checking whether `alpha=-3` overfits the validation split.

### Llama 3 8B

- **Vector**: `artifacts/plausibility_layer10_last16.pt` (layer inferred from log naming; adjust once a dedicated compute log is added).
- **Alpha search** (`logs/llama3_8b_alpha_search.log`):
  ```bash
  python scripts/search_alpha.py \
    --model meta-llama/Meta-Llama-3-8B \
    --layer 10 --last_n 16 \
    --vector_path artifacts/plausibility_layer10_last16.pt \
    --data_dir data/train.jsonl \
    --alpha_min -2 --alpha_max 2 --alpha_steps 8
  ```
  - Best alpha `≈ 0.286` with `accuracy 0.5677`, `totalCE 0.1554`.
- **Evaluations**:
  ```bash
  python scripts/evaluate.py --model meta-llama/Meta-Llama-3-8B --data_dir data/train.jsonl
  python scripts/evaluate.py --model meta-llama/Meta-Llama-3-8B \
    --data_dir data/train.jsonl \
    --vector_path artifacts/plausibility_layer10_last16.pt \
    --layer 10 --last_n 16 --alpha 0.2857142857142857
  ```
  - Baseline (`logs/llama3_8b_eval_baseline.log`): `accuracy 0.5195`, `totalCE 0.2842`.
  - Steered (`logs/llama3_8b_eval_steered.log`): `accuracy 0.5427`, `totalCE 0.2584`. Steering recovered ~2.3 accuracy points and reduced total CE by ~0.026.

## Open Items

- Re-run and log Qwen/Qwen3-1.7B (alpha search + steered evaluation).
- Capture vector-building commands for each artifact (`artifacts/vectors/` is currently unlabeled).
- Consider adding a short script that parses `logs/*.log` and appends a markdown row automatically.
- Address the `torch_dtype`/`torch.load` warnings inside `scripts/*.py` (e.g., replace `torch_dtype=` with `dtype=` and pass `weights_only=True` where possible).
