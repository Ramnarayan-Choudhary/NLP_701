# SemEval-2026 Task 11 — Inference-Time Steering (ITS)

This repository provides a complete **inference-time steering** pipeline to mitigate **content effects** in syllogistic reasoning, following the setup of **SemEval-2026 Task 11**.

It includes:

- Vector extraction from hidden states (**plausible vs. implausible**) while controlling for formal validity
- Test-time **activation steering** hooks for Hugging Face LLMs
- Static **and** tunable scaling of steering (`alpha`), with grid-search
- A lightweight classifier-free decision rule via **log-prob comparison** over candidate answers (`VALID` vs `INVALID`)
- Official task **metrics**: Accuracy, Intra/Cross Plausibility Content Effect, Total Content Effect

> This code is model-agnostic and works with any decoder-only Hugging Face model (e.g., Qwen2.5, Llama 3.2, Mistral/Gemma).

---

## Quickstart

```bash
# 1) Create env and install
pip install -e .

# 2) Download the pilot/training data (from the task repo)
python scripts/download_data.py --out data/

# 3) Compute steering vectors (e.g., use layer 16, last 32 tokens)
python scripts/compute_vectors.py \
  --model Qwen/Qwen3-0.6B \
  --layer 16 --last_n 32 --contrast plausibility --batch_size 4 \
  --data_dir data/ --out_dir artifacts/

# 4) Tune alpha with a validation split (ratio objective: accuracy / total_content_effect)
python scripts/search_alpha.py \
  --model Qwen/Qwen3-0.6B --layer 16 --last_n 32 \
  --vector_path artifacts/plausibility_layer16_last32.pt \
  --data_dir data/ --alpha_min -3.0 --alpha_max 3.0 --alpha_steps 25

# 5) Evaluate with/without steering
python scripts/evaluate.py --model Qwen/Qwen3-0.6B --data_dir data/
python scripts/evaluate.py --model Qwen/Qwen3-0.6B --data_dir data/ \
  --vector_path artifacts/plausibility_layer16_last32.pt --alpha 1.25 --layer 16 --last_n 32
```

### Notes

- Steering is applied **at the output of a chosen transformer block** and only on the final portion of the prompt (default: last 32 tokens), right **before** classifying `VALID` vs `INVALID`.
- The `compute_vectors.py` script builds a **contrastive direction** using the dataset labels (plausibility), averaged at the chosen layer/position.
- The `search_alpha.py` script performs a grid-search that **balances** accuracy against total content effect (configurable objective).
- The `evaluate.py` script now supports **dynamic α scheduling** (`--alpha-mode dynamic_margin`), which first measures baseline confidence (logit margin) per syllogism and then scales the steering strength within user-defined bounds. This yields stronger steering on ambiguous cases and lighter touch when the model is already confident.
- To reduce entanglement with unrelated global directions, `compute_vectors.py` supports `--mean-center`, which subtracts the corpus-wide activation mean before building the contrastive vector. Try this when a raw vector increases content effect more than it helps accuracy.

---

## Repository Layout

```
itsteer/
  data.py                # dataset & splits
  prompts.py             # robust prompt builder + candidate answers
  hooks.py               # low-level activation hooks
  vectorizers.py         # build steering vectors
  steering.py            # high-level steering context manager
  eval_metrics.py        # accuracy & content-effect metrics
  eval_runner.py         # evaluation orchestrator
  utils.py               # seed, device, logging
scripts/
  download_data.py       # get SemEval-2026 Task 11 data
  compute_vectors.py     # build and save steering vectors
  search_alpha.py        # tune alpha via validation objective
  evaluate.py            # run baseline and steered evaluation
configs/
  base.yaml              # default config (editable)
tests/
  test_shapes.py         # sanity-checks

## Experiment Tracking

Midterm and end-term steering runs are logged in `results/experiment_log.md`. Each entry records the exact commands, vector files, and metrics (accuracy plus content-effect breakdowns) for the evaluated models. Update that file whenever you run `scripts/search_alpha.py` or `scripts/evaluate.py` so future comparisons remain reproducible.

### Live dashboards with Weights & Biases

All CLI entry points (`compute_vectors.py`, `search_alpha.py`, `evaluate.py`) now have first-class [Weights & Biases](https://wandb.ai/) hooks so you can watch training curves, alpha sweeps, and evaluation breakdowns live:

1. Install & authenticate once per machine:
   ```bash
   pip install wandb
   wandb login
   ```
2. Pass `--wandb-mode online` (or `offline`) plus your project when launching any script. Examples:
   ```bash
   # Vector extraction with prompt previews + artifact upload
   python scripts/compute_vectors.py \
     --model Qwen/Qwen3-0.6B --data_dir data/ --layer 16 --last_n 32 \
     --contrast plausibility --out_dir artifacts/ \
     --wandb-mode online --wandb-project its-demo \
     --wandb-run-name vector-l16 --wandb-log-prompts 10 \
     --wandb-upload-vector

   # Alpha search sweep logged as a table/line chart
   python scripts/search_alpha.py \
     --model Qwen/Qwen3-0.6B --data_dir data/ \
     --vector_path artifacts/plausibility_layer16_last32.pt \
     --layer 16 --last_n 32 --alpha_min -2 --alpha_max 2 --alpha_steps 25 \
     --wandb-mode online --wandb-project its-demo --wandb-run-name alpha-scan

   # Final evaluation with confusion matrix + prediction preview
   python scripts/evaluate.py \
     --model Qwen/Qwen3-0.6B --data_dir data/ \
     --vector_path artifacts/plausibility_layer16_last32.pt \
     --layer 16 --alpha 1.25 --eval_on rest \
     --wandb-mode online --wandb-project its-demo \
     --wandb-run-name eval-rest --wandb-log-predictions 50
   ```

Shared CLI knobs:

- `--wandb-project / --wandb-entity / --wandb-run-name / --wandb-group / --wandb-tags / --wandb-notes`
- `--wandb-mode {online,offline,disabled}` controls whether logging is active (default: `disabled`).
- Script-specific helpers: `--wandb-log-prompts` & `--wandb-upload-vector` (vector extraction), `--wandb-log-table` (alpha sweep tables), `--wandb-log-predictions` (evaluation previews).
- When artifact flags are enabled, steering vectors and submission-ready predictions are versioned automatically inside the W&B UI for later download.
- Dynamic α options: `--alpha-mode dynamic_margin` together with `--dynamic-target-margin`, `--dynamic-min-alpha`, `--dynamic-max-alpha`, and `--dynamic-margin-eps` lets you customize how the steering strength adapts based on baseline confidence.
```

## Data

The scripts expect the SemEval-2026 Task 11 **training** format (English). Use:

```bash
python scripts/download_data.py --out data/
```

This pulls the pilot/training JSON(L) files from the **official task repository**.

## Citations / References

- IBM Activation Steering library — used as a reference design for vector extraction and conditional steering ideas.
- SemEval-2026 Task 11 — task description, metrics, and pilot data.
- Valentino et al. (2025) — fine-grained activation steering for content effects.

Please see the paper and repos linked in the top-level README of the project using this code.

# steps for normal running

1. go inside repo,  install dep.
2. python scripts/compute_vectors.py --model Qwen/Qwen3-0.6B --data_dir data/train.json --layer 14 --last_n 16 --contrast plausibility --batch_size 4 --max_length 512
3. python scripts/search_alpha.py --model Qwen/Qwen3-0.6B --data_dir data/train.json --vector_path artifacts/plausibility_layer14_last16.pt --layer 14 --last_n 16 --alpha_min 1 --alpha_max 2 --alpha_steps 2 --valid_frac 0.1
4. python scripts/evaluate.py --model Qwen/Qwen3-0.6B --data_dir data/train.json
5. python scripts/evaluate.py --model Qwen/Qwen3-0.6B --data_dir data/train.json --vector_path artifacts/plausibility_layer14_last16.pt --layer 14 --last_n 16 --alpha 1
