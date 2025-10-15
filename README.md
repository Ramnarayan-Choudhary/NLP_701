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
