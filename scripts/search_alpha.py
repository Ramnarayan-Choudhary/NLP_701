import argparse
import torch
import numpy as np
from itsteer.data import SemevalDataset
from itsteer.eval_runner import evaluate_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--vector_path", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--last_n", type=int, default=32)
    ap.add_argument("--alpha_min", type=float, default=-2.0)
    ap.add_argument("--alpha_max", type=float, default=2.0)
    ap.add_argument("--alpha_steps", type=int, default=21)
    ap.add_argument("--valid_frac", type=float, default=0.2)
    ap.add_argument("--objective", choices=["ratio", "acc_minus_ce"], default="ratio",
                    help="ratio: accuracy / total_content_effect; acc_minus_ce: accuracy - total_content_effect")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Safe vector load; fall back if torch version doesn't support weights_only
    try:
        v = torch.load(args.vector_path, weights_only=True)["vector"]
    except TypeError:
        v = torch.load(args.vector_path)["vector"]

    ds = SemevalDataset(args.data_dir)
    train, valid = ds.train_valid_split(valid_frac=args.valid_frac, seed=args.seed)

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    best = None  # tuple: (score, alpha_float, metrics_dict)

    for alpha in alphas:
        a = float(alpha)
        vec_cfg = {"layer": args.layer, "last_n": args.last_n, "v": v, "alpha": a}
        metrics = evaluate_model(args.model, valid, vector_conf=vec_cfg)
        acc = metrics["accuracy"]
        ce  = metrics["total_content_effect"]
        score = (acc / (ce + 1e-6)) if args.objective == "ratio" else (acc - ce)
        print(f"alpha={a:.3f} -> acc={acc:.4f} totalCE={ce:.4f} score={score:.4f}")
        if best is None or score > best[0]:
            best = (score, a, metrics)

    # Human-friendly and machine-friendly lines
    print("Best:", best)
    # This exact line is parsed by your shell to capture alpha:
    print("BEST_ALPHA", best[1])

if __name__ == "__main__":
    main()


