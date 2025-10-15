import argparse, os, torch, numpy as np
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
    args = ap.parse_args()
    v = torch.load(args.vector_path)["vector"]
    ds = SemevalDataset(args.data_dir)
    train, valid = ds.train_valid_split(valid_frac=args.valid_frac)

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    best = None
    for alpha in alphas:
        vec_cfg = {"layer": args.layer, "last_n": args.last_n, "v": v, "alpha": float(alpha)}
        metrics = evaluate_model(args.model, valid, vector_conf=vec_cfg)
        acc = metrics["accuracy"]
        ce = metrics["total_content_effect"]
        if args.objective == "ratio":
            score = acc / (ce + 1e-6)
        else:
            score = acc - ce
        print(f"alpha={alpha:.3f} -> acc={acc:.4f} totalCE={ce:.4f} score={score:.4f}")
        if best is None or score > best[0]:
            best = (score, alpha, metrics)
    print("Best:", best)

if __name__ == "__main__":
    main()
