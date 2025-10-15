import argparse, torch
from itsteer.data import SemevalDataset
from itsteer.eval_runner import evaluate_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--vector_path", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--last_n", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    ds = SemevalDataset(args.data_dir)
    vec_cfg = None
    if args.vector_path is not None:
        assert args.layer is not None and args.alpha is not None, "Need --layer and --alpha with --vector_path"
        v = torch.load(args.vector_path)["vector"]
        vec_cfg = {"layer": args.layer, "last_n": args.last_n, "v": v, "alpha": args.alpha}

    metrics = evaluate_model(args.model, ds.examples, vector_conf=vec_cfg)
    print(metrics)

if __name__ == "__main__":
    main()
