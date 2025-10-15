# scripts/download_data.py
import argparse, os, urllib.request

CANDIDATES = [
"https://github.com/neuro-symbolic-ai/semeval_2026_task_11/blob/main/train_data/train_data.json",

]

def try_fetch(url, out_path):
    try:
        print(f"Trying: {url}")
        with urllib.request.urlopen(url, timeout=30) as r, open(out_path, "wb") as f:
            f.write(r.read())
        print(f"Saved -> {out_path}")
        return True
    except Exception as e:
        print(f"  ...failed: {e}")
        return False

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", "--out_dir", dest="out_dir", default="data", help="Directory to save data into")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "train.jsonl")

    for url in CANDIDATES:
        if try_fetch(url, out_file):
            break
    else:
        raise SystemExit(
            "Could not download the training data from known paths.\n"

        )