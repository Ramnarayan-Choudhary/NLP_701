import argparse, os, torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from itsteer.data import SemevalDataset
from itsteer.prompts import build_prompt
from itsteer.vectorizers import collect_hidden_states, build_contrastive_direction
from itsteer.utils import get_device
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--layer", type=int, required=True, help="Layer index to extract from (0-based)")
    ap.add_argument("--last_n", type=int, default=32, help="Average the last N tokens of the prompt")
    ap.add_argument("--contrast", choices=["plausibility", "validity"], default="plausibility")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if device=='cuda' else torch.float32, device_map='auto' if device=='cuda' else None)

    ds = SemevalDataset(args.data_dir)
    texts_pos, texts_neg = [], []
    for ex in ds.examples:
        prompt = build_prompt(ex.syllogism).text
        if args.contrast == "plausibility":
            if ex.plausibility is None:
                continue
            (texts_pos if ex.plausibility else texts_neg).append(prompt)
        else:
            if ex.validity is None:
                continue
            (texts_pos if ex.validity else texts_neg).append(prompt)

    pos_hs = collect_hidden_states(model, tok, texts_pos, layer_idx=args.layer, pos_strategy=f"last_n_mean:{args.last_n}", batch_size=args.batch_size, max_length=args.max_length)
    neg_hs = collect_hidden_states(model, tok, texts_neg, layer_idx=args.layer, pos_strategy=f"last_n_mean:{args.last_n}", batch_size=args.batch_size, max_length=args.max_length)

    v = build_contrastive_direction(pos_hs, neg_hs, normalize=True).cpu()
    out_path = os.path.join(args.out_dir, f"{args.contrast}_layer{args.layer}_last{args.last_n}.pt")
    torch.save({"vector": v, "meta": vars(args), "dims": int(v.numel())}, out_path)
    print(f"Saved vector -> {out_path} (dim={v.numel()})")

if __name__ == "__main__":
    main()
