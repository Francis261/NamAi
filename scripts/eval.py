import argparse
import os
import sys
import time
import torch
from src.model.save_load import load_checkpoint
from src.evaluation.perplexity import evaluate_perplexity
from src.evaluation.hellaswag import evaluate_hellaswag


def find_latest_checkpoint(save_dir: str):
    if not os.path.exists(save_dir):
        return None
    ckpts = [d for d in os.listdir(save_dir)
             if d.startswith("step_") and os.path.isdir(os.path.join(save_dir, d))]
    if not ckpts:
        return None
    steps = [int(c.replace("step_", "")) for c in ckpts]
    best = ckpts[steps.index(max(steps))]
    return os.path.join(save_dir, best), max(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--perplexity", action="store_true", help="Run WikiText perplexity")
    parser.add_argument("--hellaswag", action="store_true", help="Run HellaSwag accuracy")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    parser.add_argument("--max_samples", type=int, default=10, help="Samples per eval")
    parser.add_argument("--max_seq_len", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = args.checkpoint
    if ckpt_path and not os.path.isdir(ckpt_path):
        found = find_latest_checkpoint(ckpt_path)
        if found:
            ckpt_path = found[0]
    elif not ckpt_path:
        found = find_latest_checkpoint("checkpoints")
        if found:
            ckpt_path = found[0]

    if ckpt_path and os.path.isdir(ckpt_path):
        model, extra = load_checkpoint(ckpt_path, device=device)
        step = extra.get("step", "?")
        params = sum(p.numel() for p in model.parameters())
        print(f"Checkpoint: {ckpt_path} (step={step}, {params:,} params)")
    else:
        print("No checkpoint found. Use --checkpoint to specify one.")
        sys.exit(1)

    run_all = args.all
    results = {}

    if run_all or args.perplexity:
        print("\n--- WikiText-103 Perplexity ---")
        t0 = time.time()
        ppl = evaluate_perplexity(model, max_seq_len=args.max_seq_len, max_samples=args.max_samples)
        dt = time.time() - t0
        results["wikitext_ppl"] = ppl
        print(f"  Perplexity: {ppl:.2f}  ({dt:.0f}s)")

    if run_all or args.hellaswag:
        print("\n--- HellaSwag Accuracy ---")
        t0 = time.time()
        acc = evaluate_hellaswag(model, max_seq_len=args.max_seq_len, max_samples=args.max_samples)
        dt = time.time() - t0
        results["hellaswag_acc"] = acc
        print(f"  Accuracy: {acc:.4f}  ({dt:.0f}s)")

    print(f"\nResults: {results}")


if __name__ == "__main__":
    main()
