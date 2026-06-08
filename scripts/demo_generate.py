import os
import torch
import argparse
from src.model.config import ModelConfig
from src.model.model import MambaAttentionLM
from src.model.tokenizer import get_tokenizer
from src.model.save_load import load_checkpoint


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


def generate(model, tokenizer, prompt: str, max_new: int = 100,
             temperature: float = 0.8, top_k: int = 40, device=None):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device or model.device).unsqueeze(0)

    with torch.no_grad():
        for i in range(max_new):
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < values[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{max_new}] tokens generated...", flush=True)

    return tokenizer.decode(input_ids[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint directory (or auto-find latest)")
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.isdir(ckpt_path):
            found = find_latest_checkpoint(ckpt_path)
            if found:
                ckpt_path = found[0]
                print(f"Found latest checkpoint: {ckpt_path}")
            else:
                print(f"Checkpoint not found: {ckpt_path}")
                return
        model, extra = load_checkpoint(ckpt_path, device=device)
        print(f"Loaded checkpoint, step={extra.get('step', '?')}")
    else:
        config = ModelConfig(
            vocab_size=50257, d_model=128, d_state=4, d_conv=4,
            expand_factor=2, num_layers=6, attn_every_n=3,
            num_attention_heads=4, num_kv_heads=2, d_head=32,
            sliding_window_size=256, num_global_tokens=8,
            max_seq_len=256,
        )
        model = MambaAttentionLM(config).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"\nPrompt: {args.prompt}")
    print("Generating...", flush=True)
    output = generate(model, tokenizer, args.prompt,
                      max_new=args.max_new_tokens,
                      temperature=args.temperature,
                      top_k=args.top_k, device=device)
    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
