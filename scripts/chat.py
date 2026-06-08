import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.save_load import load_model_from_dir
from src.model.tokenizer import get_tokenizer
from io import StringIO


def generate_stream(model, tokenizer, prompt: str, max_new: int = 64,
                    temperature: float = 0.7, top_k: int = 40,
                    max_context: int = 1024, timeout: float = 120):
    model.eval()
    ids = tokenizer.encode(prompt)
    if len(ids) > max_context:
        ids = ids[-max_context:]

    input_ids = torch.tensor(ids, dtype=torch.long, device=model.device).unsqueeze(0)
    start = time.time()

    for _ in range(max_new):
        if time.time() - start > timeout:
            break

        with torch.no_grad():
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature

            if top_k:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, -1:]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        next_token = next_id.item()
        input_ids = torch.cat([input_ids, next_id], dim=1)

        yield next_token

        if next_token == tokenizer.eos_token_id:
            break

        if input_ids.shape[1] > max_context:
            input_ids = input_ids[:, -max_context:]


def find_best_checkpoint(base_dir: str):
    if not os.path.exists(base_dir):
        return None
    step_dirs = []
    for d in os.listdir(base_dir):
        full = os.path.join(base_dir, d)
        if d.startswith("step_") and os.path.isdir(full):
            try:
                step = int(d.replace("step_", ""))
                step_dirs.append((step, full))
            except ValueError:
                pass
        if d == "final" and os.path.isdir(full):
            step_dirs.append((float("inf"), full))
    if not step_dirs:
        return None
    step_dirs.sort(key=lambda x: -x[0])
    return step_dirs[0][1]


def simple_demo(model, tokenizer, prompt: str, max_new: int = 20):
    print(f"\n=== Demo: '{prompt}' ===\n", flush=True)
    print(prompt, end="", flush=True)
    tokens = []
    for token_id in generate_stream(model, tokenizer, prompt,
                                     max_new=max_new, temperature=0.8, top_k=20):
        text = tokenizer.decode([token_id])
        tokens.append(token_id)
        print(text, end="", flush=True)
    print(f"\n", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--max_new", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    ckpt_path = args.checkpoint or find_best_checkpoint(args.checkpoint_dir)
    if not ckpt_path:
        print(f"No checkpoint found in '{args.checkpoint_dir}'")
        sys.exit(1)

    print(f"Loading model from {ckpt_path}...", flush=True)
    model = load_model_from_dir(ckpt_path, device="cpu")
    tokenizer = get_tokenizer()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)\n", flush=True)

    simple_demo(model, tokenizer, "Artificial intelligence will", 15)

    print("=" * 58)
    print("Interactive chat (slow on CPU, ~3-10s per token)")
    print("Commands: /temp N  /reset  /max N  /quit")
    print("=" * 58)

    conversation = ""
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input.startswith("/temp"):
            try:
                args.temperature = float(user_input.split()[1])
                print(f"  temp = {args.temperature}")
            except (IndexError, ValueError):
                print("  usage: /temp 0.7")
            continue
        if user_input == "/reset":
            conversation = ""
            print("  reset")
            continue
        if user_input.startswith("/max"):
            try:
                args.max_new = int(user_input.split()[1])
                print(f"  max_new = {args.max_new}")
            except (IndexError, ValueError):
                print("  usage: /max 100")
            continue

        prompt = conversation + f"User: {user_input}\nAssistant:"
        print("Assistant: ", end="", flush=True)

        token_ids = []
        start_time = time.time()
        for token_id in generate_stream(model, tokenizer, prompt,
                                         max_new=args.max_new,
                                         temperature=args.temperature,
                                         top_k=args.top_k,
                                         timeout=args.timeout):
            text = tokenizer.decode([token_id])
            print(text, end="", flush=True)
            token_ids.append(token_id)

        elapsed = time.time() - start_time
        if token_ids:
            print(f"\n  [{len(token_ids)} tokens in {elapsed:.0f}s, {elapsed/len(token_ids):.1f}s/tok]")
            conversation = prompt + tokenizer.decode(token_ids) + "\n"
        else:
            print("[no output generated]")

        if len(conversation) > 8000:
            conversation = conversation[-4000:]


if __name__ == "__main__":
    main()
