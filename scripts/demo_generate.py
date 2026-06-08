import torch
from src.model.config import ModelConfig
from src.model.model import MambaAttentionLM
from src.model.tokenizer import get_tokenizer


def generate(model, tokenizer, prompt: str, max_new: int = 100,
             temperature: float = 0.8, top_k: int = 40):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new):
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

    return tokenizer.decode(input_ids[0].tolist())


def main():
    config = ModelConfig(
        vocab_size=50257,
        d_model=256,
        d_state=4,
        d_conv=4,
        expand_factor=2,
        num_layers=4,
        attn_every_n=2,
        num_attention_heads=4,
        num_kv_heads=2,
        d_head=64,
        sliding_window_size=128,
        num_global_tokens=8,
        max_seq_len=256,
    )

    model = MambaAttentionLM(config)
    tokenizer = get_tokenizer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {model.device}")

    prompts = [
        "The future of AI is",
        "Once upon a time",
        "The key to building intelligent systems is",
    ]

    for prompt in prompts:
        output = generate(model, tokenizer, prompt, max_new=50, temperature=1.0, top_k=40)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")


if __name__ == "__main__":
    main()
