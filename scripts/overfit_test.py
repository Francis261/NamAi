import torch
import torch.nn as nn
from src.model.config import ModelConfig
from src.model.model import MambaAttentionLM


def overfit_test():
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
        swiglu_hidden_mult=2.75,
        sliding_window_size=64,
        num_global_tokens=8,
        max_seq_len=256,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = MambaAttentionLM(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    print("Starting forward pass...")
    model.train()
    initial_loss = None
    for step in range(100):
        optimizer.zero_grad()
        logits, loss = model(input_ids, labels)
        if initial_loss is None:
            initial_loss = loss.item()
            print(f"Initial loss: {initial_loss:.4f}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"  step {step+1:>3d}: loss = {loss.item():.6f}")

    final_loss = loss.item()
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Loss reduced: {initial_loss / final_loss:.1f}x")
    assert final_loss < 0.1, f"Overfit test FAILED: final loss {final_loss} >= 0.1"
    print("Overfit test PASSED")


if __name__ == "__main__":
    overfit_test()
