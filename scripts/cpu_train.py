import torch
import torch.nn as nn
import torch.nn.functional as F
import time, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import ModelConfig
from src.model.model import MambaAttentionLM
from src.model.tokenizer import get_tokenizer

TRAIN_TEXT = """A Christmas Carol by Charles Dickens.

Marley was dead: to begin with. There is no doubt whatever about that. The register
of his burial was signed by the clergyman, the clerk, the undertaker, and the chief
mourner. Scrooge signed it: and Scrooge's name was good upon 'Change, for anything
he chose to put his hand to. Old Marley was as dead as a door-nail.

Mind! I don't mean to say that I know, of my own knowledge, what there is particularly
dead about a door-nail. I might have been inclined, myself, to regard a coffin-nail
as the deadest piece of ironmongery in the trade. But the wisdom of our ancestors is
in the simile; and my unhallowed hands shall not disturb it, or the Country's done
for. You will therefore permit me to repeat, emphatically, that Marley was as dead
as a door-nail.

Scrooge knew he was dead? Of course he did. How could it be otherwise? Scrooge and
he were partners for I don't know how many years. Scrooge was his sole executor, his
sole administrator, his sole assign, his sole residuary legatee, his sole friend, and
sole mourner. And even Scrooge was not so dreadfully cut up by the sad event, but
that he was an excellent man of business on the very day of the funeral, and
solemnised it with an undoubted bargain.

The mention of Marley's funeral brings me back to the point I started from. There is
no doubt that Marley was dead. This must be distinctly understood, or nothing
wonderful can come of the story I am going to relate."""


def main():
    config = ModelConfig(
        vocab_size=50257,
        d_model=128,
        d_state=4,
        d_conv=4,
        expand_factor=2,
        num_layers=4,
        attn_every_n=2,
        num_attention_heads=4,
        num_kv_heads=2,
        d_head=32,
        swiglu_hidden_mult=2.75,
        sliding_window_size=64,
        num_global_tokens=4,
        max_seq_len=64,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaAttentionLM(config).to(device)
    tokenizer = get_tokenizer()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params | Device: {device}", flush=True)

    tokens = tokenizer.encode(TRAIN_TEXT)
    print(f"Training tokens: {len(tokens):,}", flush=True)
    seq_len = config.max_seq_len
    stride = seq_len

    chunks = []
    for i in range(0, len(tokens) - seq_len, stride):
        chunk = torch.tensor(tokens[i:i + seq_len + 1], dtype=torch.long)
        chunks.append((chunk[:-1], chunk[1:]))

    if not chunks:
        tokens_padded = tokens + [tokenizer.eos_token_id] * (seq_len + 1 - len(tokens))
        chunk = torch.tensor(tokens_padded[:seq_len + 1], dtype=torch.long)
        chunks.append((chunk[:-1], chunk[1:]))

    print(f"Created {len(chunks)} training sequences", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    steps = 80
    log_every = 20

    @torch.no_grad()
    def generate(prompt: str, max_new: int = 60):
        model.eval()
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new):
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / 0.9
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
        return tokenizer.decode(input_ids[0].tolist())

    model.train()
    start = time.time()
    for step in range(1, steps + 1):
        total_loss = 0.0
        for input_ids, labels in chunks:
            input_ids = input_ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            optimizer.zero_grad()
            logits, loss = model(input_ids, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(chunks)

        if step % log_every == 0 or step == 1:
            ppl = math.exp(avg_loss)
            elapsed = time.time() - start
            print(f"  step {step:>4d} | loss {avg_loss:.4f} | ppl {ppl:.2f} | "
                  f"{elapsed:.0f}s elapsed", flush=True)

    elapsed = time.time() - start
    print(f"\nTraining complete: {elapsed:.1f}s", flush=True)

    print(f"\nAfter training — prompt: 'Scrooge was'")
    print(f"  {generate('Scrooge was')}", flush=True)

    @torch.no_grad()
    def sample_text(seed: str, length: int = 100):
        model.eval()
        ids = tokenizer.encode(seed)
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(length):
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            top_k = 20
            values, _ = torch.topk(probs, top_k)
            probs[probs < values[:, -1:]] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
            if input_ids.shape[1] > config.max_seq_len:
                input_ids = input_ids[:, -config.max_seq_len:]
        return tokenizer.decode(input_ids[0].tolist())

    print(f"\n--- Generated text (seed: 'Marley was') ---")
    print(sample_text("Marley was"))
    print("---", flush=True)


if __name__ == "__main__":
    main()
