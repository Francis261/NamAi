import os
import sys
import math
import time
import signal
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from src.model.config import ModelConfig
from src.model.model import MambaAttentionLM
from src.training.data import create_dataloader
from src.training.data_mix import DataMixConfig, MultiSourceDataset
from src.training.optimizer import configure_optimizer, get_cosine_schedule
from src.model.save_load import save_checkpoint


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


def train_continuous(config: ModelConfig, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"Device: {device}")

    model = MambaAttentionLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params")

    wandb_run = None
    try:
        import wandb
        if args.wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or None,
                config={
                    "params": total_params,
                    **{k: getattr(config, k) for k in
                       ["d_model", "num_layers", "attn_every_n", "num_attention_heads",
                        "num_kv_heads", "d_head", "sliding_window_size", "num_global_tokens",
                        "d_state", "max_seq_len", "use_checkpointing"]},
                },
            )
    except ImportError:
        pass

    if args.data_mix:
        print(f"Using multi-source data mix: {args.data_mix}")
        mix_config = DataMixConfig.from_yaml(args.data_mix)
        dataset = MultiSourceDataset(mix_config)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=lambda batch: {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
            },
            num_workers=0,
            pin_memory=False,
        )
    else:
        dataloader = create_dataloader(
            batch_size=args.batch_size,
            max_seq_len=config.max_seq_len,
            dataset_name=args.dataset,
            split="train",
        )

    optimizer = configure_optimizer(
        model, weight_decay=args.weight_decay, lr=args.lr,
    )
    scheduler = get_cosine_schedule(
        optimizer, args.warmup_steps, args.max_steps, args.min_lr
    )

    scaler = GradScaler("cuda", enabled=(args.precision == "bf16" and device.type == "cuda"))

    step = 0
    tokens_seen = 0
    epoch = 0
    save_dir = args.save_dir

    if args.resume or (save_dir and not args.no_auto_resume):
        ckpt_path = args.resume
        ckpt_step = 0
        if not ckpt_path and save_dir:
            found = find_latest_checkpoint(save_dir)
            if found:
                ckpt_path, ckpt_step = found

        if ckpt_path:
            from src.model.save_load import load_checkpoint
            print(f"Resuming from {ckpt_path} (step {ckpt_step})")
            try:
                model, extra = load_checkpoint(ckpt_path, device=device)
                if "optimizer_state_dict" in extra:
                    optimizer.load_state_dict(extra["optimizer_state_dict"])
                if "step" in extra:
                    step = extra["step"]
                if "tokens_seen" in extra:
                    tokens_seen = extra["tokens_seen"]
                for _ in range(step):
                    scheduler.step()
                print(f"  Restored step={step} tokens={tokens_seen:,}")
            except Exception as e:
                print(f"  Resume failed: {e}, starting fresh")

    interrupt_requested = False
    def _handle_interrupt(sig, frame):
        nonlocal interrupt_requested
        if interrupt_requested:
            print("\nForced exit.")
            sys.exit(1)
        interrupt_requested = True
        print("\nSIGINT received — saving checkpoint and stopping...")
    signal.signal(signal.SIGINT, _handle_interrupt)

    log_accum = {"loss": 0.0, "count": 0, "tokens": 0}
    step_start = time.time()
    model.train()

    try:
        while not interrupt_requested:
            epoch += 1
            epoch_start = time.time()
            for batch in dataloader:
                if interrupt_requested:
                    break

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                with autocast("cuda", dtype=torch.bfloat16, enabled=(args.precision == "bf16" and device.type == "cuda")):
                    logits, loss = model(input_ids, labels)

                scaler.scale(loss).backward()
                tokens_this = input_ids.numel()

                if (step + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                tokens_seen += tokens_this
                log_accum["loss"] += loss.item()
                log_accum["count"] += 1
                log_accum["tokens"] += tokens_this

                if step % args.log_interval == 0 and log_accum["count"] > 0:
                    avg_loss = log_accum["loss"] / log_accum["count"]
                    lr_current = optimizer.param_groups[0]["lr"]
                    ppl = math.exp(min(avg_loss, 20))
                    elapsed = time.time() - step_start
                    tok_s = log_accum["tokens"] / max(1, elapsed)
                    print(
                        f"step={step:>6d} | loss={avg_loss:.4f} | ppl={ppl:.2f} | "
                        f"lr={lr_current:.2e} | tok={tokens_seen:>10,} | "
                        f"{tok_s:>6.0f} tok/s"
                    )
                    if wandb_run:
                        wandb_run.log({
                            "loss": avg_loss, "perplexity": ppl,
                            "lr": lr_current, "tokens": tokens_seen,
                            "step": step, "tok/s": tok_s,
                        })
                    log_accum = {"loss": 0.0, "count": 0, "tokens": 0}
                    step_start = time.time()

                if save_dir and step % args.save_interval == 0 and step > 0:
                    save_checkpoint(save_dir, f"step_{step}", model, config,
                        step=step, optimizer_state_dict=optimizer.state_dict(),
                        loss=loss.item(), tokens_seen=tokens_seen,
                    )

                step += 1

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch} complete: {epoch_time:.0f}s, {tokens_seen:,} tokens")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if save_dir and step > 0:
            save_checkpoint(save_dir, f"step_{step}", model, config,
                step=step, optimizer_state_dict=optimizer.state_dict(),
                loss=loss.item() if 'loss' in locals() else 0,
                tokens_seen=tokens_seen,
            )
            print(f"Saved emergency checkpoint: {save_dir}/step_{step}")

        if wandb_run:
            wandb_run.finish()

        print(f"Training stopped. Step={step} Tokens={tokens_seen:,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/350m.yaml")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--data_mix", type=str, default=None,
                        help="Path to data_mix.yaml for multi-source training")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_auto_resume", action="store_true",
                        help="Disable auto-resume from latest checkpoint")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mamba-attention-hybrid")
    parser.add_argument("--wandb_run", type=str, default=None)
    args = parser.parse_args()

    config = ModelConfig.from_yaml(args.config)
    train_continuous(config, args)
