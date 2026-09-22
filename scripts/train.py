from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unicore_eeg import UniCOREEG, UniCOREEGConfig
from unicore_eeg.losses import UniCORELoss
from unicore_eeg.model import count_parameters
from unicore_eeg.synthetic import SyntheticEEGDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UniCORE-EEG on synthetic artifact mixtures.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--sample-rate", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--use-public-sources", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("runs/unicore_synth"))
    return parser.parse_args()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    config = UniCOREEGConfig(
        in_channels=args.channels,
        sample_rate=args.sample_rate,
        window_size=args.length,
    )
    model: torch.nn.Module = UniCOREEG(config).to(device)
    if args.compile:
        model = torch.compile(model)
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    dataset = SyntheticEEGDataset(
        samples=args.samples,
        channels=args.channels,
        length=args.length,
        sample_rate=args.sample_rate,
        use_public_sources=args.use_public_sources,
        data_root=args.data_root,
    )
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(
        dataset,
        (train_len, val_len),
        generator=torch.Generator().manual_seed(123),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    loss_fn = UniCORELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_val = float("inf")

    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    print(f"device={device} gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"params={count_parameters(raw_model):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_logs: dict[str, float] = {}
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} train")
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(
                    batch["noisy"],
                    metadata=batch["metadata"],
                    disabled_experts=batch["disabled_experts"],
                )
                loss, logs = loss_fn(outputs, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in logs.items():
                train_logs[key] = train_logs.get(key, 0.0) + float(value)
            progress.set_postfix(loss=f"{float(logs['loss']):.3f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch}/{args.epochs} val"):
                batch = move_batch(batch, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outputs = model(
                        batch["noisy"],
                        metadata=batch["metadata"],
                        disabled_experts=batch["disabled_experts"],
                    )
                    _, logs = loss_fn(outputs, batch)
                val_loss += float(logs["loss"])
        val_loss /= max(len(val_loader), 1)
        print(f"epoch={epoch} val_loss={val_loss:.4f}")

        checkpoint = {
            "config": config.__dict__,
            "model": raw_model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, args.out / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, args.out / "best.pt")


if __name__ == "__main__":
    main()
