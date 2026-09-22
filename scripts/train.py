"""在合成伪迹混合数据上训练完整六专家 UniCORE-EEG。

T0.7 的三处改动（手册 §8 T0.7）：

* ``--num-workers`` 默认值由 ``0`` 改为 ``8``，并补 ``persistent_workers`` /
  ``prefetch_factor``（§0.3.7 实测 ``num_workers=0`` 会让 GPU 空转 23%–35%）；
* ``loss.backward()`` → ``loss.sum().backward()``，否则 ``--data-parallel``
  在跨卡聚合后必然报 ``grad can be implicitly created only for scalar outputs``；
* 新增 ``--config``，按"命令行 > 配置文件 > 代码默认值"合成设置（§5.4），
  并写出 ``run_manifest.json``（§5.3）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unicore_eeg import UniCOREEG, UniCOREEGConfig, paths
from unicore_eeg.config import resolve_settings
from unicore_eeg.losses import UniCORELoss
from unicore_eeg.manifest import write_run_manifest
from unicore_eeg.model import count_parameters
from unicore_eeg.runtime import configure_runtime, dataloader_kwargs
from unicore_eeg.synthetic import SyntheticEEGDataset


#: 代码内默认值（优先级最低）。num_workers 由 0 改为 8 是 T0.7 的唯一行为变更。
DEFAULTS: dict[str, Any] = {
    "epochs": 10,
    "samples": 8192,
    "batch_size": 64,
    "channels": 1,
    "length": 1000,
    "sample_rate": 500,
    "lr": 2e-4,
    "num_workers": 8,
    "compile": False,
    "data_parallel": False,
    "use_public_sources": False,
    "data_root": paths.RAW_ROOT,
    "out": paths.RUNS_ROOT / "unicore_synth",
}

CONFIG_MAP = {
    "sample_rate": "signal.sample_rate",
    "length": "signal.window_size",
    "batch_size": "dataloader.batch_size",
    "num_workers": "dataloader.num_workers",
}


def parse_args() -> tuple[argparse.Namespace, dict[str, Any] | None]:
    parser = argparse.ArgumentParser(description="Train UniCORE-EEG on synthetic artifact mixtures.")
    parser.add_argument("--config", type=Path, help="实验配置（YAML）；命令行显式参数优先级更高")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--channels", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sample-rate", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--compile", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--data-parallel", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--use-public-sources", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--data-root", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=argparse.SUPPRESS)
    parsed = parser.parse_args()
    settings, config = resolve_settings(parsed, DEFAULTS, parsed.config, CONFIG_MAP)
    return argparse.Namespace(**settings), config


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def main() -> None:
    args, run_config = parse_args()
    args.out = Path(args.out)
    args.data_root = Path(args.data_root)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 手册 §0.3.4/§5.5：本机无 Triton，请求 --compile 会直接抛错而不是静默降级。
    runtime = configure_runtime(run_config, enable_compile=args.compile, device=device)

    config = UniCOREEGConfig(
        in_channels=args.channels,
        sample_rate=args.sample_rate,
        window_size=args.length,
    )
    model: torch.nn.Module = UniCOREEG(config).to(device)
    if runtime["torch_compile"]:
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
    loader_kwargs = dataloader_kwargs(args.batch_size, args.num_workers, device)
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    loss_fn = UniCORELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_val = float("inf")

    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    print(f"device={device} gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"params={count_parameters(raw_model):,} num_workers={args.num_workers}")

    write_run_manifest(
        args.out,
        command=sys.argv,
        settings=vars(args),
        config=run_config,
        dataset={
            "name": "synthetic",
            "samples": args.samples,
            "train_samples": train_len,
            "val_samples": val_len,
            "channels": args.channels,
            "channel_list": None,
            "window_size": args.length,
            "sample_rate": args.sample_rate,
            "use_public_sources": args.use_public_sources,
            "data_root": str(args.data_root),
            "split_generator_seed": 123,
        },
        extra={"runtime": runtime, "loss_weights": dict(loss_fn.weights.__dict__)},
    )

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
            # T0.7：DataParallel 下损失是"长度 = 卡数"的向量，必须 .sum()。
            # 单卡时它是 0 维张量，.sum() 等价，不改变任何数值。
            scaler.scale(loss).sum().backward()
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
        print(f"epoch={epoch} val_loss={val_loss:.4f}", flush=True)

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

    print(f"done: best_val={best_val:.4f} out={args.out}")


if __name__ == "__main__":
    main()
