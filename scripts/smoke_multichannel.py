"""多通道冒烟：对一组通道数各跑一次前向 + 损失 + 反向（手册 T1.5 步骤 1）。

对每个 ``C``：构造一个真实 batch（走 :class:`~unicore_eeg.synthetic.SyntheticEEGDataset`），
跑 ``forward → UniCORELoss → backward`` 若干步，记录参数量、显存峰值与单步耗时。

为什么用真实 batch 而不是随机张量：通道数会一路影响观测抽取（FFT/自相关）、
专家卷积、解码器通道数与空间模块，随机张量绕不过这些真实分支。

显存注意：实测 ``C=128 且 batch=128`` 必 OOM（见 `runs/_env/NOTES.md`），
所以默认 batch 取 8，要用大 batch 请显式指定并自行承担风险。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unicore_eeg import UniCOREEG, UniCOREEGConfig, paths  # noqa: E402
from unicore_eeg.losses import UniCORELoss  # noqa: E402
from unicore_eeg.model import count_parameters  # noqa: E402
from unicore_eeg.montage import load_montage  # noqa: E402
from unicore_eeg.runtime import configure_runtime  # noqa: E402
from unicore_eeg.synthetic import SyntheticEEGDataset  # noqa: E402

DEFAULT_CHANNELS = (1, 3, 8, 34, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-channel smoke test (T1.5)")
    parser.add_argument("--channels", type=int, nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--sample-rate", type=int, default=500)
    parser.add_argument("--steps", type=int, default=3, help="每个 C 跑几步，取耗时中位数")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-public-sources", action="store_true")
    parser.add_argument(
        "--montage",
        default=None,
        help="可选：用某个 montage 的真实坐标喂给合成器（缺省用确定性半球布局）",
    )
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=paths.RUNS_ROOT / "smoke_multichannel" / "summary.json")
    return parser.parse_args()


def build_batch(args: argparse.Namespace, channels: int, device: torch.device) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.montage:
        spec = load_montage(args.montage)
        if spec.channel_count != channels:
            raise SystemExit(
                f"--montage {args.montage} 有 {spec.channel_count} 个通道，与 C={channels} 不一致"
            )
        kwargs["montage_coords"] = spec.coords
    dataset = SyntheticEEGDataset(
        samples=max(args.batch_size, 16),
        channels=channels,
        length=args.length,
        sample_rate=args.sample_rate,
        seed=args.seed,
        use_public_sources=args.use_public_sources,
        split="train",
        **kwargs,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_one(
    args: argparse.Namespace, channels: int, device: torch.device, precision: str
) -> dict[str, Any]:
    record: dict[str, Any] = {"channels": channels, "batch_size": args.batch_size}
    try:
        batch = build_batch(args, channels, device)
        config = UniCOREEGConfig(
            in_channels=channels, sample_rate=args.sample_rate, window_size=args.length
        )
        model = UniCOREEG(config).to(device)
        criterion = UniCORELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # bf16 只在 CUDA 上开：本机 CPU 的 FFT 不支持 bf16（模型内部已统一 float32，
        # 但保持"CPU 一律 fp32"更简单也更可复现）
        use_bf16 = precision == "bf16" and device.type == "cuda"
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else None

        parameters = count_parameters(model)
        record["parameters"] = parameters
        record["metadata_dim"] = int(batch["metadata"].size(-1))

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings: list[float] = []
        losses: list[float] = []
        for _ in range(max(args.steps, 1)):
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            if autocast is None:
                outputs = model(batch["noisy"])
            else:
                with autocast:
                    outputs = model(batch["noisy"])
            loss, logs = criterion(outputs, batch)
            loss.sum().backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - started)
            losses.append(float(logs["loss"]))
        record.update(
            {
                "status": "ok",
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "ms_per_step_median": sorted(timings)[len(timings) // 2] * 1000.0,
                "peak_memory_gib": (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else None
                ),
            }
        )
        del model, criterion, optimizer, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except BaseException as error:  # noqa: BLE001 - 冒烟要把失败如实记下来而不是中断整轮
        if "out of memory" in str(error).lower() or "OutOfMemoryError" in type(error).__name__:
            record.update({"status": "OOM", "error": str(error).splitlines()[0][:200]})
        else:
            record.update(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": __import__("traceback").format_exc(limit=4),
                }
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return record


def run_shared_checkpoint(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    """Use one model/state_dict for every C, without optimizer updates between inputs."""
    model = UniCOREEG(
        UniCOREEGConfig(in_channels=1, sample_rate=args.sample_rate, window_size=args.length)
    ).to(device).eval()
    parameter_count = count_parameters(model)
    records: list[dict[str, Any]] = []
    for channels in args.channels:
        batch = build_batch(args, channels, device)
        with torch.no_grad():
            output = model(
                batch["noisy"],
                metadata=batch["metadata"],
                coords=batch.get("coords"),
                channel_mask=torch.ones(batch["noisy"].shape[:2], dtype=torch.bool, device=device),
            )
        records.append({
            "channels": channels,
            "status": "ok" if tuple(output["clean"].shape) == tuple(batch["noisy"].shape) else "shape_error",
            "output_shape": list(output["clean"].shape),
            "parameters": parameter_count,
        })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    # configure_runtime 期望拿到**整份配置**（它自己会取 `runtime` 段）；
    # 直接传扁平的 runtime 字典会被静默忽略并退回默认值（踩过一次）。
    active = configure_runtime(
        {
            "runtime": {
                "precision": args.precision,
                "torch_compile": False,
                "cudnn_benchmark": device.type == "cuda",
                "grad_scaler": False,
                "matmul_precision": "high",
            }
        },
        device=device,
    )
    if device.type == "cpu" and args.precision == "bf16":
        print("提示：CPU 上 bf16 走的是 float32 路径（本机 CPU 的 FFT 不支持 bf16）")

    results: list[dict[str, Any]] = []
    shared_results = run_shared_checkpoint(args, device)
    print("shared checkpoint:", ", ".join(f"C={item['channels']}:{item['status']}" for item in shared_results))
    print(
        f"device={device} precision={active['precision']} batch={args.batch_size} "
        f"length={args.length} montage={args.montage}"
    )
    print(f"{'C':>5s} {'params':>12s} {'status':>7s} {'ms/step':>9s} {'peakGiB':>8s}  note")
    for channels in args.channels:
        record = run_one(args, channels, device, active["precision"])
        results.append(record)
        peak = record.get("peak_memory_gib")
        step = record.get("ms_per_step_median")
        print(
            f"{channels:5d} {record.get('parameters', 0):12,d} {record['status']:>7s} "
            f"{(f'{step:.1f}' if step else '-'):>9s} {(f'{peak:.2f}' if peak else '-'):>8s}  "
            f"{record.get('error', '')}"
        )

    failures = [item for item in results if item["status"] != "ok"]
    payload = {
        "device": str(device),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "length": args.length,
        "steps": args.steps,
        "montage": args.montage,
        "results": results,
        "shared_checkpoint": shared_results,
        "all_ok": not failures,
        "shared_checkpoint_all_ok": all(item["status"] == "ok" for item in shared_results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shown = args.out if args.out.is_absolute() else paths.PROJECT_ROOT / args.out
    shown = shown.relative_to(paths.PROJECT_ROOT) if shown.is_relative_to(paths.PROJECT_ROOT) else shown
    print(f"\n{len(results) - len(failures)}/{len(results)} 个通道数通过；汇总写入 {shown}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
