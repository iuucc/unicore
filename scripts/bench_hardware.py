"""硬件与数据加载基线（手册 T0.7 步骤 2）。

产出 ``runs/_env/hardware_baseline.json``，含两组数据：

1. **GPU 吞吐**：``C ∈ {1, 8, 34, 64, 128}`` × ``batch ∈ {32, 64, 128}``，
   用真实模型 + 真实损失做 bf16 前向+反向，记录 ``ms/step``、``samples/s``、
   PyTorch 分配器峰值。**OOM 的组合记为 ``"OOM"`` 而不是中断整轮测量。**
2. **DataLoader 吞吐**：``num_workers ∈ {0, 4, 8, 12}``，记录端到端与稳态
   ``samples/s`` 以及测量后的剩余物理内存。

两组都**每组合至少 2000 个样本**（``--min-samples``）——手册 §0.3.7/§0.3.8 的既有数值
来自 8 step / 384 样本的短测，只能定方向，不能定参数。

Windows 注意事项（手册 §0.3.7 两个坑）：入口有 ``if __name__ == "__main__":`` 守卫；
结束时统计残留子进程，避免 worker 泄漏（实测一次泄漏 16 进程 ≈ 16.5 GB）。
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from unicore_eeg import paths
from unicore_eeg.losses import UniCORELoss
from unicore_eeg.model import ARTIFACT_NAMES, UniCOREEG, UniCOREEGConfig, count_parameters


DEFAULT_CHANNELS = (1, 8, 34, 64, 128)
DEFAULT_BATCHES = (32, 64, 128)
DEFAULT_WORKERS = (0, 4, 8, 12)
#: DataLoader 段落要覆盖的通道数。1 = 当前主线；34 = 手册 §0.3.7 短测的口径，便于对照。
DEFAULT_DATALOADER_CHANNELS = (1, 34)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the local hardware / dataloader baseline (T0.7).")
    parser.add_argument("--out", type=Path, default=paths.RUNS_ROOT / "_env" / "hardware_baseline.json")
    parser.add_argument("--channels", type=int, nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--workers", type=int, nargs="+", default=list(DEFAULT_WORKERS))
    parser.add_argument(
        "--dataloader-channels",
        type=int,
        nargs="+",
        default=list(DEFAULT_DATALOADER_CHANNELS),
        help="DataLoader 段落要覆盖的通道数；必须含 1（主线单通道）与 34（可与 §0.3.7 短测对照）",
    )
    parser.add_argument("--min-samples", type=int, default=2000, help="每组合至少测量多少个样本")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--sample-rate", type=int, default=500)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--use-public-sources", action="store_true", default=True)
    parser.add_argument("--no-public-sources", dest="use_public_sources", action="store_false")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-dataloader", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# --------------------------------------------------------------------------
# 环境信息与内存
# --------------------------------------------------------------------------


def git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def memory_snapshot() -> dict[str, float]:
    """剩余/总物理内存（GiB）。psutil 缺失时返回空字典而不是中断测量。"""
    try:
        import psutil
    except ImportError:
        return {}
    info = psutil.virtual_memory()
    return {
        "total_gib": round(info.total / 1024**3, 2),
        "available_gib": round(info.available / 1024**3, 2),
        "used_percent": float(info.percent),
    }


def leftover_worker_processes() -> list[dict[str, Any]]:
    """列出资进程残留的 python 子进程（手册 §0.3.7 坑 2）。"""
    try:
        import psutil
    except ImportError:
        return []
    me = psutil.Process()
    leftovers = []
    for child in me.children(recursive=True):
        try:
            leftovers.append(
                {
                    "pid": child.pid,
                    "rss_mib": round(child.memory_info().rss / 1024**2, 1),
                    "cmdline": " ".join(child.cmdline()[:4]),
                }
            )
        except psutil.Error:
            continue
    return leftovers


# --------------------------------------------------------------------------
# GPU 吞吐
# --------------------------------------------------------------------------


def is_oom(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def measure_gpu_combo(
    channels: int,
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    record: dict[str, Any] = {"channels": channels, "batch_size": batch_size}
    oom_error = None
    # OOM 会污染显存状态，先清一次并把峰值统计复位。
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        config = UniCOREEGConfig(
            in_channels=channels,
            sample_rate=args.sample_rate,
            window_size=args.window_size,
        )
        model = UniCOREEG(config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = UniCORELoss()
        record["params"] = count_parameters(model)

        def build_batch() -> dict[str, torch.Tensor]:
            # 生成器必须与张量同设备：CPU 生成器配 device="cuda" 会抛
            # RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'。
            # 用 CUDA 生成器而不是"CPU 生成后搬运"，否则 H2D 拷贝会计入单步耗时。
            generator = torch.Generator(device=device).manual_seed(args.seed)
            expert_count = len(ARTIFACT_NAMES)
            noisy = torch.randn(batch_size, channels, args.window_size, generator=generator, device=device)
            clean = torch.randn(batch_size, channels, args.window_size, generator=generator, device=device)
            artifacts = torch.randn(
                batch_size, expert_count, channels, args.window_size, generator=generator, device=device
            )
            return {
                "noisy": noisy,
                "clean": clean,
                "artifacts": artifacts,
                "labels": torch.zeros(batch_size, expert_count, device=device),
                "label_mask": torch.ones(batch_size, expert_count, device=device),
                "component_mask": torch.ones(batch_size, expert_count, device=device),
                "severity": torch.zeros(batch_size, expert_count, device=device),
                "is_clean": torch.ones(batch_size, device=device),
                "disabled_experts": torch.zeros(batch_size, expert_count, dtype=torch.bool, device=device),
                "metadata": torch.zeros(batch_size, 8, device=device),
            }

        def step(batch: dict[str, torch.Tensor]) -> None:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(
                    batch["noisy"],
                    metadata=batch["metadata"],
                    disabled_experts=batch["disabled_experts"],
                )
                loss, _ = loss_fn(outputs, batch)
            loss.sum().backward()
            optimizer.step()

        # 预热：让 cudnn benchmark / 分配器缓存先稳定下来，不进入统计。
        warmup_batch = build_batch()
        for _ in range(max(args.warmup_steps, 0)):
            step(warmup_batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        steps = max(math.ceil(args.min_samples / batch_size), 3)
        batch = build_batch()
        started = time.perf_counter()
        for _ in range(steps):
            step(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        samples = steps * batch_size
        record.update(
            {
                "status": "ok",
                "steps": steps,
                "samples": samples,
                "elapsed_s": round(elapsed, 3),
                "ms_per_step": round(elapsed / steps * 1000.0, 2),
                "samples_per_s": round(samples / elapsed, 2),
                "allocator_peak_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
                if device.type == "cuda"
                else None,
                "allocator_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 1024**3, 3)
                if device.type == "cuda"
                else None,
            }
        )
        del model, optimizer, batch, warmup_batch
    except BaseException as error:  # noqa: BLE001 - OOM 必须被记录而不是中断整轮
        if is_oom(error):
            record["status"] = "OOM"
            record["error"] = str(error).splitlines()[0][:200]
        else:
            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc(limit=3)
    finally:
        # 无论成功、OOM 还是其它异常，都要释放显存，否则后续组合连续 OOM。
        torch.cuda.empty_cache()
    return record


def measure_gpu(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    results = []
    for channels in args.channels:
        for batch_size in args.batches:
            print(f"[gpu] C={channels:<4d} batch={batch_size:<4d} ...", end="", flush=True)
            record = measure_gpu_combo(channels, batch_size, args, device)
            results.append(record)
            if record["status"] == "ok":
                print(
                    f" {record['samples_per_s']:>8.1f} samples/s  "
                    f"{record['ms_per_step']:>8.2f} ms/step  "
                    f"peak={record['allocator_peak_gib']:.2f} GiB"
                )
            else:
                print(f" {record['status']}: {record.get('error', '')}")
    return {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "device_count": torch.cuda.device_count(),
        "total_memory_gib": round(
            torch.cuda.get_device_properties(device).total_memory / 1024**3, 2
        )
        if device.type == "cuda"
        else None,
        "precision": "bf16",
        "min_samples_per_combo": args.min_samples,
        "warmup_steps": args.warmup_steps,
        "combos": results,
    }


# --------------------------------------------------------------------------
# DataLoader 吞吐
# --------------------------------------------------------------------------


def measure_dataloader_combo(
    num_workers: int,
    channels: int,
    batch_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    from unicore_eeg.synthetic import SyntheticEEGDataset

    record: dict[str, Any] = {
        "num_workers": num_workers,
        "channels": channels,
        "batch_size": batch_size,
    }
    batches = max(math.ceil(args.min_samples / batch_size), 3)
    dataset = SyntheticEEGDataset(
        samples=batches * batch_size,
        channels=channels,
        length=args.window_size,
        sample_rate=args.sample_rate,
        seed=args.seed,
        use_public_sources=args.use_public_sources,
        data_root=paths.RAW_ROOT,
        split="train",
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,  # 只测加载侧，排除 H2D 拷贝与 GPU 占用的干扰
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)

    memory_before = memory_snapshot()
    try:
        started = time.perf_counter()
        after_first = None
        seen = 0
        for index, batch in enumerate(loader):
            seen += int(batch["noisy"].size(0))
            if index == 0:
                after_first = time.perf_counter()
        finished = time.perf_counter()
        elapsed = finished - started
        steady_elapsed = finished - (after_first or started)
        steady_samples = seen - min(batch_size, seen)
        record.update(
            {
                "status": "ok",
                "batches": batches,
                "samples": seen,
                "elapsed_s": round(elapsed, 3),
                "end_to_end_samples_per_s": round(seen / elapsed, 2),
                # 首个 batch 含 worker 启动成本，稳态值才代表加载能力
                "steady_samples_per_s": round(steady_samples / steady_elapsed, 2) if steady_elapsed > 0 else None,
                "first_batch_latency_s": round((after_first - started), 3) if after_first else None,
            }
        )
    except BaseException as error:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        del loader
        del dataset
        time.sleep(1.0)  # 给 persistent worker 一点退出时间

    record["memory_before"] = memory_before
    record["memory_after"] = memory_snapshot()
    record["leftover_children"] = leftover_worker_processes()
    return record


def measure_dataloader(args: argparse.Namespace) -> dict[str, Any]:
    """DataLoader 吞吐。

    **必须覆盖 C=1**：手册 §0.3.7 的既有短测是在 C=34 上做的，但当前主线实验是单通道
    （C=1），而合成数据集的单样本成本随通道数变化，两者的加载能力不可互相代言。
    T0.7 的验收（单通道回归的墙钟下降）依赖 C=1 的数字。
    """
    batch_size = 128 if 128 in args.batches else args.batches[-1]
    by_channels: list[dict[str, Any]] = []
    for channels in args.dataloader_channels:
        combos = []
        for num_workers in args.workers:
            print(f"[dataloader] C={channels:<4d} num_workers={num_workers:<3d} ...", end="", flush=True)
            record = measure_dataloader_combo(num_workers, channels, batch_size, args)
            combos.append(record)
            if record["status"] == "ok":
                print(
                    f" end2end={record['end_to_end_samples_per_s']:>8.1f}  "
                    f"steady={str(record['steady_samples_per_s']):>8s} samples/s  "
                    f"avail={record['memory_after'].get('available_gib', 'n/a')} GiB  "
                    f"children={len(record['leftover_children'])}"
                )
            else:
                print(f" {record['status']}: {record.get('error', '')}")
        by_channels.append({"channels": channels, "combos": combos})
    return {
        "channels_measured": list(args.dataloader_channels),
        "batch_size": batch_size,
        "min_samples_per_combo": args.min_samples,
        "use_public_sources": args.use_public_sources,
        "groups": by_channels,
    }


# --------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    paths.assert_readonly_pool(args.out, operation="write hardware baseline into")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "probe": "T0.7 hardware and dataloader baseline",
        "measured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git_head(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "platform": platform.platform(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "memory_before": memory_snapshot(),
        "device": str(device),
    }

    if not args.skip_gpu:
        payload["gpu"] = measure_gpu(args, device)
    if not args.skip_dataloader:
        payload["dataloader"] = measure_dataloader(args)

    payload["memory_after"] = memory_snapshot()
    payload["leftover_children"] = leftover_worker_processes()

    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    if payload["leftover_children"]:
        print(f"WARNING: {len(payload['leftover_children'])} leftover child process(es) detected")


if __name__ == "__main__":
    main()
