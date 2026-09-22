"""运行配置的落地（手册 §5.5「运行配置」）。

§5.5 的四条硬性约束必须在代码里强制执行，而不是写在文档里等人自觉：

1. ``torch_compile`` 必须为 ``false``（本机 Triton 缺失，§0.3.4）；
2. ``num_workers`` 不得为 0（§0.3.7 实测 GPU 空转 23%–35%）；
3. 多卡只用 ``CUDA_VISIBLE_DEVICES`` 做进程隔离，**禁止 DDP / 跨卡梯度同步**（本机无 NCCL）；
4. 需要更大等效 batch 时用梯度累积，不要把 ``batch_size`` 提到 256（单卡 15.92 GiB 必 OOM）。

本模块把 1、2 做成可执行检查，把 3、4 的取值集中在 ``dataloader_kwargs`` 与
``configure_runtime`` 里，供所有实验脚本共用，避免各写一份后逐渐漂移。
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from . import config as config_module


__all__ = [
    "MAX_BATCH_SIZE",
    "MAX_TOTAL_WORKERS",
    "TritonMissingError",
    "configure_runtime",
    "dataloader_kwargs",
    "check_ddp_allowed",
]


#: 单卡 15.92 GiB 下 bf16 的安全上限；256 必 OOM（§0.3.8 实测）。
MAX_BATCH_SIZE = 128

#: 双卡并行时两个进程的 worker 之和上限；每 worker 常驻约 1.4 GB（§0.3.7）。
MAX_TOTAL_WORKERS = 16


class TritonMissingError(RuntimeError):
    """本机缺 Triton，``torch.compile`` 不可用。"""


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def configure_runtime(
    config: Mapping[str, Any] | None = None,
    *,
    enable_compile: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """应用全局运行配置，返回生效设置。

    ``config`` 可为 ``configs/base.yaml`` 的 ``runtime`` 段所在映射；缺省时用 §5.5 的值。
    ``enable_compile=True`` 且本机无 Triton 时**直接抛错**，而不是静默降级——
    静默降级会让人以为开了编译却实际没开。
    """
    settings = dict(config_module.get(config, "runtime", {}) or {}) if config else {}
    torch_compile = bool(settings.get("torch_compile", False)) or bool(enable_compile)
    if torch_compile and not _triton_available():
        raise TritonMissingError(
            "torch.compile 已请求但本机缺少 Triton。手册 §0.3.4 把 torch_compile 列为硬约束禁止项"
            "（缺失 Triton 时编译不可靠）。请在配置里保持 runtime.torch_compile=false，"
            "或先安装 triton 再启用。"
        )

    if bool(settings.get("cudnn_benchmark", True)) and device is not None and device.type == "cuda":
        # 固定输入形状下有效；与本文件的 deterministic 选项互斥
        torch.backends.cudnn.benchmark = True
    precision = str(settings.get("matmul_precision", "high"))
    torch.set_float32_matmul_precision(precision)

    return {
        "precision": str(settings.get("precision", "bf16")),
        # bf16 不需要损失缩放；现有 GradScaler(enabled=False) 保持
        "grad_scaler": bool(settings.get("grad_scaler", False)),
        "torch_compile": torch_compile,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_precision": precision,
        "device": str(device) if device is not None else None,
    }


def check_ddp_allowed(device_count: int, use_ddp: bool) -> None:
    """§5.5 约束 3：本机无 NCCL，禁止 DDP / 跨卡梯度同步。"""
    if use_ddp:
        nccl = torch.distributed.is_available() and torch.distributed.is_nccl_available()
        raise RuntimeError(
            f"禁止使用 DDP（请求 device_count={device_count}，NCCL 可用={nccl}）。"
            "本机 NCCL 不可用且两块卡位于不同 PCIe 根端口，跨卡梯度同步会成为瓶颈；"
            "请按 §0.3.6 策略 A 用 CUDA_VISIBLE_DEVICES 做进程隔离。"
        )


def dataloader_kwargs(
    batch_size: int,
    num_workers: int,
    device: torch.device,
    *,
    pin_memory: bool | None = None,
) -> dict[str, Any]:
    """构造 DataLoader 参数，并把 §5.5 的取值约束变成现场报错。

    * ``persistent_workers`` / ``prefetch_factor`` **只在** ``num_workers > 0`` 时传入：
      PyTorch 在 0 worker 下传这两个参数会直接报错（这也是手册 §0.3.7 提到的坑）。
    * 梯度累积是训练循环的事，不是 DataLoader 参数，因此**不**出现在返回值里；
      训练循环从配置的 ``dataloader.grad_accum_steps`` 读取。
    """
    if batch_size > MAX_BATCH_SIZE:
        raise ValueError(
            f"batch_size={batch_size} 超过单卡安全上限 {MAX_BATCH_SIZE}（§0.3.8：256 必 OOM）。"
            "需要更大等效 batch 时请提高 grad_accum_steps，而不是 batch_size。"
        )
    if num_workers < 0:
        raise ValueError(f"num_workers 不能为负数：{num_workers}")
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda" if pin_memory is None else pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs
