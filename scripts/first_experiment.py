from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unicore_eeg import ARTIFACT_NAMES, UniCOREEG, UniCOREEGConfig, paths
from unicore_eeg.config import resolve_settings
from unicore_eeg.losses import UniCORELoss
from unicore_eeg.manifest import write_run_manifest
from unicore_eeg.model import count_parameters
from unicore_eeg.runtime import configure_runtime, dataloader_kwargs
from unicore_eeg.synthetic import SyntheticEEGDataset

# 注意：**不要**在模块顶层 import pandas / sklearn。
# Windows 的 DataLoader 用 spawn 起 worker，每个 worker 会重新导入本模块（作为 __mp_main__），
# 顶层的 sklearn → scipy.stats/interpolate 导入链偶发抛
# `SystemError: error return without exception set`（scipy._lib._docscrape 处理文档字符串时），
# 会让 worker 启动即崩溃并被反复重建。pandas / sklearn 只有主进程的打分与报表环节需要，
# 因此下沉到用到它们的函数内部。


CONDITION_NAMES = (
    "clean",
    *ARTIFACT_NAMES,
    "heldout_harmonic",
    "heldout_ocular_drift",
    "heldout_myogenic",
    "heldout_cardiac",
    "heldout_motion_transient",
)
ROUTE_MODES = ("oracle", "learned", "all")

#: 代码内默认值（优先级最低）。与改造前的 argparse 默认值一致，
#: 唯一的例外是 num_workers：T0.7 按手册 §0.3.7 由 0 改为 8。
DEFAULTS: dict[str, Any] = {
    "epochs": 4,
    "train_samples": 4096,
    "eval_samples": 1200,
    "batch_size": 24,
    "channels": 1,
    "length": 1000,
    "sample_rate": 500,
    "lr": 1e-4,
    "num_workers": 8,
    "seed": 42,
    "use_public_sources": False,
    "data_root": paths.RAW_ROOT,
    "checkpoint": None,
    "out": paths.RUNS_ROOT / "first_experiment",
}

#: argparse dest → 配置点分路径。配置文件里取不到的项回落到 DEFAULTS。
CONFIG_MAP = {
    "sample_rate": "signal.sample_rate",
    "length": "signal.window_size",
    "batch_size": "dataloader.batch_size",
    "num_workers": "dataloader.num_workers",
}


def parse_args() -> tuple[argparse.Namespace, dict[str, Any] | None]:
    """解析命令行，并按"命令行 > --config > 代码默认值"合成生效设置。

    所有可由配置提供的选项都用 ``default=argparse.SUPPRESS``：未显式给出时
    不出现在命名空间里，因此 ``resolve_settings`` 能准确区分"用户没传"与
    "用户传了默认值"。这样既满足手册 §5.4 的配置化要求，又不改变原有 CLI 的语义
    （T0.5 的复现命令原样可跑）。
    """
    parser = argparse.ArgumentParser(description="Run the registered UniCORE-EEG first experiment.")
    parser.add_argument("--config", type=Path, help="实验配置（YAML）；命令行显式参数优先级更高")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--train-samples", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--eval-samples", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--channels", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sample-rate", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--use-public-sources", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--data-root", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=argparse.SUPPRESS)
    parsed = parser.parse_args()
    settings, config = resolve_settings(parsed, DEFAULTS, parsed.config, CONFIG_MAP)
    return argparse.Namespace(**settings), config


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def configure_stage(model: UniCOREEG, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = stage != "residual"
    if stage == "decomposition":
        for parameter in model.residual_refiner.parameters():
            parameter.requires_grad = False
    elif stage == "residual":
        for module in (model.residual_refiner, model.identity_gate):
            for parameter in module.parameters():
                parameter.requires_grad = True


def stage_for_epoch(epoch: int, epochs: int) -> str:
    if epochs < 3:
        return "joint"
    if epoch <= max(1, math.floor(epochs * 0.5)):
        return "decomposition"
    if epoch <= max(2, math.ceil(epochs * 0.75)):
        return "residual"
    return "joint"


def train_model(model: UniCOREEG, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> None:
    loss_fn = UniCORELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    current_stage = ""
    optimizer: torch.optim.Optimizer | None = None
    skipped_steps = 0
    for epoch in range(1, args.epochs + 1):
        stage = stage_for_epoch(epoch, args.epochs)
        if stage != current_stage:
            configure_stage(model, stage)
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=args.lr * (0.2 if stage == "joint" and epoch > 1 else 1.0),
                weight_decay=1e-4,
            )
            current_stage = stage
        assert optimizer is not None
        model.train()
        running = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs} {stage}")
        for step, cpu_batch in enumerate(progress, start=1):
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(
                    batch["noisy"],
                    metadata=batch["metadata"],
                    disabled_experts=batch["disabled_experts"],
                    enable_residual=stage != "decomposition",
                )
                loss, logs = loss_fn(outputs, batch)
            if not torch.isfinite(loss):
                skipped_steps += 1
                progress.write(f"skipping non-finite loss: { {key: float(value) for key, value in logs.items()} }")
                if skipped_steps > 16:
                    raise RuntimeError("too many non-finite training steps")
                continue
            # T0.7（手册 §0.3.6 策略 B）：必须用 .sum()。在 torch.nn.DataParallel 下损失会被
            # 跨卡聚合成"长度 = 卡数"的向量，直接 backward() 会报
            # RuntimeError: grad can be implicitly created only for scalar outputs。
            # 单卡时它是 0 维张量，.sum() 是等价操作，不改变任何数值。
            scaler.scale(loss).sum().backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                1.0,
            )
            if not torch.isfinite(grad_norm):
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                progress.write(f"skipping non-finite gradient at epoch={epoch} step={step}")
                if skipped_steps > 16:
                    raise RuntimeError("too many non-finite training steps")
                continue
            scaler.step(optimizer)
            scaler.update()
            running += float(logs["loss"])
            progress.set_postfix(loss=f"{running / step:.3f}")
        checkpoint = {
            "config": model.config.__dict__,
            "model": model.state_dict(),
            "epoch": epoch,
            "stage": stage,
        }
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise RuntimeError(f"model became non-finite at epoch {epoch}")
        torch.save(checkpoint, args.out / f"epoch_{epoch:02d}.pt")
        torch.save(checkpoint, args.out / "last.pt")


def _sample_correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.flatten(1) - pred.flatten(1).mean(dim=-1, keepdim=True)
    target = target.flatten(1) - target.flatten(1).mean(dim=-1, keepdim=True)
    return (pred * target).sum(dim=-1) / (pred.norm(dim=-1) * target.norm(dim=-1)).clamp_min(1e-8)


@torch.no_grad()
def evaluate(model: UniCOREEG, loader: DataLoader, device: torch.device) -> tuple[pd.DataFrame, dict[str, object]]:
    # 函数内导入：避免 spawn 出来的 DataLoader worker 也把 sklearn/scipy.stats 拖进来（见文件头说明）。
    import pandas as pd
    from sklearn.metrics import f1_score, roc_auc_score

    model.eval()
    rows: list[dict[str, object]] = []
    routing = {mode: torch.zeros(len(CONDITION_NAMES), len(ARTIFACT_NAMES), device=device) for mode in ROUTE_MODES}
    routing_count = torch.zeros(len(CONDITION_NAMES), device=device)
    all_labels, all_probabilities, all_artifact_presence = [], [], []

    for cpu_batch in tqdm(loader, desc="evaluate routes"):
        batch = move_batch(cpu_batch, device)
        for mode in ROUTE_MODES:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(
                    batch["noisy"],
                    metadata=batch["metadata"],
                    route_mode=mode,
                    oracle_labels=batch["labels"] if mode == "oracle" else None,
                    disabled_experts=batch["disabled_experts"],
                )
            if mode == "learned":
                all_labels.append(batch["labels"].cpu())
                all_probabilities.append(outputs["probabilities"].float().cpu())
                all_artifact_presence.append(outputs["artifact_presence_probability"].float().cpu())
            total_artifact = batch["artifacts"].sum(dim=1)
            predicted_artifact = outputs["artifact_sum_norm"] * outputs["mad"]
            clean_error = (outputs["clean"] - batch["clean"]).flatten(1)
            noisy_error = (batch["noisy"] - batch["clean"]).flatten(1)
            artifact_error = (predicted_artifact - total_artifact).flatten(1)
            target_flat = batch["clean"].flatten(1)
            rrmse = clean_error.norm(dim=-1) / target_flat.norm(dim=-1).clamp_min(1e-8)
            artifact_norm = total_artifact.flatten(1).norm(dim=-1)
            artifact_rrmse = artifact_error.norm(dim=-1) / artifact_norm.clamp_min(1e-8)
            snr_improvement = 10.0 * torch.log10(
                noisy_error.square().mean(dim=-1).clamp_min(1e-10) / clean_error.square().mean(dim=-1).clamp_min(1e-10)
            )
            clean_samples = batch["is_clean"].bool()
            artifact_rrmse = artifact_rrmse.masked_fill(clean_samples, float("nan"))
            snr_improvement = snr_improvement.masked_fill(clean_samples, float("nan"))
            correlation = _sample_correlation(outputs["clean"], batch["clean"])
            modification = (outputs["clean"] - batch["noisy"]).abs().flatten(1).mean(dim=-1)
            for sample in range(batch["noisy"].size(0)):
                condition = int(batch["condition"][sample])
                row = {
                    "route_mode": mode,
                    "condition": CONDITION_NAMES[condition],
                    "rrmse": float(rrmse[sample]),
                    "correlation": float(correlation[sample]),
                    "snr_improvement_db": float(snr_improvement[sample]),
                    "artifact_rrmse": float(artifact_rrmse[sample]),
                    "input_modification_mae": float(modification[sample]),
                    "bypass": bool(outputs["bypass"][sample]),
                    "artifact_presence_probability": float(outputs["artifact_presence_probability"][sample]),
                    "unknown_probability": float(outputs["probabilities"][sample, -1]),
                    "unknown_route": float(outputs["route"][sample, -1]),
                }
                for expert_index, expert_name in enumerate(ARTIFACT_NAMES):
                    row[f"probability_{expert_name}"] = float(outputs["probabilities"][sample, expert_index])
                rows.append(row)
                routing[mode][condition] += outputs["route"][sample].float()
                if mode == "learned":
                    routing_count[condition] += 1

    frame = pd.DataFrame(rows)
    labels = torch.cat(all_labels).numpy()
    probabilities = torch.cat(all_probabilities).numpy()
    artifact_presence = torch.cat(all_artifact_presence).numpy()
    artifact_targets = (labels.sum(axis=1) > 0).astype(np.float32)
    route_metrics = {
        "macro_f1_at_0_5": float(f1_score(labels, probabilities >= 0.5, average="macro", zero_division=0)),
        "macro_f1_at_route_threshold": float(
            f1_score(labels, probabilities >= model.config.probability_threshold, average="macro", zero_division=0)
        ),
        "macro_auroc": float(roc_auc_score(labels, probabilities, average="macro")),
        "artifact_presence_auroc": float(roc_auc_score(artifact_targets, artifact_presence)),
        "unknown_detection_auroc": float(roc_auc_score(labels[:, -1], probabilities[:, -1])),
        "unknown_false_activation_clean": float(probabilities[labels.sum(axis=1) == 0, -1].mean()),
        "unknown_false_activation_known": float(probabilities[(labels[:, :-1].sum(axis=1) > 0) & (labels[:, -1] == 0), -1].mean()),
    }
    matrices = {
        mode: (matrix / routing_count.clamp_min(1).unsqueeze(-1)).cpu().numpy().tolist()
        for mode, matrix in routing.items()
    }
    return frame, {"routing": route_metrics, "activation_matrices": matrices}


def write_report(frame: pd.DataFrame, diagnostics: dict[str, object], out: Path) -> None:
    # 同 evaluate()：函数内导入，保持主模块顶层轻量。
    import pandas as pd

    frame.to_csv(out / "sample_metrics.csv", index=False)
    summary = frame.groupby(["route_mode", "condition"], sort=False).agg(
        rrmse=("rrmse", "mean"),
        correlation=("correlation", "mean"),
        snr_improvement_db=("snr_improvement_db", "mean"),
        artifact_rrmse=("artifact_rrmse", "mean"),
        input_modification_mae=("input_modification_mae", "mean"),
        bypass_rate=("bypass", "mean"),
    ).reset_index()
    summary.to_csv(out / "summary_metrics.csv", index=False)
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    lines = [
        "# UniCORE-EEG 首个实验报告",
        "",
        "## 实验范围",
        "",
        "完整六专家模型；清洁输入、五类已知单伪迹、未见机制 Unknown，以及五类禁用对应已知专家的留一条件；比较 Oracle、Learned 和 All Experts 路由。",
        "",
        "## 路由检测",
        "",
    ]
    for key, value in diagnostics["routing"].items():
        lines.append(f"- {key}: {value:.4f}")
    lines.extend(["", "## 信号结果", "", summary.to_markdown(index=False, floatfmt=".4f"), "", "## 专家激活矩阵", ""])
    for mode, matrix in diagnostics["activation_matrices"].items():
        lines.extend([f"### {mode}", "", pd.DataFrame(matrix, index=CONDITION_NAMES, columns=ARTIFACT_NAMES).to_markdown(floatfmt=".3f"), ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args, run_config = parse_args()
    args.out = Path(args.out)
    args.data_root = Path(args.data_root)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = configure_runtime(run_config, device=device)

    # 手册 §5.3：run 目录必须含 run_manifest.json（含 git 状态、代码版本哈希、
    # 配置副本、数据集规模、采样率与通道列表）。写在训练之前，崩溃也不丢登记。
    write_run_manifest(
        args.out,
        command=sys.argv,
        settings=vars(args),
        config=run_config,
        dataset={
            "name": "synthetic",
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "eval_mode": "first_experiment",
            "channels": args.channels,
            "channel_list": None,
            "window_size": args.length,
            "sample_rate": args.sample_rate,
            "use_public_sources": args.use_public_sources,
            "data_root": str(args.data_root),
        },
        extra={"route_modes": list(ROUTE_MODES), "condition_names": list(CONDITION_NAMES)},
    )

    model_config = UniCOREEGConfig(in_channels=args.channels, sample_rate=args.sample_rate, window_size=args.length)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model_config = UniCOREEGConfig(**checkpoint["config"])
    model = UniCOREEG(model_config).to(device)
    if args.checkpoint:
        model.load_state_dict(checkpoint["model"])
    train_set = SyntheticEEGDataset(
        samples=args.train_samples,
        channels=args.channels,
        length=args.length,
        sample_rate=args.sample_rate,
        seed=args.seed,
        use_public_sources=args.use_public_sources,
        data_root=args.data_root,
        split="train",
    )
    eval_set = SyntheticEEGDataset(
        samples=args.eval_samples,
        channels=args.channels,
        length=args.length,
        sample_rate=args.sample_rate,
        seed=args.seed + 1_000_000,
        mode="first_experiment",
        use_public_sources=args.use_public_sources,
        data_root=args.data_root,
        split="test",
    )
    loader_kwargs = dataloader_kwargs(args.batch_size, args.num_workers, device)
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    eval_loader = DataLoader(eval_set, shuffle=False, **loader_kwargs)
    print(
        f"device={device} params={count_parameters(model):,} num_workers={args.num_workers} "
        f"precision={runtime['precision']} torch_compile={runtime['torch_compile']}"
    )
    if not args.checkpoint:
        train_model(model, train_loader, args, device)
    frame, diagnostics = evaluate(model, eval_loader, device)
    write_report(frame, diagnostics, args.out)
    print(f"report={args.out / 'report.md'}")


if __name__ == "__main__":
    main()
