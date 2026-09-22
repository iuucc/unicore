from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from unicore_eeg import ARTIFACT_NAMES
from unicore_eeg.on004784 import (
    ON004784_FAMILIES,
    ON004784_TASKS,
    On004784WindowDataset,
    discover_on004784_recordings,
    load_ground_truth_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and sanity-check OpenNeuro/NEMAR on004784.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/on004784"))
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--sample-rate", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path("data/on004784_sanity_report.md"))
    return parser.parse_args()


def format_labels(labels: torch.Tensor) -> str:
    active = [name for name, value in zip(ARTIFACT_NAMES, labels.tolist()) if value > 0.5]
    return ", ".join(active) if active else "clean"


def main() -> None:
    args = parse_args()
    recordings = discover_on004784_recordings(args.root)
    ground_truth = load_ground_truth_summary(args.root)
    dataset = On004784WindowDataset(
        args.root,
        channels=args.channels,
        length=args.length,
        sample_rate=args.sample_rate,
        max_windows=None,
    )

    first_by_task: dict[str, int] = {}
    for index, window in enumerate(dataset.windows):
        task = dataset.recordings[window.recording_index].task
        first_by_task.setdefault(task, index)

    lines = [
        "# on004784 数据接入 Sanity Check",
        "",
        "## 数据审计",
        "",
        f"- 根目录：`{args.root.resolve()}`",
        f"- 任务条件：{', '.join(ON004784_TASKS)}",
        f"- Ground truth 文件：`{ground_truth['path']}`",
        f"- Ground truth 形状：`{ground_truth['shape']}`，来源通道组：`{ground_truth['source_groups']}`",
        "",
        "## BIDS 记录",
        "",
        "| task | family | labels | sfreq | channels | eeg_channels | samples | duration_s |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for recording in recordings:
        labels = torch.tensor(recording.labels)
        lines.append(
            f"| {recording.task} | {ON004784_FAMILIES[recording.family]} | {format_labels(labels)} | "
            f"{recording.sample_rate:.1f} | {recording.channels} | {len(recording.eeg_channel_indices)} | "
            f"{recording.samples} | {recording.duration:.1f} |"
        )

    lines.extend([
        "",
        "## 窗口样本检查",
        "",
        f"- 窗口长度：`{args.length}` 点；目标采样率：`{args.sample_rate}` Hz；通道数：`{args.channels}`",
        f"- 总窗口数：`{len(dataset)}`",
        "",
        "| task | shape | finite | mean | std | labels |",
        "|---|---|---:|---:|---:|---|",
    ])
    for task in ON004784_TASKS:
        sample = dataset[first_by_task[task]]
        eeg = sample["eeg"]
        assert isinstance(eeg, torch.Tensor)
        labels = sample["labels"]
        assert isinstance(labels, torch.Tensor)
        lines.append(
            f"| {task} | `{tuple(eeg.shape)}` | {bool(torch.isfinite(eeg).all())} | "
            f"{float(eeg.mean()):.4f} | {float(eeg.std()):.4f} | {format_labels(labels)} |"
        )

    lines.extend([
        "",
        "## 结论",
        "",
        "- `.set/.fdt` 可由 `mne` 正常读取。",
        "- 六类条件已映射到 UniCORE 六专家标签空间，其中 Facial 与 Neck 都属于 `myogenic`，但保留不同 family 便于后续分层统计。",
        "- 当前检查只验证数据读取、标签映射、窗口切片和张量数值稳定性；暂不启动训练。",
        "",
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"windows={len(dataset)} tasks={len(recordings)} gt_shape={ground_truth['shape']}")


if __name__ == "__main__":
    main()
