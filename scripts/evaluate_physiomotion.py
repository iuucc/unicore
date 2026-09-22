from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from unicore_eeg import ARTIFACT_NAMES, UniCOREEG, UniCOREEGConfig
from unicore_eeg.paths import pool_path
from unicore_eeg.physiomotion import PHYSIOMOTION_FAMILIES, PhysioMotionWindowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate routing on real PhysioMotion annotations.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=pool_path("physiomotion"))
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 31)))
    parser.add_argument("--max-windows", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    # 默认 0 而非 §5.5 建议的 8：这是短评估（默认 2000 窗口）。
    # 实测 Windows spawn 起 8 个 worker 的固定启动成本约 40 s（每个 worker 都要重新 import torch），
    # 在 2000 窗口量级上大于并行加载省下的时间。窗口数上万时再调高。
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/first_experiment/physiomotion"))
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = UniCOREEGConfig(**checkpoint["config"])
    model = UniCOREEG(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = PhysioMotionWindowDataset(
        args.root,
        subjects=args.subjects,
        channels=config.in_channels,
        sample_rate=config.sample_rate,
        length=config.window_size,
        max_windows=args.max_windows,
    )
    loader_kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)
    labels, probabilities, routes, families, bypasses = [], [], [], [], []
    for batch in tqdm(loader, desc="PhysioMotion"):
        eeg = batch["eeg"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(eeg)
        labels.append(batch["labels"])
        probabilities.append(outputs["probabilities"].float().cpu())
        routes.append(outputs["route"].float().cpu())
        families.append(batch["family"])
        bypasses.append(outputs["bypass"].float().cpu())
    labels_tensor = torch.cat(labels).numpy()
    probability_tensor = torch.cat(probabilities).numpy()
    route_tensor = torch.cat(routes).numpy()
    family_tensor = torch.cat(families).numpy()
    bypass_tensor = torch.cat(bypasses).numpy()

    supported = (1, 2, 4)
    rows = []
    for index in supported:
        rows.append({
            "expert": ARTIFACT_NAMES[index],
            "f1": f1_score(labels_tensor[:, index], probability_tensor[:, index] >= 0.5, zero_division=0),
            "auroc": roc_auc_score(labels_tensor[:, index], probability_tensor[:, index]),
        })
    metrics = pd.DataFrame(rows)
    matrix = pd.DataFrame(
        [route_tensor[family_tensor == family].mean(axis=0) for family in range(len(PHYSIOMOTION_FAMILIES))],
        index=PHYSIOMOTION_FAMILIES,
        columns=ARTIFACT_NAMES,
    )
    baseline_bypass = float(bypass_tensor[family_tensor == 0].mean())
    metrics.to_csv(args.out / "routing_metrics.csv", index=False)
    matrix.to_csv(args.out / "activation_matrix.csv")
    report = "\n".join([
        "# PhysioMotion 真实路由外测",
        "",
        "该数据没有逐点清洁真值，本报告只评价标签识别、专家激活和基线旁路，不报告重建优劣。",
        "",
        "## 分类指标",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"基线窗口 clean-bypass 比例：{baseline_bypass:.4f}",
        "",
        "## 类别 x 专家激活矩阵",
        "",
        matrix.to_markdown(floatfmt=".3f"),
    ])
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(f"report={args.out / 'report.md'}")


if __name__ == "__main__":
    main()
