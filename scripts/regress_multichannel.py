"""单通道回归对照（手册 T1.5 步骤 2）。

把"改造前"与"改造后"的单通道指标并列成表，输出
``runs/_regress/regress_multichannel.md``，并断言关键指标差 < 0.03。

对照的三个 run：

============================  ===============================================
run                           含义
============================  ===============================================
``baseline_singlechannel``    T0.5：改造前的历史基线复现（num_workers=0）
``after_t07_numworkers8``     T0.7：只有 num_workers 由 0 改 8，模型未动
``after_obs_multichannel``    T1.2：观测抽取器多通道化之后
``after_mixer``               T1.4：合成器真混音之后
============================  ===============================================

判读方法：T1.2/T1.4 各自的效应要**减去 num_workers 带来的差异**才对得上口径，
所以表里同时给出"相对 T0.5 基线"与"相对 T0.7 同口径对照"两组差。

默认**只做对照**（不启动训练）：一个 run 要 50 分钟，不该被脚本误触发。
确实缺 run 时加 ``--execute`` 才会补跑。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unicore_eeg import paths  # noqa: E402

#: 手册 T1.2 / T1.4 / T1.5 规定的验收阈值。
PAIRED_TOLERANCE = 0.03

#: run 名 → (说明, 输出目录)
RUNS: dict[str, tuple[str, Path]] = {
    "baseline_singlechannel": ("T0.5 改造前基线（num_workers=0）", paths.RUNS_ROOT / "baseline_singlechannel_seed42"),
    "after_t07_numworkers8": ("T0.7 仅改 num_workers（同口径对照）", paths.RUNS_ROOT / "after_t07_numworkers8_seed42"),
    "after_obs_multichannel": ("T1.2 观测抽取器多通道化", paths.RUNS_ROOT / "after_obs_multichannel_seed42"),
    "after_mixer": ("T1.4 合成器真混音", paths.RUNS_ROOT / "after_mixer_seed42"),
}

#: 每个变体要跟哪个 run 比（同口径），以及是否强制要求已存在。
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("after_t07_numworkers8", "baseline_singlechannel"),
    ("after_obs_multichannel", "after_t07_numworkers8"),
    ("after_obs_multichannel", "baseline_singlechannel"),
    ("after_mixer", "after_obs_multichannel"),
    ("after_mixer", "baseline_singlechannel"),
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-channel regression comparison (T1.5)")
    parser.add_argument("--out", type=Path, default=paths.RUNS_ROOT / "_regress" / "regress_multichannel.md")
    parser.add_argument("--tolerance", type=float, default=PAIRED_TOLERANCE)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="补跑缺失的 run（每个约 50 分钟）。默认只做对照。",
    )
    return parser.parse_args()


def load_routing(run_dir: Path) -> dict[str, float] | None:
    path = run_dir / "diagnostics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    routing = payload.get("routing")
    if not isinstance(routing, dict):
        return None
    return {key: float(value) for key, value in routing.items() if isinstance(value, (int, float))}


def run_missing(name: str, run_dir: Path) -> None:
    """按 T0.5 的口径补跑一个缺失的 run（走单卡串行调度器，便于留日志）。"""
    command = [
        sys.executable,
        "-u",
        str(paths.PROJECT_ROOT / "scripts" / "run_dual_gpu.py"),
        "--script",
        "scripts/first_experiment.py",
        "--gpus",
        "0",
        "--num-workers",
        "8",
        "--seeds",
        "42",
        "--extra",
        "--epochs 12 --train-samples 65536 --eval-samples 6000 --batch-size 64 --use-public-sources",
        "--out-prefix",
        str(run_dir),
    ]
    print(f"[execute] {name} 缺失，开始补跑：\n  " + " ".join(command))
    result = subprocess.run(command, cwd=paths.PROJECT_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"补跑 {name} 失败，返回码 {result.returncode}")


def main() -> None:
    args = parse_args()
    metrics: dict[str, dict[str, float]] = {}
    missing: list[str] = []

    for name, (_, run_dir) in RUNS.items():
        routing = load_routing(run_dir)
        if routing is None:
            missing.append(name)
            if args.execute:
                run_missing(name, run_dir)
                routing = load_routing(run_dir)
        if routing is not None:
            metrics[name] = routing

    if not metrics:
        raise SystemExit("没有任何可用的 run 指标，先跑 T0.5 / T1.2 / T1.4")

    baseline = metrics.get("baseline_singlechannel", {})
    metric_names = sorted(set(baseline) or set().union(*metrics.values()))

    lines: list[str] = []
    lines.append("# 单通道回归对照（T1.5）")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append(f"- 验收阈值：关键指标绝对差 < **{args.tolerance}**（手册 T1.2 / T1.4 / T1.5）")
    lines.append(f"- 生成命令：`python scripts/regress_multichannel.py`")
    lines.append("")
    lines.append("| run | 含义 | 目录 | 状态 |")
    lines.append("|---|---|---|---|")
    for name, (description, run_dir) in RUNS.items():
        state = "已纳入" if name in metrics else "**缺失**"
        lines.append(f"| `{name}` | {description} | `{run_dir.name}` | {state} |")
    lines.append("")

    lines.append("## 各 run 的路由指标")
    lines.append("")
    present = [name for name in RUNS if name in metrics]
    header = "| 指标 | " + " | ".join(f"`{name}`" for name in present) + " |"
    lines.append(header)
    lines.append("|---" * (len(present) + 1) + "|")
    for metric in metric_names:
        cells = []
        for name in present:
            value = metrics[name].get(metric)
            cells.append(f"{value:.4f}" if value is not None else "—")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 逐对比对（绝对差）")
    lines.append("")
    lines.append("| 变体 | 对照 | macro_auroc 差 | 最大指标差 | 判定 |")
    lines.append("|---|---|---:|---:|---|")
    worst_overall = 0.0
    compared = 0
    skipped: list[tuple[str, str]] = []
    for variant, reference in COMPARISONS:
        if variant not in metrics or reference not in metrics:
            lines.append(f"| `{variant}` | `{reference}` | — | — | 缺 run，未比对 |")
            skipped.append((variant, reference))
            continue
        current, base = metrics[variant], metrics[reference]
        shared = [metric for metric in base if metric in current]
        diffs = {metric: abs(base[metric] - current[metric]) for metric in shared}
        auroc_diff = diffs.get("macro_auroc")
        worst = max(diffs.values()) if diffs else float("nan")
        worst_overall = max(worst_overall, worst if worst == worst else 0.0)
        compared += 1
        verdict = "PASS" if worst < args.tolerance else "**FAIL**"
        if variant == "after_mixer" and reference == "baseline_singlechannel":
            verdict += "（手册 T1.4 的正式判据）"
        lines.append(
            f"| `{variant}` | `{reference}` | "
            f"{(f'{auroc_diff:.5f}' if auroc_diff is not None else '—')} | {worst:.5f} | {verdict} |"
        )

    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(
        f"- 纳入 {len(metrics)} 个 run；{len(missing)} 个缺失"
        + (f"（{', '.join('`' + name + '`' for name in missing)}）" if missing else "")
    )
    lines.append(f"- 已完成 {compared}/{len(COMPARISONS)} 组逐对比对")
    if skipped:
        lines.append(
            "- ⚠️ **对照不完整**：有 "
            f"{len(skipped)} 组因缺 run 未比对，此时不应声称回归通过。"
        )
    if compared:
        lines.append(
            f"- 已比对部分的最大指标差：**{worst_overall:.5f}**（阈值 {args.tolerance}）"
        )
    lines.append(
        "- 判读提醒：`after_obs_multichannel` / `after_mixer` 相对 `baseline_singlechannel` "
        "的差里含 num_workers 由 0 改 8 带来的约 0.002 量级差异（见 T0.7 记录），"
        "要单独看 T1.2 / T1.4 的效应应对照 `after_t07_numworkers8` 那一行。"
    )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"写入 {args.out}")
    print(f"完成 {compared}/{len(COMPARISONS)} 组比对")
    if compared:
        print(f"已比对部分的最大指标差 {worst_overall:.5f}（阈值 {args.tolerance}）")
    if skipped:
        print(f"⚠️ 有 {len(skipped)} 组因缺 run 未比对，回归不能算通过")
        raise SystemExit(2)
    if worst_overall >= args.tolerance:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
