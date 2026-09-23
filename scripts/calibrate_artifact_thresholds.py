"""Validation-only artifact threshold calibration and reliability analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unicore_eeg import ARTIFACT_NAMES, UniCOREEG, UniCOREEGConfig
from unicore_eeg.batching import collate_variable_channels
from unicore_eeg.routing_metrics import masked_routing_metrics, masked_threshold_calibration
from unicore_eeg.synthetic import SyntheticEEGDataset


def collect(model, loader, device):
    labels, masks, probabilities = [], [], []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch["noisy"], metadata=batch["metadata"], disabled_experts=batch["disabled_experts"])
            labels.append(batch["labels"].cpu().numpy())
            masks.append(batch["label_mask"].cpu().numpy())
            probabilities.append(output["probabilities"].float().cpu().numpy())
    return tuple(np.concatenate(items) for items in (labels, masks, probabilities))


def metric_payload(rows, summary, thresholds):
    return {"thresholds": {name: float(value) for name, value in zip(ARTIFACT_NAMES, thresholds)}, "metrics": rows, "summary": summary}


def reliability_plot(labels, probabilities, masks, out):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for index, (axis, name) in enumerate(zip(axes.flat, ARTIFACT_NAMES)):
        valid = masks[:, index].astype(bool)
        y, p = labels[valid, index], probabilities[valid, index]
        bins = np.linspace(0, 1, 11)
        centers, observed = [], []
        for low, high in zip(bins[:-1], bins[1:]):
            selected = (p >= low) & ((p <= high) if high == 1 else (p < high))
            if selected.any():
                centers.append(float(p[selected].mean())); observed.append(float(y[selected].mean()))
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.plot(centers, observed, "o-", label=name)
        axis.set_title(name); axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.grid(alpha=0.25)
    figure.supxlabel("mean predicted probability"); figure.supylabel("fraction positive")
    figure.tight_layout(); figure.savefig(out, dpi=180); plt.close(figure)


def ocular_analysis(labels, probabilities, masks, threshold, out, figure_dir):
    from sklearn.metrics import precision_recall_curve, auc
    import matplotlib.pyplot as plt
    index = ARTIFACT_NAMES.index("ocular_drift")
    valid = masks[:, index].astype(bool)
    y, p = labels[valid, index], probabilities[valid, index]
    quantiles = np.quantile(p, [0, .01, .05, .25, .5, .75, .95, .99, 1]).tolist()
    precision, recall, _ = precision_recall_curve(y, p)
    pr_auc = float(auc(recall, precision))
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label=f"AUPRC={pr_auc:.4f}")
    axis.set_xlabel("recall"); axis.set_ylabel("precision"); axis.set_title("ocular_drift precision-recall")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout(); figure.savefig(figure_dir / "ocular_pr_curve.png", dpi=180); plt.close(figure)
    positive = p[y == 1]; negative = p[y == 0]
    report = """# Ocular calibration analysis

The validation-selected threshold is `{threshold:.8f}`. The fixed 0.5 decision
gave F1=0 because the positive probabilities are mostly below 0.5; this does
not contradict AUROC=0.9223, which measures ranking quality across all possible
thresholds.

## Probability distribution

- all quantiles (0%, 1%, 5%, 25%, 50%, 75%, 95%, 99%, 100%): `{quantiles}`
- positive mean/min/max: `{pos_mean:.6f}` / `{pos_min:.6f}` / `{pos_max:.6f}`
- negative mean/min/max: `{neg_mean:.6f}` / `{neg_min:.6f}` / `{neg_max:.6f}`
- validation positives: `{support}`
- validation AUROC: `{auroc:.6f}`
- validation AUPRC: `{auprc:.6f}`

The precision-recall curve is saved as `figures/ocular_pr_curve.png`.
""".format(threshold=threshold, quantiles=quantiles, pos_mean=float(positive.mean()), pos_min=float(positive.min()), pos_max=float(positive.max()), neg_mean=float(negative.mean()), neg_min=float(negative.min()), neg_max=float(negative.max()), support=int(y.sum()), auroc=float(masked_routing_metrics(labels, probabilities, label_mask=masks)[0][index]["auroc"]), auprc=pr_auc)
    out.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-samples", type=int, default=1200)
    parser.add_argument("--test-samples", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    args.out.mkdir(parents=True, exist_ok=True); figures = args.out / "figures"; figures.mkdir(exist_ok=True)
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = UniCOREEG(UniCOREEGConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=False); model.eval()
    make_loader = lambda samples, seed, split: DataLoader(SyntheticEEGDataset(samples=samples, channels=1, seed=seed, split=split, mode="first_experiment"), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_variable_channels)
    val = collect(model, make_loader(args.val_samples, args.seed + 100_000, "val"), device)
    test = collect(model, make_loader(args.test_samples, args.seed + 1_000_000, "test"), device)
    val_labels, val_masks, val_probabilities = val
    thresholds = masked_threshold_calibration(val_labels, val_probabilities, val_masks, max_fpr=None)
    (args.out / "thresholds.json").write_text(json.dumps({name: float(value) for name, value in zip(ARTIFACT_NAMES, thresholds)}, indent=2), encoding="utf-8")
    before_rows, before_summary = masked_routing_metrics(test[0], test[2], label_mask=test[1])
    after_rows, after_summary = masked_routing_metrics(test[0], test[2], thresholds=thresholds, label_mask=test[1])
    (args.out / "before_calibration_metrics.json").write_text(json.dumps(metric_payload(before_rows, before_summary, np.full(6, .5)), indent=2), encoding="utf-8")
    (args.out / "after_calibration_metrics.json").write_text(json.dumps(metric_payload(after_rows, after_summary, thresholds), indent=2), encoding="utf-8")
    reliability_plot(test[0], test[2], test[1], figures / "calibration_curve.png")
    ocular_analysis(val_labels, val_probabilities, val_masks, float(thresholds[1]), args.out / "ocular_calibration_analysis.md", figures)
    (args.out / "run_info.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "validation_seed": args.seed + 100_000, "test_seed": args.seed + 1_000_000, "threshold_source": "validation_only"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
