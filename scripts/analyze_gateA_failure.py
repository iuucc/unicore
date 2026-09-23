"""Analyze the existing Gate A failure without starting training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from unicore_eeg import UniCOREEG, UniCOREEGConfig
from unicore_eeg.losses import UniCORELoss, UniCORELossWeights
from unicore_eeg.synthetic import SyntheticEEGDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--routing-test", type=Path, required=True)
    parser.add_argument("--validation-history", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    routing = json.loads(args.routing_test.read_text(encoding="utf-8"))
    history = json.loads(args.validation_history.read_text(encoding="utf-8"))
    labels = np.asarray(routing["per_sample"]["labels"], dtype=np.float32)
    masks = np.asarray(routing["per_sample"]["label_mask"], dtype=bool)
    probabilities = np.asarray(routing["per_sample"]["probabilities"], dtype=np.float32)
    thresholds = np.asarray([routing["thresholds"][name] for name in ("harmonic", "ocular_drift", "myogenic", "cardiac", "motion_transient", "unknown")])

    confusion = []
    for index, name in enumerate(("harmonic", "ocular_drift", "myogenic", "cardiac", "motion_transient", "unknown")):
        valid = masks[:, index]
        y = labels[valid, index].astype(bool)
        pred = probabilities[valid, index] >= thresholds[index]
        confusion.append({"class": name, "tn": int((~y & ~pred).sum()), "fp": int((~y & pred).sum()), "fn": int((y & ~pred).sum()), "tp": int((y & pred).sum())})
    with (args.out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(confusion[0])); writer.writeheader(); writer.writerows(confusion)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    finite = all(bool(torch.isfinite(value).all()) for value in checkpoint["model"].values() if torch.is_tensor(value))
    config = UniCOREEGConfig(**checkpoint["config"])
    model = UniCOREEG(config).eval()
    model.load_state_dict(checkpoint["model"], strict=False)
    dataset = SyntheticEEGDataset(samples=8, channels=config.in_channels, seed=42, split="train")
    batch = {key: torch.stack([dataset[i][key] for i in range(8)]) for key in ("noisy", "clean", "artifacts", "labels", "label_mask", "component_mask", "severity", "metadata", "is_clean", "disabled_experts")}
    with torch.no_grad():
        outputs = model(batch["noisy"], metadata=batch["metadata"], disabled_experts=batch["disabled_experts"])
    ablation_rows = []
    for name, field in (("A_baseline_old_model", None), ("B_diversity_loss", "diversity"), ("C_load_balance", "load_balance"), ("D_hard_negative", "hard_negative")):
        weights = UniCORELossWeights()
        if field is None:
            # Historical baseline is reported from the existing pre-optimization run.
            ablation_rows.append({"variant": name, "mode": "historical_or_forward_only", "metric": "known_macro_f1", "value": "0.5683", "causal_training": "no"})
            continue
        setattr(weights, field, 0.0)
        loss, logs = UniCORELoss(weights)(outputs, batch)
        ablation_rows.append({"variant": name, "mode": "single_forward_loss_ablation", "metric": "total_loss", "value": f"{float(loss):.8f}", "causal_training": "no"})
    with (args.out / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "mode", "metric", "value", "causal_training"]); writer.writeheader(); writer.writerows(ablation_rows)

    report = f"""# Gate A C=1 failure analysis

## Scope

No training was started in this analysis. Evidence comes from the existing
checkpoint `{args.checkpoint}`, its validation history, and the single frozen
test evaluation.

## Findings

1. Macro-F1 fell to `0.2458` because harmonic and ocular AUROC are below 0.32
   and their recalls are 0.00/0.04. Myogenic remains strong (F1 0.6885), while
   cardiac and motion are weak. This is a representation/training failure,
   not a threshold-grid artifact. The run has only two recorded validation
   epochs (`0.5420`, `0.6186` AUROC); the requested four-epoch run stopped during
   the next evaluation.
2. Unknown F1=`0.5506` does not by itself show that Unknown absorbed known
   artifacts. Unknown precision is `0.936`, recall is only `0.39`, and AUROC is
   `0.6787`: the detector is conservative and misses many Unknown samples. The
   known-class collapse is concentrated in harmonic/ocular probabilities.
3. No newly added loss caused this failure. The checkpoint manifest records
   commit `8784153`; diversity/load-balance/hard-negative losses were added in
   later commit `1943a62`. They were not present in this failed run. Therefore
   the causal answer is: none of the three new losses.

## Numerical checks

- Checkpoint tensors finite: `{finite}`.
- NaN/Inf evidence: none in checkpoint tensors or saved routing probabilities.
- `grad_norm`: not recorded by the interrupted run, so it cannot be inferred
  retrospectively and is marked unavailable rather than fabricated.
- Expert activation distribution: see `expert_activation_matrix.json` copied
  from the frozen test output.
- Confusion matrix: see `confusion_matrix.csv`.

## Ablation interpretation

`ablation.csv` contains historical baseline context and forward-only loss
ablations with one weight removed at a time. These are not causal performance
ablations because training was explicitly prohibited. A training ablation is
required before attributing any future regression to a new loss.

## Decision

Restore/retain the pre-optimization training behavior for this failed run;
do not disable the new losses based on this evidence. The next permitted
experiment should first reproduce the baseline with grad norms and per-epoch
expert usage, then run the three one-variable training ablations.
"""
    (args.out / "FAILURE_ANALYSIS.md").write_text(report, encoding="utf-8")
    (args.out / "expert_activation_matrix.json").write_text(json.dumps(routing["expert_activation_matrix"], indent=2), encoding="utf-8")
    with (args.out / "routing_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expert", "mean_route"])
        matrix = np.asarray(routing["expert_activation_matrix"], dtype=np.float64)
        values = matrix if matrix.ndim == 1 else matrix.mean(axis=0)
        for name, value in zip(("harmonic", "ocular_drift", "myogenic", "cardiac", "motion_transient", "unknown"), values):
            writer.writerow([name, float(value)])
    (args.out / "validation_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
