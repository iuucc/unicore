"""Fixed 64-sample router/tokenizer overfit diagnostic.

This is a routing-head diagnostic only: reconstruction and residual paths are
kept out of the loss. Metrics are computed with ``label_mask`` so held-out
known classes do not count as ordinary negatives for known-class recognition.
"""
from __future__ import annotations

import argparse
import json
import subprocess
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


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _head_gradients(model: UniCOREEG) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    prob_weight = model.tokenizer.prob.weight.grad
    prob_bias = model.tokenizer.prob.bias.grad
    for index, name in enumerate(ARTIFACT_NAMES[:5]):
        result[f"known_{name}_prob_weight_row"] = (
            None if prob_weight is None else float(prob_weight[index].detach().norm())
        )
        result[f"known_{name}_prob_bias"] = None if prob_bias is None else float(prob_bias[index].detach().abs())
    for name, parameter in model.unknown_detector.named_parameters():
        result[f"unknown_detector.{name}"] = (
            None if parameter.grad is None else float(parameter.grad.detach().norm())
        )
    return result


def _all_finite_nonzero(values: dict[str, float | None]) -> bool:
    return all(value is not None and np.isfinite(value) and value > 0.0 for value in values.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing CPU diagnostic")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    dataset = SyntheticEEGDataset(samples=64, channels=args.channels, seed=args.seed, mode="first_experiment")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_channels,
    )
    batches = [_move(batch, device) for batch in loader]
    model = UniCOREEG(UniCOREEGConfig(in_channels=args.channels)).to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("tokenizer.") or name.startswith("unknown_detector.")
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=5e-3)
    final_loss = float("nan")
    final_head_gradients: dict[str, float | None] = {}

    model.train()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        for batch in batches:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    batch["noisy"],
                    metadata=batch["metadata"],
                    disabled_experts=batch["disabled_experts"],
                    route_mode="learned",
                    enable_residual=False,
                )
                mask = batch["label_mask"]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output["probability_logits"].float(),
                    batch["labels"],
                    weight=mask,
                    reduction="sum",
                ) / mask.sum().clamp_min(1.0)
            total_loss = total_loss + loss
            loss.backward()
        final_head_gradients = _head_gradients(model)
        optimizer.step()
        final_loss = float(total_loss.detach())

    model.eval()
    outputs = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in batches:
            outputs.append(
                model(
                    batch["noisy"],
                    metadata=batch["metadata"],
                    disabled_experts=batch["disabled_experts"],
                    route_mode="learned",
                    enable_residual=False,
                )
            )
    labels = torch.cat([batch["labels"] for batch in batches]).float().cpu().numpy()
    label_mask = torch.cat([batch["label_mask"] for batch in batches]).float().cpu().numpy()
    disabled = torch.cat([batch["disabled_experts"] for batch in batches]).cpu().numpy()
    sample_ids = torch.cat([batch["sample_id"] for batch in batches]).cpu().numpy()
    conditions = torch.cat([batch["condition"] for batch in batches]).cpu().numpy()
    probabilities = torch.cat([output["probabilities"] for output in outputs]).float().cpu().numpy()
    logits = torch.cat([output["probability_logits"] for output in outputs]).float().cpu().numpy()

    rows_05, summary_05 = masked_routing_metrics(labels, probabilities, label_mask=label_mask)
    calibrated_thresholds = masked_threshold_calibration(labels, probabilities, label_mask, max_fpr=0.10)
    rows_calibrated, summary_calibrated = masked_routing_metrics(
        labels, probabilities, calibrated_thresholds, label_mask
    )
    logit_margins = []
    for index, name in enumerate(ARTIFACT_NAMES):
        valid = label_mask[:, index].astype(bool)
        y = labels[valid, index]
        class_logits = logits[valid, index]
        positive = class_logits[y == 1]
        negative = class_logits[y == 0]
        logit_margins.append(
            {
                "class": name,
                "positive_logit_mean": float(positive.mean()) if len(positive) else None,
                "negative_logit_mean": float(negative.mean()) if len(negative) else None,
                "logit_margin": float(positive.mean() - negative.mean()) if len(positive) and len(negative) else None,
            }
        )

    payload = {
        "channels": args.channels,
        "samples": 64,
        "steps": args.steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "loss": final_loss,
        "git_sha": _git_sha(),
        "command": sys.argv,
        "metrics_at_0_5": rows_05,
        "summary_at_0_5": summary_05,
        "calibrated_thresholds": dict(zip(ARTIFACT_NAMES, calibrated_thresholds.tolist())),
        "metrics_calibrated_on_same_64": rows_calibrated,
        "summary_calibrated_on_same_64": summary_calibrated,
        "logit_margins": logit_margins,
        "head_gradient_norms": final_head_gradients,
        "all_six_heads_finite_nonzero_gradients": _all_finite_nonzero(final_head_gradients),
        "diagnostic_gate": {
            "known_macro_auroc_ge_0_95": bool(summary_calibrated["known_macro_auroc"] >= 0.95),
            "known_macro_f1_ge_0_80": bool(summary_calibrated["known_macro_f1"] >= 0.80),
            "each_known_auroc_ge_0_85": bool(all(row["auroc"] >= 0.85 for row in rows_calibrated[:5])),
            "all_six_heads_finite_nonzero_gradients": _all_finite_nonzero(final_head_gradients),
            "unknown_f1_ge_0_95": bool(summary_calibrated["unknown_f1"] >= 0.95),
            "passed": bool(
                summary_calibrated["known_macro_auroc"] >= 0.95
                and summary_calibrated["known_macro_f1"] >= 0.80
                and all(row["auroc"] >= 0.85 for row in rows_calibrated[:5])
                and _all_finite_nonzero(final_head_gradients)
                and summary_calibrated["unknown_f1"] >= 0.95
            ),
        },
        "per_sample": {
            "sample_id": sample_ids.tolist(),
            "condition": conditions.tolist(),
            "labels": labels.tolist(),
            "label_mask": label_mask.tolist(),
            "probabilities": probabilities.tolist(),
            "logits": logits.tolist(),
            "disabled_experts": disabled.tolist(),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"router_overfit_c{args.channels}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
