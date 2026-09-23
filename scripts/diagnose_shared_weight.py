"""Complete C=1 baseline diagnostic with epoch-level routing evidence.

The training path intentionally disables all post-baseline additions.  The
legacy mode evaluates the preserved pre-shared checkpoint without retraining;
at C=1 the old and shared per-channel encoder shapes are identical.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unicore_eeg import ARTIFACT_NAMES, UniCOREEG, UniCOREEGConfig
from unicore_eeg.batching import collate_variable_channels
from unicore_eeg.losses import UniCORELoss, UniCORELossWeights
from unicore_eeg.routing_metrics import masked_routing_metrics
from unicore_eeg.synthetic import SyntheticEEGDataset


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def class_counts(dataset: SyntheticEEGDataset) -> dict[str, int]:
    counts = np.zeros(len(ARTIFACT_NAMES), dtype=np.int64)
    for index in range(len(dataset)):
        counts += dataset[index]["labels"].numpy().astype(np.int64)
    return {name: int(value) for name, value in zip(ARTIFACT_NAMES, counts)}


@torch.no_grad()
def evaluate(model, loader, device):
    labels, masks, probabilities = [], [], []
    for cpu_batch in loader:
        batch = move(cpu_batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch["noisy"], metadata=batch["metadata"], disabled_experts=batch["disabled_experts"])
        labels.append(batch["labels"].cpu().numpy())
        masks.append(batch["label_mask"].cpu().numpy())
        probabilities.append(output["probabilities"].float().cpu().numpy())
    labels, masks, probabilities = map(np.concatenate, (labels, masks, probabilities))
    rows, summary = masked_routing_metrics(labels, probabilities, label_mask=masks)
    return rows, summary


def train(args):
    device = torch.device("cuda:0")
    config = UniCOREEGConfig(in_channels=1, domain_calibration_enabled=False)
    model = UniCOREEG(config).to(device)
    weights = UniCORELossWeights(diversity=0.0, load_balance=0.0, hard_negative=0.0)
    loss_fn = UniCORELoss(weights)
    train_set = SyntheticEEGDataset(samples=args.train_samples, channels=1, seed=args.seed, split="train", mode="first_experiment")
    val_set = SyntheticEEGDataset(samples=args.val_samples, channels=1, seed=args.seed + 100_000, split="val", mode="first_experiment")
    test_set = SyntheticEEGDataset(samples=args.test_samples, channels=1, seed=args.seed + 1_000_000, split="test", mode="first_experiment")
    loader = lambda dataset, shuffle: DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_variable_channels)
    train_loader, val_loader, test_loader = loader(train_set, True), loader(val_set, False), loader(test_set, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        usage = torch.zeros(6, device=device)
        entropy = 0.0
        score_matrix = torch.zeros(6, 6, device=device)
        grad_norm = 0.0
        for cpu_batch in tqdm(train_loader, desc=f"shared C=1 epoch {epoch}/{args.epochs}"):
            batch = move(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch["noisy"], metadata=batch["metadata"], disabled_experts=batch["disabled_experts"])
                loss, logs = loss_fn(output, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise RuntimeError(f"non-finite grad_norm at epoch={epoch}")
            grad_norm += float(grad)
            optimizer.step()
            for key in ("loss", "final", "artifact", "route", "diversity", "load_balance", "hard_negative"):
                sums[key] = sums.get(key, 0.0) + float(logs[key])
            usage += output["route"].detach().float().mean(dim=0)
            route = output["route"].detach().float().clamp_min(1e-8)
            entropy += float((-route * route.log()).sum(dim=-1).mean())
            score_matrix += torch.bincount(
                (batch["labels"].argmax(dim=-1) * 6 + output["route"].argmax(dim=-1)).flatten(), minlength=36
            ).reshape(6, 6).float()
        steps = max(len(train_loader), 1)
        rows, summary = evaluate(model.eval(), val_loader, device)
        record = {
            "epoch": epoch,
            "total_loss": sums["loss"] / steps,
            "reconstruction_loss": sums["final"] / steps,
            "artifact_classification_loss": sums["route"] / steps,
            "grad_norm": grad_norm / steps,
            "router_entropy": entropy / steps,
            "expert_activation_matrix": (usage / steps).cpu().tolist(),
            "routing_matrix": score_matrix.cpu().tolist(),
            "validation_known_macro_f1": summary["known_macro_f1"],
            "validation_known_macro_auroc": summary["known_macro_auroc"],
            "validation_per_class_f1": {row["class"]: row["f1"] for row in rows},
        }
        history.append(record)
        torch.save({"config": model.config.__dict__, "model": model.state_dict(), "epoch": epoch}, args.out / f"epoch_{epoch:02d}.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    rows, summary = evaluate(model, test_loader, device)
    payload = {"mode": "shared_baseline", "class_counts": class_counts(train_set), "history": history, "test_summary": summary, "test_metrics": rows, "new_loss_weights": {"diversity": 0.0, "load_balance": 0.0, "hard_negative": 0.0}, "domain_calibration": False}
    (args.out / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def legacy_eval(args):
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.legacy_checkpoint, map_location="cpu")
    model = UniCOREEG(UniCOREEGConfig(**checkpoint["config"])).to(device)
    try:
        model.load_state_dict(checkpoint["model"], strict=False).eval()
    except RuntimeError as error:
        payload = {"mode": "legacy_checkpoint", "checkpoint": str(args.legacy_checkpoint), "status": "incompatible_with_current_model", "error": str(error), "note": "The preserved pre-shared encoder checkpoint has a different tokenizer input contract; no fabricated A/B metric is reported."}
        (args.out / "legacy_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    dataset = SyntheticEEGDataset(samples=args.test_samples, channels=1, seed=args.seed + 1_000_000, split="test", mode="first_experiment")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_variable_channels)
    rows, summary = evaluate(model, loader, device)
    payload = {"mode": "legacy_checkpoint", "checkpoint": str(args.legacy_checkpoint), "test_summary": summary, "test_metrics": rows, "note": "C=1 old and shared per-channel encoder shapes are identical; this is historical checkpoint context, not a new old-encoder training run."}
    (args.out / "legacy_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=1200)
    parser.add_argument("--test-samples", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if not args.skip_train:
        train(args)
    if args.legacy_checkpoint:
        legacy_eval(args)


if __name__ == "__main__":
    main()
