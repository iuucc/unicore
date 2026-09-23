"""在 validation 上冻结路由阈值，并可对冻结后的 test 运行一次诊断。

该脚本不读取或调整 test 指标；``--split val`` 只产生校准结果，传入
``--test-checkpoint`` 时才会在相同 checkpoint 上对 test 做一次最终评估。
"""
from __future__ import annotations

import argparse
import hashlib
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
from unicore_eeg.synthetic import SyntheticEEGDataset


def _ece(prob: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & (prob <= high if high == 1 else prob < high)
        if mask.any():
            result += float(mask.mean()) * abs(float(prob[mask].mean()) - float(target[mask].mean()))
    return result


def _collect(model: UniCOREEG, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels, probs, presence, routes = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch["noisy"], metadata=batch["metadata"],
                    disabled_experts=batch["disabled_experts"], route_mode="learned")
            labels.append(batch["labels"].float().cpu().numpy())
            probs.append(output["probabilities"].float().cpu().numpy())
            presence.append(output["artifact_presence_probability"].float().cpu().numpy())
            routes.append(output["route"].float().cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs), np.concatenate(presence), np.concatenate(routes)


def _metrics(labels: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> tuple[list[dict[str, object]], dict[str, float]]:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    pred = probs >= thresholds[None, :]
    rows: list[dict[str, object]] = []
    for index, name in enumerate(ARTIFACT_NAMES):
        y, p, z = labels[:, index], probs[:, index], pred[:, index]
        tp = float(((z == 1) & (y == 1)).sum()); fp = float(((z == 1) & (y == 0)).sum())
        fn = float(((z == 0) & (y == 1)).sum()); tn = float(((z == 0) & (y == 0)).sum())
        ops = {}
        for limit in (0.05, 0.10):
            feasible = []
            for candidate in np.linspace(0.01, 0.99, 99):
                candidate_pred = p >= candidate
                candidate_fp = float(((candidate_pred == 1) & (y == 0)).sum())
                candidate_tn = float(((candidate_pred == 0) & (y == 0)).sum())
                fpr = candidate_fp / max(candidate_fp + candidate_tn, 1.0)
                if fpr <= limit:
                    feasible.append((float(f1_score(y, candidate_pred, zero_division=0)), float(candidate), fpr))
            ops[f"fpr_le_{limit:.2f}"] = (max(feasible) if feasible else "not feasible")
        rows.append({"class": name, "precision": float(precision_score(y, z, zero_division=0)),
            "recall": float(recall_score(y, z, zero_division=0)), "f1": float(f1_score(y, z, zero_division=0)),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
            "auprc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
            "support": float(y.sum()), "false_positive_rate": fp / max(fp + tn, 1.0),
            "false_negative_rate": fn / max(fn + tp, 1.0), "ece": _ece(p, y),
            "brier": float(np.mean((p - y) ** 2)), "operating_points": ops})
    known = pred[:, :5]; known_y = labels[:, :5]
    known_auroc = [roc_auc_score(labels[:, i], probs[:, i]) for i in range(5) if len(np.unique(labels[:, i])) == 2]
    known_auprc = [average_precision_score(labels[:, i], probs[:, i]) for i in range(5) if labels[:, i].sum()]
    summary = {"known_macro_f1": float(f1_score(known_y, known, average="macro", zero_division=0)),
        "known_macro_auroc": float(np.mean(known_auroc)), "known_macro_auprc": float(np.mean(known_auprc)),
        "six_class_macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, pred, average="micro", zero_division=0)),
        "macro_auroc": float(np.nanmean([row["auroc"] for row in rows])),
        "macro_auprc": float(np.nanmean([row["auprc"] for row in rows]))}
    return rows, summary


def _calibrate(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    from sklearn.metrics import f1_score
    grid = np.linspace(0.05, 0.95, 19)
    return np.asarray([max(grid, key=lambda t: f1_score(labels[:, i], probs[:, i] >= t, zero_division=0)) for i in range(labels.shape[1])])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--test-checkpoint", type=Path)
    parser.add_argument("--thresholds", type=Path, help="validation routing JSON；test 模式必须提供")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--use-public-sources", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--montage")
    parser.add_argument("--montage-channels", nargs="+")
    parser.add_argument("--bypass-source", choices=("presence_head", "max_probability"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing CPU calibration")
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = UniCOREEGConfig(**checkpoint["config"])
    if args.bypass_source:
        model_config.bypass_source = args.bypass_source
    model = UniCOREEG(model_config).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    coords = None
    if args.montage:
        from unicore_eeg.montage import load_montage, resolve_channels
        spec = load_montage(args.montage)
        names = list(args.montage_channels or spec.channels)
        _, coords, _ = resolve_channels(names, spec)
        args.channels = len(names)
    dataset = SyntheticEEGDataset(samples=args.samples, channels=args.channels, seed=args.seed,
        mode="first_experiment", split=args.split, use_public_sources=args.use_public_sources,
        data_root=args.data_root, montage_coords=coords)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_variable_channels)
    labels, probs, presence, routes = _collect(model, loader, device)
    if args.split == "val":
        thresholds = _calibrate(labels, probs)
    else:
        if args.thresholds is None:
            raise SystemExit("test evaluation requires --thresholds from a completed validation calibration")
        frozen = json.loads(args.thresholds.read_text(encoding="utf-8"))["thresholds"]
        thresholds = np.asarray([float(frozen[name]) for name in ARTIFACT_NAMES])
    rows, summary = _metrics(labels, probs, thresholds)
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    payload = {"split": args.split, "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "git_sha": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv, "data_config": {"use_public_sources": args.use_public_sources, "data_root": str(args.data_root), "montage": args.montage, "montage_channels": args.montage_channels},
        "seed": args.seed, "samples": args.samples, "bypass_source": model_config.bypass_source,
        "thresholds": dict(zip(ARTIFACT_NAMES, thresholds.tolist())),
        "metrics": rows, "summary": summary, "artifact_presence_mean": float(presence.mean()),
        "source_split_manifest": dataset.public.split_manifest() if dataset.public else None}
    payload["expert_activation_matrix"] = routes.mean(axis=0).tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"routing_{args.split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
