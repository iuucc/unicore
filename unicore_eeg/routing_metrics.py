from __future__ import annotations

from typing import Any

import numpy as np

from .model import ARTIFACT_NAMES


KNOWN_INDICES = tuple(range(5))
UNKNOWN_INDEX = 5


def expected_label_mask(labels: np.ndarray) -> np.ndarray:
    return np.ones_like(labels, dtype=bool)


def ece(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (probability >= low) & (probability <= high if high == 1 else probability < high)
        if in_bin.any():
            result += float(in_bin.mean()) * abs(float(probability[in_bin].mean()) - float(target[in_bin].mean()))
    return result


def masked_threshold_calibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
    label_mask: np.ndarray | None = None,
    grid: np.ndarray | None = None,
    max_fpr: float | None = 0.10,
) -> np.ndarray:
    from sklearn.metrics import f1_score

    labels = np.asarray(labels, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    mask = expected_label_mask(labels) if label_mask is None else np.asarray(label_mask, dtype=bool)
    thresholds = []
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    for index in range(labels.shape[1]):
        valid = mask[:, index]
        if not valid.any():
            thresholds.append(0.5)
            continue
        y = labels[valid, index]
        p = probabilities[valid, index]
        candidates = []
        for candidate in grid:
            prediction = p >= candidate
            negatives = y == 0
            fp = float((prediction & negatives).sum())
            tn = float((~prediction & negatives).sum())
            fpr = fp / max(fp + tn, 1.0)
            candidates.append((float(candidate), fpr, float(f1_score(y, prediction, zero_division=0))))
        feasible = [row for row in candidates if max_fpr is None or row[1] <= max_fpr]
        pool = feasible if feasible else candidates
        thresholds.append(max(pool, key=lambda row: (row[2], -row[1], row[0]))[0])
    return np.asarray(thresholds, dtype=np.float32)


def masked_routing_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
    label_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    mask = expected_label_mask(labels) if label_mask is None else np.asarray(label_mask, dtype=bool)
    thresholds = (
        np.full(labels.shape[1], 0.5, dtype=np.float32)
        if thresholds is None
        else np.asarray(thresholds, dtype=np.float32)
    )
    rows: list[dict[str, Any]] = []
    micro_targets: list[np.ndarray] = []
    micro_predictions: list[np.ndarray] = []
    for index, name in enumerate(ARTIFACT_NAMES):
        valid = mask[:, index]
        y = labels[valid, index]
        p = probabilities[valid, index]
        z = p >= thresholds[index]
        if valid.any():
            micro_targets.append(y)
            micro_predictions.append(z.astype(np.float32))
        tp = float(((z == 1) & (y == 1)).sum())
        fp = float(((z == 1) & (y == 0)).sum())
        fn = float(((z == 0) & (y == 1)).sum())
        tn = float(((z == 0) & (y == 0)).sum())
        operating_points: dict[str, object] = {}
        for limit in (0.05, 0.10):
            feasible = []
            for candidate in np.linspace(0.01, 0.99, 99):
                candidate_pred = p >= candidate
                candidate_fp = float(((candidate_pred == 1) & (y == 0)).sum())
                candidate_tn = float(((candidate_pred == 0) & (y == 0)).sum())
                fpr = candidate_fp / max(candidate_fp + candidate_tn, 1.0)
                if fpr <= limit:
                    feasible.append(
                        {
                            "threshold": float(candidate),
                            "f1": float(f1_score(y, candidate_pred, zero_division=0)),
                            "fpr": fpr,
                        }
                    )
            operating_points[f"fpr_le_{limit:.2f}"] = (
                max(feasible, key=lambda item: item["f1"]) if feasible else "not feasible"
            )
        rows.append(
            {
                "class": name,
                "valid_samples": int(valid.sum()),
                "masked_samples": int((~valid).sum()),
                "precision": float(precision_score(y, z, zero_division=0)) if valid.any() else float("nan"),
                "recall": float(recall_score(y, z, zero_division=0)) if valid.any() else float("nan"),
                "f1": float(f1_score(y, z, zero_division=0)) if valid.any() else float("nan"),
                "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
                "auprc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
                "support": float(y.sum()) if valid.any() else float("nan"),
                "false_positive_rate": fp / max(fp + tn, 1.0),
                "false_negative_rate": fn / max(fn + tp, 1.0),
                "ece": ece(p, y) if valid.any() else float("nan"),
                "brier": float(np.mean((p - y) ** 2)) if valid.any() else float("nan"),
                "operating_points": operating_points,
            }
        )
    known_rows = rows[:5]
    summary = {
        "known_macro_f1": _nanmean([row["f1"] for row in known_rows]),
        "known_macro_auroc": _nanmean([row["auroc"] for row in known_rows]),
        "known_macro_auprc": _nanmean([row["auprc"] for row in known_rows]),
        "six_class_macro_f1": _nanmean([row["f1"] for row in rows]),
        "macro_auroc": _nanmean([row["auroc"] for row in rows]),
        "macro_auprc": _nanmean([row["auprc"] for row in rows]),
        "unknown_f1": float(rows[UNKNOWN_INDEX]["f1"]),
        "unknown_auroc": float(rows[UNKNOWN_INDEX]["auroc"]),
        "micro_f1": float(
            f1_score(np.concatenate(micro_targets), np.concatenate(micro_predictions), zero_division=0)
        )
        if micro_targets
        else float("nan"),
    }
    return rows, summary


def _nanmean(values: list[object]) -> float:
    numeric = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(numeric)) if np.isfinite(numeric).any() else float("nan")
