from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unicore_eeg import ARTIFACT_NAMES
from unicore_eeg.routing_metrics import masked_routing_metrics, masked_threshold_calibration
from unicore_eeg.synthetic import SyntheticEEGDataset


def _features(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    flat = signal.reshape(-1)
    spectrum = np.abs(np.fft.rfft(flat))
    freqs = np.fft.rfftfreq(flat.size, d=1.0 / sample_rate)
    bands = []
    for low, high in ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 80)):
        in_band = (freqs >= low) & (freqs < high)
        bands.append(float(np.log1p(spectrum[in_band].mean())) if in_band.any() else 0.0)
    return np.asarray(
        [
            float(flat.mean()),
            float(flat.std()),
            float(np.mean(np.abs(flat))),
            float(np.max(flat) - np.min(flat)),
            float(np.percentile(flat, 95) - np.percentile(flat, 5)),
            *bands,
        ],
        dtype=np.float32,
    )


def _matrix(dataset: SyntheticEEGDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, mask = [], [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        x.append(_features(sample["noisy"].numpy(), dataset.sample_rate))
        y.append(sample["labels"].numpy())
        mask.append(sample["label_mask"].numpy())
    return np.stack(x), np.stack(y), np.stack(mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--sample-rate", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("mixed", "first_experiment"), default="first_experiment")
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression

    train = SyntheticEEGDataset(
        samples=args.train_samples,
        channels=args.channels,
        length=args.length,
        sample_rate=args.sample_rate,
        seed=args.seed,
        mode=args.mode,
        split="train",
    )
    val = SyntheticEEGDataset(
        samples=args.val_samples,
        channels=args.channels,
        length=args.length,
        sample_rate=args.sample_rate,
        seed=args.seed + 100_000,
        mode=args.mode,
        split="val",
    )
    test = SyntheticEEGDataset(
        samples=args.test_samples,
        channels=args.channels,
        length=args.length,
        sample_rate=args.sample_rate,
        seed=args.seed + 1_000_000,
        mode=args.mode,
        split="test",
    )
    x_train, y_train, mask_train = _matrix(train)
    x_val, y_val, mask_val = _matrix(val)
    x_test, y_test, mask_test = _matrix(test)
    val_prob = np.zeros_like(y_val, dtype=np.float32)
    test_prob = np.zeros_like(y_test, dtype=np.float32)
    model_status: dict[str, str] = {}
    for index, name in enumerate(ARTIFACT_NAMES):
        valid = mask_train[:, index].astype(bool)
        y = y_train[valid, index]
        if len(np.unique(y)) < 2:
            val_prob[:, index] = float(y[0]) if len(y) else 0.0
            test_prob[:, index] = float(y[0]) if len(y) else 0.0
            model_status[name] = "constant"
            continue
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
        model.fit(x_train[valid], y)
        val_prob[:, index] = model.predict_proba(x_val)[:, 1]
        test_prob[:, index] = model.predict_proba(x_test)[:, 1]
        model_status[name] = "logistic_regression"
    thresholds = masked_threshold_calibration(y_val, val_prob, mask_val)
    val_rows, val_summary = masked_routing_metrics(y_val, val_prob, thresholds, mask_val)
    test_rows, test_summary = masked_routing_metrics(y_test, test_prob, thresholds, mask_test)
    payload = {
        "command": sys.argv,
        "model_status": model_status,
        "thresholds_from_validation": dict(zip(ARTIFACT_NAMES, thresholds.tolist())),
        "validation": {"metrics": val_rows, "summary": val_summary},
        "test": {"metrics": test_rows, "summary": test_summary},
        "data_config": vars(args) | {"out": str(args.out)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
