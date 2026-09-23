from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unicore_eeg import ARTIFACT_NAMES
from unicore_eeg.synthetic import PublicSignalPools, SyntheticEEGDataset


def _split_source_manifest(root: Path, sample_rate: int) -> dict[str, object]:
    manifests = {
        split: PublicSignalPools(root=root, split=split, sample_rate=sample_rate).split_manifest()
        for split in ("train", "val", "test")
    }
    intersections: dict[str, dict[str, int]] = {}
    for kind in ("clean", "eog", "emg", "ecg"):
        train = set(manifests["train"][kind])
        val = set(manifests["val"][kind])
        test = set(manifests["test"][kind])
        intersections[kind] = {
            "train_val": len(train & val),
            "train_test": len(train & test),
            "val_test": len(val & test),
        }
    return {"available_source_indices": manifests, "source_intersections": intersections}


def _audit_split(args: argparse.Namespace, split: str, seed_offset: int) -> dict[str, object]:
    dataset = SyntheticEEGDataset(
        samples=args.samples,
        channels=args.channels,
        length=args.length,
        sample_rate=args.sample_rate,
        seed=args.seed + seed_offset,
        mode=args.mode,
        use_public_sources=args.use_public_sources,
        data_root=args.data_root,
        split=split,
    )
    positives = np.zeros(len(ARTIFACT_NAMES), dtype=np.int64)
    masked = np.zeros(len(ARTIFACT_NAMES), dtype=np.int64)
    clean = 0
    unknown = 0
    single = 0
    compound = 0
    conditions: Counter[int] = Counter()
    for index in range(len(dataset)):
        sample = dataset[index]
        labels = sample["labels"].numpy()
        label_mask = sample["label_mask"].numpy()
        positives += labels.astype(np.int64)
        masked += (label_mask == 0).astype(np.int64)
        known_count = int(labels[:5].sum())
        clean += int(labels.sum() == 0)
        unknown += int(labels[5] == 1)
        single += int(known_count == 1 and labels[5] == 0)
        compound += int(known_count + int(labels[5] == 1) > 1)
        conditions[int(sample["condition"])] += 1
    payload: dict[str, object] = {
        "samples": len(dataset),
        "seed": args.seed + seed_offset,
        "positive_by_class": dict(zip(ARTIFACT_NAMES, positives.tolist())),
        "masked_by_class": dict(zip(ARTIFACT_NAMES, masked.tolist())),
        "clean": clean,
        "single_artifact": single,
        "compound_artifact": compound,
        "unknown": unknown,
        "condition_counts": {str(key): value for key, value in sorted(conditions.items())},
    }
    if dataset.public is not None:
        payload["available_source_indices_for_split"] = dataset.public.split_manifest()
    else:
        payload["available_source_indices_for_split"] = None
        payload["used_source_indices"] = None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--sample-rate", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("mixed", "first_experiment"), default="first_experiment")
    parser.add_argument("--use-public-sources", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    payload: dict[str, object] = {
        "command": sys.argv,
        "data_config": {
            "samples": args.samples,
            "channels": args.channels,
            "length": args.length,
            "sample_rate": args.sample_rate,
            "mode": args.mode,
            "use_public_sources": args.use_public_sources,
            "data_root": str(args.data_root),
        },
        "splits": {
            "train": _audit_split(args, "train", 0),
            "val": _audit_split(args, "val", 100_000),
            "test": _audit_split(args, "test", 1_000_000),
        },
    }
    if args.use_public_sources:
        payload["public_source_split_check"] = _split_source_manifest(args.data_root, args.sample_rate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
