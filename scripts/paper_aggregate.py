"""Aggregate multi-seed metrics into the paper_final layout."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="JSON files with scalar metrics")
    parser.add_argument("--out", type=Path, default=Path("runs/paper_final"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["metric", "mean", "std", "n"])
        for key in keys:
            values = np.asarray([float(row[key]) for row in rows if key in row])
            writer.writerow([key, float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0, len(values)])
    (args.out / "README.md").write_text(
        "# Paper final results\n\nGenerated from real run JSON files. Checkpoint and dataset paths are recorded in each run manifest.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
