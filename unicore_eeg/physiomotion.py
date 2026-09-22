from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .model import ARTIFACT_NAMES


PHYSIOMOTION_FAMILIES = ("clean", "ocular", "myogenic", "motion", "mixed")


def _prepare_windows_openmp() -> None:
    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass(frozen=True)
class PhysioMotionRecord:
    edf: Path
    channel: str
    start: float
    stop: float
    labels: tuple[float, ...]
    family: int
    annotation: str


def map_annotation(annotation: str) -> tuple[Tensor, int]:
    annotation = annotation.lower()
    labels = torch.zeros(len(ARTIFACT_NAMES))
    if annotation in {"open_base", "close_base"}:
        return labels, 0
    labels[1] = float("blink" in annotation or "eyem" in annotation)
    labels[2] = float(any(term in annotation for term in ("tongue", "swallow", "chew", "eyebrow")))
    labels[4] = float("headm" in annotation)
    active = int(labels.bool().sum())
    if active > 1:
        family = 4
    elif labels[1]:
        family = 1
    elif labels[2]:
        family = 2
    elif labels[4]:
        family = 3
    else:
        family = 4
        labels[-1] = 1.0
    return labels, family


class PhysioMotionWindowDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        subjects: list[int] | None = None,
        channels: int = 1,
        sample_rate: int = 500,
        length: int = 1000,
        max_windows: int | None = 2000,
        seed: int = 17,
    ) -> None:
        self.root = Path(root)
        self.channels = channels
        self.sample_rate = sample_rate
        self.length = length
        self.records = self._index(subjects or list(range(1, 31)))
        if max_windows is not None and len(self.records) > max_windows:
            generator = torch.Generator().manual_seed(seed)
            selected = torch.randperm(len(self.records), generator=generator)[:max_windows].tolist()
            self.records = [self.records[index] for index in sorted(selected)]
        self._cached_path: Path | None = None
        self._cached_raw: mne.io.BaseRaw | None = None

    def _index(self, subjects: list[int]) -> list[PhysioMotionRecord]:
        _prepare_windows_openmp()
        import pandas as pd

        records = []
        derivative = self.root / "derivatives" / "preprocessed_BIDS"
        annotations = self.root / "derivatives" / "Manual_Annotations"
        for subject in subjects:
            for csv_path in sorted(annotations.glob(f"sub{subject}_run*.csv")):
                run = int(csv_path.stem.split("run")[-1])
                edf = derivative / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-artifact_run-{run:02d}_eeg.edf"
                if not edf.exists():
                    continue
                frame = pd.read_csv(csv_path).drop_duplicates(subset=["start_time", "stop_time", "label"])
                for row in frame.itertuples(index=False):
                    labels, family = map_annotation(str(row.label))
                    records.append(PhysioMotionRecord(
                        edf=edf,
                        channel=str(row.channel),
                        start=float(row.start_time),
                        stop=float(row.stop_time),
                        labels=tuple(labels.tolist()),
                        family=family,
                        annotation=str(row.label),
                    ))
        if not records:
            raise FileNotFoundError(f"no PhysioMotion records found under {self.root}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _raw(self, path: Path) -> object:
        _prepare_windows_openmp()
        import mne

        if path != self._cached_path:
            self._cached_raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            self._cached_path = path
        assert self._cached_raw is not None
        return self._cached_raw

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        raw = self._raw(record.edf)
        duration = self.length / self.sample_rate
        center = 0.5 * (record.start + record.stop)
        start_time = max(center - duration / 2.0, 0.0)
        start = int(round(start_time * raw.info["sfreq"]))
        stop = min(start + int(round(duration * raw.info["sfreq"])), raw.n_times)
        if record.channel != "ALL" and record.channel in raw.ch_names:
            picks = [raw.ch_names.index(record.channel)]
        else:
            picks = list(range(min(self.channels, len(raw.ch_names))))
        signal = torch.from_numpy(raw.get_data(picks=picks, start=start, stop=stop)).float()
        signal = torch.nn.functional.interpolate(signal.unsqueeze(0), size=self.length, mode="linear", align_corners=False).squeeze(0)
        if signal.size(0) < self.channels:
            signal = signal.expand(self.channels, -1)
        return {
            "eeg": signal[: self.channels],
            "labels": torch.tensor(record.labels),
            "family": torch.tensor(record.family, dtype=torch.long),
            "annotation": record.annotation,
        }
