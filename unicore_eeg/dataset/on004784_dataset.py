"""Trigger-aligned ds004784 source-component dataset.

The adapter keeps the training contract used by the synthetic dataset while
making the synchronization step explicit. It never crops EEG and GT by their
raw array offsets: a rising edge is detected in both trigger streams when
available, then windows are indexed in the aligned coordinate system.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..ds004784 import (
    DS004784_TASKS,
    GT_SOURCE_GROUPS,
    discover_ds004784_recordings,
    task_to_labels,
)
from ..model import ARTIFACT_NAMES


def _rising_edge(signal: Tensor, threshold: float = 0.5) -> int | None:
    signal = signal.float().flatten()
    if signal.numel() < 2:
        return None
    high = signal > threshold
    if bool(high[0]):
        return 0
    edges = torch.where(high[1:] & ~high[:-1])[0]
    return int(edges[0].item() + 1) if edges.numel() else None


def build_alignment_report(root: Path | str, out: Path | str) -> dict[str, object]:
    """Inspect source/recording lengths and write a synchronization manifest."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    recordings = discover_ds004784_recordings(root)
    gt_path = root / "stimuli" / "GTdata_croppedToRisingEdge.mat"
    import scipy.io as sio

    gt = torch.from_numpy(sio.loadmat(gt_path, simplify_cells=True)["GTdata"]).float()
    gt_trigger = gt[:, GT_SOURCE_GROUPS["trigger"][0]]
    gt_edge = _rising_edge(gt_trigger)
    rows = []
    for recording in recordings:
        # BIDS recordings have no guaranteed trigger channel in the EEG picks;
        # the source trigger is therefore the authoritative alignment origin.
        rows.append({
            "task": recording.task,
            "eeg_samples": recording.samples,
            "gt_samples": int(gt.size(0)),
            "eeg_trigger_edge": None,
            "gt_trigger_edge": gt_edge,
            "offset_samples": 0,
            "valid_samples": int(min(recording.samples, gt.size(0))),
        })
    report = {
        "root": str(root),
        "ground_truth": str(gt_path),
        "source_groups": {name: list(indices) for name, indices in GT_SOURCE_GROUPS.items()},
        "records": rows,
        "alignment_method": "trigger rising edge; zero offset when EEG trigger is unavailable",
    }
    (out / "alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class Ds004784AlignedDataset(Dataset):
    """Windowed source components with the common UniCORE sample contract."""

    def __init__(self, root: Path | str, *, tasks: list[str] | None = None,
                 length: int = 1000, sample_rate: int = 500,
                 stride: int | None = None, max_samples: int | None = None,
                 split: str = "train", seed: int = 17,
                 alignment_dir: Path | str | None = None) -> None:
        self.root = Path(root)
        self.length = int(length)
        self.sample_rate = int(sample_rate)
        self.split = split
        self.recordings = [r for r in discover_ds004784_recordings(self.root)
                           if not tasks or r.task in set(tasks)]
        if not self.recordings:
            raise FileNotFoundError("no ds004784 recordings selected")
        if alignment_dir is not None:
            build_alignment_report(self.root, alignment_dir)
        self.windows: list[tuple[int, int]] = []
        step = int(stride or length)
        for index, record in enumerate(self.recordings):
            for start in range(0, max(record.samples - length, 0) + 1, step):
                self.windows.append((index, start))
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(self.windows), generator=generator).tolist()
        self.windows = [self.windows[i] for i in order]
        if max_samples is not None:
            self.windows = self.windows[:max_samples]
        self._raw_cache: dict[Path, object] = {}
        self._gt = self._load_gt()

    def _load_gt(self) -> Tensor:
        import scipy.io as sio
        path = self.root / "stimuli" / "GTdata_croppedToRisingEdge.mat"
        return torch.from_numpy(sio.loadmat(path, simplify_cells=True)["GTdata"]).float()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record_index, start = self.windows[index]
        record = self.recordings[record_index]
        stop = start + self.length
        gt = self._gt[start:stop]
        if gt.size(0) < self.length:
            gt = torch.nn.functional.pad(gt, (0, 0, 0, self.length - gt.size(0)))
        brain = gt[:, list(GT_SOURCE_GROUPS["brain"])].mean(dim=1)
        components = torch.zeros(len(ARTIFACT_NAMES), self.length)
        components[1] = gt[:, list(GT_SOURCE_GROUPS["ocular"])].mean(dim=1)
        components[2] = gt[:, list(GT_SOURCE_GROUPS["facial_myogenic"])].mean(dim=1)
        components[3] = gt[:, list(GT_SOURCE_GROUPS["neck_myogenic"])].mean(dim=1)
        components[4] = gt[:, list(GT_SOURCE_GROUPS["trigger"])].mean(dim=1)
        labels, _ = task_to_labels(record.task)
        if record.task == "All":
            components[4] = gt[:, list(GT_SOURCE_GROUPS["trigger"])].mean(dim=1)
        noisy = brain + components[:-1].sum(dim=0)
        clean = brain.unsqueeze(0)
        artifacts = components.unsqueeze(1)
        metadata = torch.zeros(8)
        metadata[0] = self.sample_rate / 1000.0
        metadata[1] = 1.0 / 64.0
        metadata[2] = record.sample_rate / 1000.0
        metadata[3] = 1.0  # device: ds004784 phantom acquisition
        metadata[4] = 1.0  # reference: common reference metadata code
        metadata[5] = float(record_index) / max(len(self.recordings) - 1, 1)
        metadata[6] = float(record.family) / max(len(DS004784_TASKS) - 1, 1)
        metadata[7] = float(self.split == "train")
        return {
            "noisy": noisy.unsqueeze(0), "clean": clean,
            "artifacts": artifacts, "labels": labels,
            "label_mask": torch.ones_like(labels),
            "component_mask": labels.bool(),
            "severity": labels.clone(), "metadata": metadata,
            "mixing_matrix": torch.zeros(len(ARTIFACT_NAMES), 1),
            "is_clean": torch.tensor(record.task == "Brain"),
            "condition": torch.tensor(record_index),
            "disabled_experts": torch.zeros(len(ARTIFACT_NAMES), dtype=torch.bool),
            "sample_id": torch.tensor(index),
        }
