from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .model import ARTIFACT_NAMES


ON004784_TASKS = ("Brain", "Eyes", "Facial", "Neck", "Walking", "All")
ON004784_FAMILIES = ("clean", "ocular", "facial_myogenic", "neck_myogenic", "motion", "mixed")
GT_SOURCE_GROUPS = {
    "brain": tuple(range(0, 10)),
    "ocular": tuple(range(10, 12)),
    "neck_myogenic": tuple(range(12, 16)),
    "facial_myogenic": tuple(range(16, 20)),
    "trigger": (20,),
}


def _prepare_windows_openmp() -> None:
    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass(frozen=True)
class On004784Recording:
    task: str
    eeg_set: Path
    eeg_fdt: Path
    channels_tsv: Path
    labels: tuple[float, ...]
    family: int
    sample_rate: float
    channels: int
    eeg_channel_indices: tuple[int, ...]
    samples: int
    duration: float


@dataclass(frozen=True)
class On004784Window:
    recording_index: int
    start: int
    stop: int


def task_to_labels(task: str) -> tuple[Tensor, int]:
    labels = torch.zeros(len(ARTIFACT_NAMES))
    if task == "Brain":
        return labels, 0
    if task == "Eyes":
        labels[1] = 1.0
        return labels, 1
    if task == "Facial":
        labels[2] = 1.0
        return labels, 2
    if task == "Neck":
        labels[2] = 1.0
        return labels, 3
    if task == "Walking":
        labels[4] = 1.0
        return labels, 4
    if task == "All":
        labels[1] = 1.0
        labels[2] = 1.0
        labels[4] = 1.0
        return labels, 5
    raise ValueError(f"unknown on004784 task: {task}")


def _eeg_channel_indices(channels_tsv: Path) -> tuple[int, ...]:
    _prepare_windows_openmp()
    import pandas as pd

    frame = pd.read_csv(channels_tsv, sep="\t")
    if "type" not in frame:
        return tuple(range(len(frame)))
    indices = [index for index, row in frame.iterrows() if str(row["type"]).upper() == "EEG"]
    return tuple(indices or range(len(frame)))


def discover_on004784_recordings(root: Path | str) -> list[On004784Recording]:
    _prepare_windows_openmp()
    import mne

    root = Path(root)
    eeg_root = root / "sub-001" / "eeg"
    recordings: list[On004784Recording] = []
    for task in ON004784_TASKS:
        prefix = eeg_root / f"sub-001_task-{task}"
        eeg_set = prefix.with_name(prefix.name + "_eeg.set")
        eeg_fdt = prefix.with_name(prefix.name + "_eeg.fdt")
        channels_tsv = prefix.with_name(prefix.name + "_channels.tsv")
        if not eeg_set.exists() or not eeg_fdt.exists() or not channels_tsv.exists():
            raise FileNotFoundError(f"missing on004784 BIDS files for task={task} under {eeg_root}")
        raw = mne.io.read_raw_eeglab(eeg_set, preload=False, verbose="ERROR")
        labels, family = task_to_labels(task)
        recordings.append(On004784Recording(
            task=task,
            eeg_set=eeg_set,
            eeg_fdt=eeg_fdt,
            channels_tsv=channels_tsv,
            labels=tuple(labels.tolist()),
            family=family,
            sample_rate=float(raw.info["sfreq"]),
            channels=len(raw.ch_names),
            eeg_channel_indices=_eeg_channel_indices(channels_tsv),
            samples=int(raw.n_times),
            duration=float(raw.n_times / raw.info["sfreq"]),
        ))
    return recordings


def load_ground_truth_summary(root: Path | str) -> dict[str, object]:
    _prepare_windows_openmp()
    import scipy.io as sio

    root = Path(root)
    gt_path = root / "stimuli" / "GTdata_croppedToRisingEdge.mat"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    matrix = sio.loadmat(gt_path, simplify_cells=True)["GTdata"]
    return {
        "path": str(gt_path),
        "shape": tuple(int(value) for value in matrix.shape),
        "dtype": str(matrix.dtype),
        "source_groups": {name: list(indices) for name, indices in GT_SOURCE_GROUPS.items()},
    }


class On004784WindowDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        tasks: list[str] | None = None,
        channels: int = 1,
        sample_rate: int = 500,
        length: int = 1000,
        stride_seconds: float | None = None,
        max_windows: int | None = None,
        seed: int = 17,
    ) -> None:
        self.root = Path(root)
        self.channels = channels
        self.sample_rate = sample_rate
        self.length = length
        selected_tasks = set(tasks or ON004784_TASKS)
        self.recordings = [record for record in discover_on004784_recordings(self.root) if record.task in selected_tasks]
        if not self.recordings:
            raise FileNotFoundError(f"no selected on004784 recordings found under {self.root}")
        self.windows = self._index_windows(stride_seconds)
        if max_windows is not None and len(self.windows) > max_windows:
            generator = torch.Generator().manual_seed(seed)
            selected = torch.randperm(len(self.windows), generator=generator)[:max_windows].tolist()
            self.windows = [self.windows[index] for index in sorted(selected)]
        self._cached_path: Path | None = None
        self._cached_raw: object | None = None

    def _index_windows(self, stride_seconds: float | None) -> list[On004784Window]:
        windows: list[On004784Window] = []
        for recording_index, recording in enumerate(self.recordings):
            source_length = int(round(self.length * recording.sample_rate / self.sample_rate))
            stride = int(round((stride_seconds or self.length / self.sample_rate) * recording.sample_rate))
            stride = max(stride, 1)
            max_start = max(recording.samples - source_length, 0)
            for start in range(0, max_start + 1, stride):
                windows.append(On004784Window(recording_index, start, start + source_length))
        if not windows:
            raise ValueError("on004784 window index is empty")
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def _raw(self, path: Path) -> object:
        _prepare_windows_openmp()
        import mne

        if path != self._cached_path:
            self._cached_raw = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
            self._cached_path = path
        assert self._cached_raw is not None
        return self._cached_raw

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        window = self.windows[index]
        recording = self.recordings[window.recording_index]
        raw = self._raw(recording.eeg_set)
        picks = list(recording.eeg_channel_indices[: self.channels])
        signal = torch.from_numpy(raw.get_data(picks=picks, start=window.start, stop=window.stop)).float()
        signal = torch.nn.functional.interpolate(signal.unsqueeze(0), size=self.length, mode="linear", align_corners=False).squeeze(0)
        if signal.size(0) < self.channels:
            signal = signal.expand(self.channels, -1)
        center = signal.median(dim=-1, keepdim=True).values
        mad = (signal - center).abs().median(dim=-1, keepdim=True).values.mul(1.4826).clamp_min(1e-8)
        normalized = (signal - center) / mad
        metadata = torch.zeros(8)
        metadata[0] = self.sample_rate / 1000.0
        metadata[1] = self.channels / 64.0
        metadata[2] = recording.sample_rate / 1000.0
        return {
            "eeg": normalized[: self.channels].float(),
            "raw_eeg": signal[: self.channels].float(),
            "labels": torch.tensor(recording.labels, dtype=torch.float32),
            "family": torch.tensor(recording.family, dtype=torch.long),
            "metadata": metadata,
            "task": recording.task,
            "start_sample": torch.tensor(window.start, dtype=torch.long),
        }
