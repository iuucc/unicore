from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .model import ARTIFACT_NAMES


def _smooth_noise(channels: int, length: int, kernel: int, generator: torch.Generator) -> Tensor:
    noise = torch.randn(channels, length, generator=generator)
    weight = torch.ones(channels, 1, kernel) / kernel
    return torch.nn.functional.conv1d(noise.unsqueeze(0), weight, padding=kernel // 2, groups=channels).squeeze(0)


def _resample_1d(source: Tensor, length: int) -> Tensor:
    return torch.nn.functional.interpolate(source[None, None], size=length, mode="linear", align_corners=False)[0, 0]


class PublicSignalPools:
    def __init__(self, root: Path | str = "data/raw", split: str = "train", sample_rate: int = 500) -> None:
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        eeg_root = self.root / "eegdenoisenet"
        self.clean = self._load_npy(eeg_root / "EEG_all_epochs_512hz.npy")
        self.eog = self._load_npy(eeg_root / "EOG_all_epochs.npy")
        self.emg = self._load_npy(eeg_root / "EMG_all_epochs_512hz.npy")
        self.ecg = self._load_ecg(self.root / "mitdb")

    @staticmethod
    def _load_npy(path: Path) -> np.ndarray | None:
        return np.load(path, mmap_mode="r") if path.exists() else None

    @staticmethod
    def _load_ecg(path: Path) -> list[tuple[np.ndarray, float]]:
        if not path.exists():
            return []
        try:
            import wfdb
        except ImportError:
            return []
        signals = []
        for header in sorted(path.glob("*.hea")):
            try:
                record = wfdb.rdrecord(str(header.with_suffix("")))
                signals.append((np.asarray(record.p_signal[:, 0], dtype=np.float32), float(record.fs)))
            except (ValueError, OSError):
                continue
        return signals

    def _pick(self, array: np.ndarray | None, generator: torch.Generator, length: int) -> Tensor | None:
        if array is None or len(array) == 0:
            return None
        boundary = max(int(len(array) * 0.8), 1)
        low, high = (0, boundary) if self.split == "train" else (boundary, len(array))
        if high <= low:
            low, high = 0, len(array)
        segment_count = max(math.ceil(length / int(array.shape[-1])), 1)
        indices = torch.randint(low, high, (segment_count,), generator=generator).tolist()
        sample = torch.cat([
            torch.as_tensor(np.array(array[index], dtype=np.float32, copy=True)).flatten()
            for index in indices
        ])
        return _resample_1d(sample, length)

    def clean_epoch(self, generator: torch.Generator, length: int) -> Tensor | None:
        return self._pick(self.clean, generator, length)

    def eog_epoch(self, generator: torch.Generator, length: int) -> Tensor | None:
        return self._pick(self.eog, generator, length)

    def emg_epoch(self, generator: torch.Generator, length: int) -> Tensor | None:
        return self._pick(self.emg, generator, length)

    def ecg_epoch(self, generator: torch.Generator, length: int) -> Tensor | None:
        if not self.ecg:
            return None
        boundary = max(int(len(self.ecg) * 0.8), 1)
        low, high = (0, boundary) if self.split == "train" else (boundary, len(self.ecg))
        if high <= low:
            low, high = 0, len(self.ecg)
        record, source_rate = self.ecg[int(torch.randint(low, high, (), generator=generator))]
        source_length = max(int(round(length * source_rate / self.sample_rate)), 1)
        if len(record) <= source_length:
            source = torch.as_tensor(record)
        else:
            start = int(torch.randint(0, len(record) - source_length, (), generator=generator))
            source = torch.as_tensor(record[start : start + source_length])
        return _resample_1d(source.float(), length)


class SyntheticEEGDataset(Dataset):
    def __init__(
        self,
        samples: int = 4096,
        channels: int = 1,
        length: int = 1000,
        sample_rate: int = 500,
        seed: int = 7,
        mode: str = "mixed",
        use_public_sources: bool = False,
        data_root: Path | str = "data/raw",
        split: str = "train",
        unknown_episode_probability: float = 0.15,
    ) -> None:
        if mode not in {"mixed", "first_experiment"}:
            raise ValueError("mode must be 'mixed' or 'first_experiment'")
        self.samples = samples
        self.channels = channels
        self.length = length
        self.sample_rate = sample_rate
        self.seed = seed
        self.mode = mode
        self.unknown_episode_probability = unknown_episode_probability
        self.public = PublicSignalPools(data_root, split=split, sample_rate=sample_rate) if use_public_sources else None

    def __len__(self) -> int:
        return self.samples

    def _expand_channels(self, source: Tensor, generator: torch.Generator) -> Tensor:
        scale = 0.65 + 0.7 * torch.rand(self.channels, 1, generator=generator)
        delay_limit = max(int(0.008 * self.sample_rate), 1)
        return torch.stack([
            torch.roll(source, int(torch.randint(-delay_limit, delay_limit + 1, (), generator=generator)))
            for _ in range(self.channels)
        ]) * scale

    def _clean(self, generator: torch.Generator) -> Tensor:
        if self.public is not None:
            public = self.public.clean_epoch(generator, self.length)
            if public is not None:
                public = (public - public.median()) / public.abs().median().clamp_min(1e-4)
                return self._expand_channels(public, generator) * 0.2
        time = torch.arange(self.length).float() / self.sample_rate
        clean = torch.zeros(self.channels, self.length)
        for channel in range(self.channels):
            for low, high in ((1.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)):
                frequency = low + (high - low) * torch.rand((), generator=generator)
                phase = 2.0 * math.pi * torch.rand((), generator=generator)
                amplitude = 0.05 + 0.20 * torch.rand((), generator=generator)
                clean[channel] += amplitude * torch.sin(2.0 * math.pi * frequency * time + phase)
            clean[channel] += 0.03 * _smooth_noise(1, self.length, 17, generator).squeeze(0)
        return clean

    def _harmonic(self, generator: torch.Generator) -> Tensor:
        time = torch.arange(self.length).float() / self.sample_rate
        base = torch.tensor([10.0, 20.0, 50.0, 60.0])[torch.randint(0, 4, (), generator=generator)]
        drift = (torch.rand((), generator=generator) - 0.5) * 0.8
        phase = 2 * math.pi * (base * time + 0.5 * drift * time.square())
        envelope = 0.7 + 0.3 * torch.sin(2 * math.pi * (0.2 + torch.rand((), generator=generator)) * time)
        source = sum((0.8 / harmonic) * torch.sin(harmonic * phase + 2 * math.pi * torch.rand((), generator=generator)) for harmonic in (1, 2, 3))
        return self._expand_channels(source * envelope, generator)

    def _ocular(self, generator: torch.Generator) -> Tensor:
        if self.public is not None:
            public = self.public.eog_epoch(generator, self.length)
            if public is not None:
                return self._expand_channels((public - public.median()) / public.std().clamp_min(1e-4), generator)
        time = torch.arange(self.length).float() / self.sample_rate
        source = torch.zeros(self.length)
        for _ in range(int(torch.randint(1, 4, (), generator=generator))):
            center = torch.rand((), generator=generator) * time[-1]
            width = 0.05 + 0.10 * torch.rand((), generator=generator)
            source += torch.exp(-0.5 * ((time - center) / width).square())
        source += 0.5 * torch.sin(2 * math.pi * (0.1 + 0.6 * torch.rand((), generator=generator)) * time)
        return self._expand_channels(source, generator)

    def _myogenic(self, generator: torch.Generator) -> Tensor:
        if self.public is not None:
            public = self.public.emg_epoch(generator, self.length)
            if public is not None:
                return self._expand_channels((public - public.mean()) / public.std().clamp_min(1e-4), generator)
        high = torch.randn(self.length, generator=generator)
        high -= _smooth_noise(1, self.length, 31, generator).squeeze(0)
        envelope = torch.zeros(self.length)
        time = torch.arange(self.length).float()
        for _ in range(int(torch.randint(2, 6, (), generator=generator))):
            center = torch.randint(0, self.length, (), generator=generator).float()
            width = 20 + 70 * torch.rand((), generator=generator)
            envelope += torch.exp(-0.5 * ((time - center) / width).square())
        return self._expand_channels(high * envelope.clamp_max(1.5), generator)

    def _cardiac(self, generator: torch.Generator) -> Tensor:
        if self.public is not None:
            public = self.public.ecg_epoch(generator, self.length)
            if public is not None:
                return self._expand_channels((public - public.median()) / public.std().clamp_min(1e-4), generator)
        time = torch.arange(self.length).float() / self.sample_rate
        source = torch.zeros(self.length)
        period = 60.0 / (55.0 + 45.0 * torch.rand((), generator=generator))
        beat = float(torch.rand((), generator=generator) * period)
        while beat < float(time[-1]) + period:
            source += 1.4 * torch.exp(-0.5 * ((time - beat) / 0.018).square())
            source -= 0.45 * torch.exp(-0.5 * ((time - beat - 0.035) / 0.028).square())
            source += 0.2 * torch.exp(-0.5 * ((time - beat - 0.22) / 0.07).square())
            beat += period * float(0.95 + 0.1 * torch.rand((), generator=generator))
        return self._expand_channels(source, generator)

    def _motion(self, generator: torch.Generator) -> Tensor:
        source = torch.zeros(self.length)
        time = torch.arange(self.length).float()
        for _ in range(int(torch.randint(1, 5, (), generator=generator))):
            center = int(torch.randint(0, self.length, (), generator=generator))
            amplitude = (0.5 + 1.5 * torch.rand((), generator=generator)) * (1 if torch.rand((), generator=generator) > 0.5 else -1)
            decay = 15 + 100 * torch.rand((), generator=generator)
            source += amplitude * (time >= center) * torch.exp(-(time - center).clamp_min(0) / decay)
        source += 0.25 * torch.randn(self.length, generator=generator) * (source.abs() > 0.05)
        return self._expand_channels(source, generator)

    def _unknown(self, generator: torch.Generator) -> Tensor:
        time = torch.arange(self.length).float() / self.sample_rate
        start = 2.0 + 6.0 * torch.rand((), generator=generator)
        end = 70.0 + 40.0 * torch.rand((), generator=generator)
        chirp_phase = 2 * math.pi * (start * time + 0.5 * (end - start) / max(float(time[-1]), 1e-3) * time.square())
        chirp = torch.sin(chirp_phase)
        packet = (torch.sin(2 * math.pi * (0.6 + torch.rand((), generator=generator)) * time) > 0.3).float()
        quantized = torch.round(4.0 * _smooth_noise(1, self.length, 5, generator).squeeze(0)) / 4.0
        return self._expand_channels(0.7 * chirp * packet + 0.3 * quantized, generator)

    def _scale_to_snr(self, clean: Tensor, artifact: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        snr = -8.0 + 18.0 * torch.rand((), generator=generator)
        alpha = torch.sqrt(clean.square().mean().clamp_min(1e-8) / (artifact.square().mean().clamp_min(1e-8) * (10.0 ** (snr / 10.0))))
        scaled = alpha * artifact
        severity = scaled.square().mean().sqrt() / clean.square().mean().sqrt().clamp_min(1e-8)
        return scaled, severity

    def _active_families(self, index: int, generator: torch.Generator) -> list[int]:
        if self.mode == "first_experiment":
            condition = index % 12
            if condition == 0:
                return []
            if condition <= 6:
                return [condition - 1]
            return [condition - 7]
        draw = float(torch.rand((), generator=generator))
        active_count = 0 if draw < 0.15 else 1 if draw < 0.45 else 2 if draw < 0.75 else 3 if draw < 0.93 else int(torch.randint(4, 6, (), generator=generator))
        return torch.randperm(5, generator=generator)[:active_count].tolist()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        clean = self._clean(generator)
        labels = torch.zeros(len(ARTIFACT_NAMES))
        label_mask = torch.ones_like(labels)
        component_mask = torch.ones_like(labels)
        severity = torch.zeros_like(labels)
        artifacts = torch.zeros(len(ARTIFACT_NAMES), self.channels, self.length)
        disabled = torch.zeros(len(ARTIFACT_NAMES), dtype=torch.bool)
        artifact_fns = (self._harmonic, self._ocular, self._myogenic, self._cardiac, self._motion, self._unknown)
        active = self._active_families(index, generator)

        if self.mode == "mixed" and torch.rand((), generator=generator) < 0.08:
            active.append(5)
        condition = index % 12 if self.mode == "first_experiment" else None
        held_out = condition - 7 if condition is not None and condition >= 7 else None
        if self.mode == "mixed" and active and torch.rand((), generator=generator) < self.unknown_episode_probability:
            known = [family for family in active if family < 5]
            if known:
                held_out = known[int(torch.randint(0, len(known), (), generator=generator))]

        for family in sorted(set(active)):
            raw = artifact_fns[family](generator)
            scaled, family_severity = self._scale_to_snr(clean, raw, generator)
            target_family = family
            if family == held_out:
                target_family = 5
                disabled[family] = True
                label_mask[family] = 0.0
            artifacts[target_family] += scaled
            labels[target_family] = 1.0
            severity[target_family] = torch.maximum(severity[target_family], family_severity)
            if family == held_out:
                component_mask[family] = 0.0

        noisy = clean + artifacts.sum(dim=0)
        metadata = torch.zeros(8)
        metadata[0] = self.sample_rate / 1000.0
        metadata[1] = self.channels / 64.0
        return {
            "noisy": noisy.float(),
            "clean": clean.float(),
            "artifacts": artifacts.float(),
            "labels": labels.float(),
            "label_mask": label_mask.float(),
            "component_mask": component_mask.float(),
            "severity": severity.float(),
            "is_clean": torch.tensor(float(not active)),
            "disabled_experts": disabled,
            "metadata": metadata,
            "condition": torch.tensor(condition if condition is not None else (0 if not active else active[0] + 1), dtype=torch.long),
            "sample_id": torch.tensor(index, dtype=torch.long),
        }
