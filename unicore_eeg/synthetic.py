"""合成 EEG 数据集：真空间混音版（手册 T1.4）。

与改造前的关键差别：每个伪迹先作为**独立源** ``s_k(t)`` 生成，再经
:class:`~unicore_eeg.spatial_mixer.SpatialMixer` 乘上该伪迹的空间先验混音矩阵、
加上逐通道传播延迟，最后与清洁信号相加（手册 §6.5）。

旧实现（``_expand_channels``）只是对同一个源做"缩放 + 时移"后堆叠，通道之间
没有真正的空间结构，空间投影模块在这种数据上学不到东西。

样本字典新增两个字段：

``mixing_matrix``  形状 ``(6, C)``——**实际施加**到每个伪迹下标上的混音权重（真值）。
                   未激活的族对应全零行。之所以不记录"本次生成的 A 的全部列"：
                   held-out 族的能量会被重定向到下标 5，此时下标 5 的分量来自
                   **另一个族**，照抄 A 的第 5 列会与事实不符，直接污染阶段 4 的空间指标。
``coords``         形状 ``(C, 3)``——本样本使用的电极坐标。手册 T1.4 写的是
                   ``(C, 2)``，这里保留三维：二维投影依赖于坐标系朝向
                   （``project_to_disk`` 假设 y=前），而 ds004784 的体模坐标系不是 RAS，
                   把投影结果固化进数据集会把一个错误假设变成既成事实。
                   模型自己按前两个分量取用（见 ``SpatialProjectionHead``）。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .model import ARTIFACT_NAMES
from .spatial_mixer import SpatialMixer, canonical_head_layout


def _smooth_noise(channels: int, length: int, kernel: int, generator: torch.Generator) -> Tensor:
    noise = torch.randn(channels, length, generator=generator)
    weight = torch.ones(channels, 1, kernel) / kernel
    return torch.nn.functional.conv1d(noise.unsqueeze(0), weight, padding=kernel // 2, groups=channels).squeeze(0)


def _resample_1d(source: Tensor, length: int) -> Tensor:
    return torch.nn.functional.interpolate(source[None, None], size=length, mode="linear", align_corners=False)[0, 0]


class PublicSignalPools:
    """只读公共源池，按确定性 60/20/20 源索引切分。

    ``split`` 不能再把所有非 train 值都当成 test：validation 必须拥有独立
    的 clean/EOG/EMG/ECG 源索引，且与 train/test 没有交集。
    """
    SPLIT_RANGES = {"train": (0.0, 0.6), "val": (0.6, 0.8), "test": (0.8, 1.0)}

    def __init__(self, root: Path | str = "data/raw", split: str = "train", sample_rate: int = 500) -> None:
        if split not in self.SPLIT_RANGES:
            raise ValueError(f"split must be one of {tuple(self.SPLIT_RANGES)}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        eeg_root = self.root / "eegdenoisenet"
        self.clean = self._load_npy(eeg_root / "EEG_all_epochs_512hz.npy")
        self.eog = self._load_npy(eeg_root / "EOG_all_epochs.npy")
        self.emg = self._load_npy(eeg_root / "EMG_all_epochs_512hz.npy")
        self.ecg = self._load_ecg(self.root / "mitdb")

    def source_indices(self, kind: str) -> list[int]:
        """返回该 split 可使用的原始源索引（便于登记和防泄漏测试）。"""
        arrays = {"clean": self.clean, "eog": self.eog, "emg": self.emg, "ecg": self.ecg}
        if kind not in arrays:
            raise KeyError(kind)
        count = len(arrays[kind]) if arrays[kind] is not None else 0
        start_ratio, end_ratio = self.SPLIT_RANGES[self.split]
        low = int(math.floor(count * start_ratio))
        high = int(math.floor(count * end_ratio)) if end_ratio < 1.0 else count
        if count and high <= low:
            high = min(count, low + 1)
        return list(range(low, high))

    def split_manifest(self) -> dict[str, object]:
        return {kind: self.source_indices(kind) for kind in ("clean", "eog", "emg", "ecg")}

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
        indices = self.source_indices("clean" if array is self.clean else "eog" if array is self.eog else "emg")
        if not indices:
            return None
        low, high = min(indices), max(indices) + 1
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
        indices = self.source_indices("ecg")
        if not indices:
            return None
        low, high = min(indices), max(indices) + 1
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
    """合成多通道 EEG：清洁源与 6 类伪迹源都经过真空间混合。"""

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
        montage_coords: Tensor | None = None,
        montage_mask: Tensor | None = None,
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

        if montage_coords is None:
            coords = canonical_head_layout(channels)
        else:
            coords = torch.as_tensor(montage_coords, dtype=torch.float32)
            if coords.dim() != 2 or coords.size(-1) != 3:
                raise ValueError(f"montage_coords 需要 (C, 3)，得到 {tuple(coords.shape)}")
            if coords.size(0) != channels:
                raise ValueError(
                    f"montage_coords 有 {coords.size(0)} 个通道，与 channels={channels} 不一致"
                )
        mask = None if montage_mask is None else torch.as_tensor(montage_mask, dtype=torch.bool)
        if mask is not None and mask.numel() != channels:
            raise ValueError(f"montage_mask 长度 {mask.numel()} 与 channels={channels} 不一致")
        self.mixer = SpatialMixer(coords, sample_rate=sample_rate, mask=mask)

    def __len__(self) -> int:
        return self.samples

    # ------------------------------------------------------------------
    # 源生成：全部返回单通道 (T,) 的"源时程"，混音统一交给 SpatialMixer
    # ------------------------------------------------------------------
    def _clean(self, generator: torch.Generator) -> Tensor:
        if self.public is not None:
            public = self.public.clean_epoch(generator, self.length)
            if public is not None:
                normalized = (public - public.median()) / public.abs().median().clamp_min(1e-4)
                return self.mixer.mix_clean(normalized, generator) * 0.2
        time = torch.arange(self.length).float() / self.sample_rate
        clean = torch.zeros(self.channels, self.length)
        # 两个独立的节律源分别做空间混合后相加：既保留通道间的幅度/相位差异
        # （来自空间混合），又保留通道各自的节律内容（来自独立源），
        # 不是"同一个源复制到所有通道"。
        for _ in range(2):
            source = torch.zeros(self.length)
            for low, high in ((1.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)):
                frequency = low + (high - low) * torch.rand((), generator=generator)
                phase = 2.0 * math.pi * torch.rand((), generator=generator)
                amplitude = 0.05 + 0.20 * torch.rand((), generator=generator)
                source = source + amplitude * torch.sin(2.0 * math.pi * frequency * time + phase)
            source = source + 0.03 * _smooth_noise(1, self.length, 17, generator).squeeze(0)
            clean = clean + self.mixer.mix_clean(source, generator)
        return clean

    def _harmonic_source(self, generator: torch.Generator) -> Tensor:
        time = torch.arange(self.length).float() / self.sample_rate
        base = torch.tensor([10.0, 20.0, 50.0, 60.0])[torch.randint(0, 4, (), generator=generator)]
        drift = (torch.rand((), generator=generator) - 0.5) * 0.8
        phase = 2 * math.pi * (base * time + 0.5 * drift * time.square())
        envelope = 0.7 + 0.3 * torch.sin(2 * math.pi * (0.2 + torch.rand((), generator=generator)) * time)
        source = sum((0.8 / harmonic) * torch.sin(harmonic * phase + 2 * math.pi * torch.rand((), generator=generator)) for harmonic in (1, 2, 3))
        return source * envelope

    def _ocular_source(self, generator: torch.Generator) -> Tensor:
        if self.public is not None:
            public = self.public.eog_epoch(generator, self.length)
            if public is not None:
                return (public - public.median()) / public.std().clamp_min(1e-4)
        time = torch.arange(self.length).float() / self.sample_rate
        source = torch.zeros(self.length)
        for _ in range(int(torch.randint(1, 4, (), generator=generator))):
            center = torch.rand((), generator=generator) * time[-1]
            width = 0.05 + 0.10 * torch.rand((), generator=generator)
            source += torch.exp(-0.5 * ((time - center) / width).square())
        source += 0.5 * torch.sin(2 * math.pi * (0.1 + 0.6 * torch.rand((), generator=generator)) * time)
        return source

    def _myogenic_source(self, generator: torch.Generator) -> Tensor:
        if self.public is not None:
            public = self.public.emg_epoch(generator, self.length)
            if public is not None:
                return (public - public.mean()) / public.std().clamp_min(1e-4)
        high = torch.randn(self.length, generator=generator)
        high -= _smooth_noise(1, self.length, 31, generator).squeeze(0)
        envelope = torch.zeros(self.length)
        time = torch.arange(self.length).float()
        for _ in range(int(torch.randint(2, 6, (), generator=generator))):
            center = torch.randint(0, self.length, (), generator=generator).float()
            width = 20 + 70 * torch.rand((), generator=generator)
            envelope += torch.exp(-0.5 * ((time - center) / width).square())
        return high * envelope.clamp_max(1.5)

    def _cardiac_source(self, generator: torch.Generator) -> Tensor:
        if self.public is not None:
            public = self.public.ecg_epoch(generator, self.length)
            if public is not None:
                return (public - public.median()) / public.std().clamp_min(1e-4)
        time = torch.arange(self.length).float() / self.sample_rate
        source = torch.zeros(self.length)
        period = 60.0 / (55.0 + 45.0 * torch.rand((), generator=generator))
        beat = float(torch.rand((), generator=generator) * period)
        while beat < float(time[-1]) + period:
            source += 1.4 * torch.exp(-0.5 * ((time - beat) / 0.018).square())
            source -= 0.45 * torch.exp(-0.5 * ((time - beat - 0.035) / 0.028).square())
            source += 0.2 * torch.exp(-0.5 * ((time - beat - 0.22) / 0.07).square())
            beat += period * float(0.95 + 0.1 * torch.rand((), generator=generator))
        return source

    def _motion_source(self, generator: torch.Generator) -> Tensor:
        source = torch.zeros(self.length)
        time = torch.arange(self.length).float()
        for _ in range(int(torch.randint(1, 5, (), generator=generator))):
            center = int(torch.randint(0, self.length, (), generator=generator))
            amplitude = (0.5 + 1.5 * torch.rand((), generator=generator)) * (1 if torch.rand((), generator=generator) > 0.5 else -1)
            decay = 15 + 100 * torch.rand((), generator=generator)
            source += amplitude * (time >= center) * torch.exp(-(time - center).clamp_min(0) / decay)
        source += 0.25 * torch.randn(self.length, generator=generator) * (source.abs() > 0.05)
        return source

    def _unknown_source(self, generator: torch.Generator) -> Tensor:
        time = torch.arange(self.length).float() / self.sample_rate
        start = 2.0 + 6.0 * torch.rand((), generator=generator)
        end = 70.0 + 40.0 * torch.rand((), generator=generator)
        chirp_phase = 2 * math.pi * (start * time + 0.5 * (end - start) / max(float(time[-1]), 1e-3) * time.square())
        chirp = torch.sin(chirp_phase)
        packet = (torch.sin(2 * math.pi * (0.6 + torch.rand((), generator=generator)) * time) > 0.3).float()
        quantized = torch.round(4.0 * _smooth_noise(1, self.length, 5, generator).squeeze(0)) / 4.0
        return 0.7 * chirp * packet + 0.3 * quantized

    # ------------------------------------------------------------------
    def _scale_to_snr(self, clean: Tensor, artifact: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        """按目标 SNR 缩放**混音后的多通道**伪迹（手册 T1.4 步骤 4）。

        改造前是对单通道伪迹做缩放再堆叠；现在 ``artifact`` 已经是 ``(C, T)``，
        所以 SNR 是在整段多通道信号上定义的。
        """
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
        disabled = torch.zeros(len(ARTIFACT_NAMES), dtype=torch.bool)
        source_fns = (
            self._harmonic_source,
            self._ocular_source,
            self._myogenic_source,
            self._cardiac_source,
            self._motion_source,
            self._unknown_source,
        )
        active = self._active_families(index, generator)

        if self.mode == "mixed" and torch.rand((), generator=generator) < 0.08:
            active.append(5)
        condition = index % 12 if self.mode == "first_experiment" else None
        held_out = condition - 7 if condition is not None and condition >= 7 else None
        if self.mode == "mixed" and active and torch.rand((), generator=generator) < self.unknown_episode_probability:
            known = [family for family in active if family < 5]
            if known:
                held_out = known[int(torch.randint(0, len(known), (), generator=generator))]

        # 6 个独立源 → 真空间混音 → 逐族的 (C, T) 多通道伪迹
        sources = torch.stack([fn(generator) for fn in source_fns])
        mixed_by_family, matrix = self.mixer.mix_by_family(sources, generator)  # (6, C, T), (C, 6)

        # 一个伪迹下标只允许收到一个源族的能量：held-out 族会被重定向到下标 5，
        # 若同时还有 unknown 源（也写下标 5），该下标就成了两族之和，
        # 无法用一列混音权重表示。这里让二者互斥——语义也更干净：
        # 下标 5 表示"某一族未知伪迹"，不是"两族未知伪迹的叠加"。
        if held_out is not None and 5 in active:
            active = [family for family in active if family != 5]

        artifacts = torch.zeros(len(ARTIFACT_NAMES), self.channels, self.length)
        applied = torch.zeros(len(ARTIFACT_NAMES), self.channels)
        for family in sorted(set(active)):
            scaled, family_severity = self._scale_to_snr(clean, mixed_by_family[family], generator)
            target_family = family
            if family == held_out:
                target_family = 5
                disabled[family] = True
                label_mask[family] = 0.0
            artifacts[target_family] += scaled
            applied[target_family] = matrix[:, family]
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
            # (6, C)：行顺序同 ARTIFACT_NAMES，与 artifacts 的第一维对齐；
            # 记的是"实际施加"的权重，未激活的族为全零行
            "mixing_matrix": applied.float(),
            "coords": self.mixer.coords.clone(),
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
