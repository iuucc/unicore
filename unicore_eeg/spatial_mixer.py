"""真空间混音（手册 T1.4、§6.5）。

合成数据的"多通道"必须是**真混音**：每个伪迹先作为独立源 ``s_k(t)`` 生成，
再按该伪迹的空间先验乘上混音矩阵 ``A ∈ R^{C×K}``、加上逐通道传播延迟，
最后与清洁信号相加。

被替换掉的旧做法（``synthetic.py::_expand_channels``）只是对同一个源做
"缩放 + 时移"后堆叠，通道之间没有真正的空间结构，空间投影模块在这种数据上
学不到任何东西。

各伪迹的空间先验（手册 §6.5）：

===============  ==========================================================
harmonic         稀疏随机，2–4 个通道非零（模拟参考/局部干扰）
ocular_drift     前额优势：以最靠前的电极为中心做高斯衰减
myogenic         局域：随机选 1 个中心通道，按距离高斯衰减
cardiac          弥散 + 全通道近同相（各通道符号一致，低秩）
motion_transient 全通道，幅度逐通道随机
unknown          随机稀疏
===============  ==========================================================

帧约定
------
先验需要知道"前"和"上"是哪个方向。参考坐标表用 MNE head RAS
（x=右、y=前、z=上）；ds004784 的体模坐标系不同（x=前、z=上），
由 montage 配置的 ``frame_axes`` 声明。``coords=None`` 时用确定性的
半球螺旋布局，只在纯合成场景下使用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

__all__ = [
    "ARTIFACT_MIXING_KINDS",
    "DEFAULT_FRAME_AXES",
    "HARMONIC_ACTIVE_RANGE",
    "OCULAR_FLOOR",
    "OCULAR_SIGMA",
    "SpatialMixer",
    "build_mixing_matrix",
    "canonical_head_layout",
    "resolve_frame_axes",
]

#: 与 ``model.ARTIFACT_NAMES`` 顺序一致（这里不 import，避免模块间循环依赖，
#: 由测试断言两者相同）。
ARTIFACT_MIXING_KINDS: tuple[str, ...] = (
    "harmonic",
    "ocular_drift",
    "myogenic",
    "cardiac",
    "motion_transient",
    "unknown",
)

#: harmonic 先验的非零通道数范围（手册 §6.5：2–4 个）。
HARMONIC_ACTIVE_RANGE = (2, 4)

#: ocular 先验的高斯宽度（单位：头部半径）与远端下界。
#: 取值理由见 ``build_mixing_matrix`` 里 ocular 分支的注释：
#: 纯高斯 σ=0.35 会把枕区压到 1e-10 量级（等于该族在后部不存在），
#: 而下界取太大会把远端压平、丢掉空间梯度。σ=0.9 + floor=0.02 给出
#: 前额 1.0 / 中央约 0.42 / 枕区约 0.06 的分布，与 EOG 的容积传导量级接近。
OCULAR_SIGMA = 0.9
OCULAR_FLOOR = 0.02

#: 默认帧：MNE head RAS。
DEFAULT_FRAME_AXES: dict[str, tuple[float, float, float]] = {
    "up": (0.0, 0.0, 1.0),
    "anterior": (0.0, 1.0, 0.0),
}


def resolve_frame_axes(frame: Mapping[str, Sequence[float]] | None) -> dict[str, Tensor]:
    """把 montage 配置里的 ``frame_axes`` 归一化成两个单位向量。"""
    axes: dict[str, Tensor] = {}
    for key, default in DEFAULT_FRAME_AXES.items():
        value = (frame or {}).get(key, default)
        vector = torch.tensor([float(v) for v in value], dtype=torch.float32)
        if vector.numel() != 3:
            raise ValueError(f"frame_axes[{key!r}] 需要 3 个分量，得到 {vector.numel()} 个")
        axes[key] = vector / vector.norm().clamp_min(1e-8)
    return axes


def canonical_head_layout(channels: int) -> Tensor:
    """确定性的半球螺旋布局，用于 ``coords=None`` 的纯合成场景。

    用 Fibonacci 螺旋在上半球（``z ≥ 0``）均匀撒点，再归一化到单位球面。
    它没有真实的解剖含义，只是让"前额优势""局域肌电"这类先验在一个
    稳定的几何下成立；真实数据集必须传 montage 坐标。
    """
    if channels < 1:
        raise ValueError("channels 必须为正")
    golden = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for index in range(channels):
        # z ∈ (0, 1]，避免落到极点导致方位角无定义
        z = 1.0 - (index + 0.5) / channels * 0.98 - 0.01
        radius = math.sqrt(max(1.0 - z * z, 0.0))
        azimuth = golden * index
        points.append((radius * math.cos(azimuth), radius * math.sin(azimuth), z))
    return torch.tensor(points, dtype=torch.float32)


@dataclass
class SpatialMixer:
    """按伪迹空间先验生成混音矩阵与传播延迟。

    ``coords`` 形状 ``(C, 3)``；``mask`` 为 ``None`` 时视作全部有效。
    坐标单位是头部半径，与 :mod:`unicore_eeg.montage` 保持一致。
    """

    coords: Tensor
    sample_rate: int = 500
    mask: Tensor | None = None
    frame_axes: Mapping[str, Sequence[float]] | None = None

    def __post_init__(self) -> None:
        if self.coords.dim() != 2 or self.coords.size(-1) != 3:
            raise ValueError(f"coords 需要 (C, 3)，得到 {tuple(self.coords.shape)}")
        self.coords = self.coords.float()
        if self.mask is None:
            self.mask = torch.ones(self.coords.size(0), dtype=torch.bool)
        else:
            self.mask = self.mask.bool()
        self.axes = resolve_frame_axes(self.frame_axes)
        # 有效通道的"前"向分量，用于前额优势先验
        anterior = self.coords @ self.axes["anterior"]
        valid = self.mask
        if bool(valid.any()):
            self._anterior = torch.where(valid, anterior, anterior.min() - 1.0)
            self._frontal_index = int(torch.where(valid, anterior, anterior.min() - 1.0).argmax())
            self._posterior_index = int(
                torch.where(valid, anterior, anterior.max() + 1.0).argmin()
            )
        else:  # pragma: no cover - 全部通道无坐标时退化为通道 0
            self._anterior = torch.zeros_like(anterior)
            self._frontal_index = 0
            self._posterior_index = 0

    # ------------------------------------------------------------------
    @property
    def channels(self) -> int:
        return self.coords.size(0)

    @property
    def frontal_index(self) -> int:
        return self._frontal_index

    @property
    def posterior_index(self) -> int:
        return self._posterior_index

    def _distance_to(self, index: int) -> Tensor:
        return (self.coords - self.coords[index][None, :]).norm(dim=-1)

    @staticmethod
    def _jitter(generator: torch.Generator, low: float, high: float, size: int) -> Tensor:
        return low + (high - low) * torch.rand(size, generator=generator)

    # ------------------------------------------------------------------
    def build_mixing_matrix(self, kind: str, generator: torch.Generator) -> Tensor:
        """按伪迹类型生成 ``A[:, k]``，形状 ``(C,)``。"""
        channels = self.channels
        if kind == "harmonic":
            weights = torch.zeros(channels)
            low, high = HARMONIC_ACTIVE_RANGE
            count = int(torch.randint(low, high + 1, (), generator=generator))
            count = min(count, channels)
            chosen = torch.randperm(channels, generator=generator)[:count]
            sign = torch.where(
                torch.rand(count, generator=generator) < 0.5,
                -torch.ones(count),
                torch.ones(count),
            )
            weights[chosen] = sign * self._jitter(generator, 0.5, 1.0, count)
        elif kind == "ocular_drift":
            # 前额优势：以最靠前的电极为中心做高斯衰减，但留一个很小的下界。
            # 为什么需要下界：Fp 系列到枕区约 2.3 个头半径，纯高斯 σ=0.35 会让
            # exp(-0.5·(2.3/0.35)²) ≈ 4e-10 —— 枕区实际上完全收不到眼动分量，
            # 于是"按空间分布去除眼动"这个任务在后部退化成"什么都不用做"，
            # 阶段 4 的空间一致性指标也会因此虚高。真实的容积传导在枕区
            # 仍保留几个百分点的幅度，σ=0.9 加 0.02 的下界能同时满足
            # "前额显著占优"与"远端非零且保留梯度"两点。
            distance = self._distance_to(self._frontal_index)
            falloff = torch.exp(-0.5 * (distance / OCULAR_SIGMA).square())
            weights = OCULAR_FLOOR + (1.0 - OCULAR_FLOOR) * falloff
            weights = weights * self._jitter(generator, 0.85, 1.15, channels)
        elif kind == "myogenic":
            # 局域：随机中心通道 + 高斯局域衰减
            centre = int(torch.randint(0, channels, (), generator=generator))
            sigma = float(self._jitter(generator, 0.15, 0.35, 1))
            weights = torch.exp(-0.5 * (self._distance_to(centre) / sigma).square())
            weights = weights * self._jitter(generator, 0.6, 1.0, channels)
        elif kind == "cardiac":
            # 弥散 + 全通道近同相：符号一致、幅度只有小幅波动
            weights = self._jitter(generator, 0.85, 1.15, channels)
        elif kind == "motion_transient":
            weights = self._jitter(generator, 0.3, 1.2, channels)
        elif kind == "unknown":
            weights = torch.zeros(channels)
            count = min(int(torch.randint(3, 9, (), generator=generator)), channels)
            chosen = torch.randperm(channels, generator=generator)[:count]
            weights[chosen] = self._jitter(generator, 0.4, 1.0, count)
        else:
            raise ValueError(f"未知的伪迹类型 {kind!r}")
        # 逐通道独立方差 → 行向量方向固定、幅度带随机性；无效通道（无坐标）置零但保留比例
        weights = weights / weights.abs().max().clamp_min(1e-6)
        return weights

    def build_delays(self, generator: torch.Generator, max_delay_ms: float = 8.0) -> Tensor:
        """传播延迟，单位采样点，形状 ``(C,)``，范围 ``U(-max, +max)`` 毫秒。"""
        limit = max(int(round(max_delay_ms / 1000.0 * self.sample_rate)), 1)
        return torch.randint(-limit, limit + 1, (self.channels,), generator=generator)

    def build_matrix(self, generator: torch.Generator) -> Tensor:
        """全部 6 类伪迹的混音矩阵，形状 ``(C, K)``。"""
        columns = [self.build_mixing_matrix(kind, generator) for kind in ARTIFACT_MIXING_KINDS]
        return torch.stack(columns, dim=-1)

    # ------------------------------------------------------------------
    def _apply_delay(self, source: Tensor, delays: Tensor) -> Tensor:
        """把 ``(T,)`` 的源按 ``(C,)`` 的延迟展开成 ``(C, T)``。"""
        length = source.size(-1)
        index = torch.arange(length, device=source.device)
        shifted = (index[None, :] - delays[:, None]) % length
        return torch.gather(source[None, :].expand(delays.size(0), length), 1, shifted)

    def mix_by_family(
        self,
        sources: Tensor,
        generator: torch.Generator,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """逐族返回混音后的多通道信号。

        ``sources`` 形状 ``(K, T)``。返回 ``(mixed (K, C, T), matrix (C, K))``。
        与 :meth:`mix_sources` 的区别是**不把 6 族求和**——合成器需要对每一族
        单独做 SNR 缩放后再叠加（手册 T1.4 步骤 4），所以必须能拿到单族的
        多通道伪迹。
        """
        if sources.dim() != 2:
            raise ValueError(f"sources 需要 (K, T)，得到 {tuple(sources.shape)}")
        if sources.size(0) != len(ARTIFACT_MIXING_KINDS):
            raise ValueError(
                f"sources 第一维应为 {len(ARTIFACT_MIXING_KINDS)} 类伪迹，得到 {sources.size(0)}"
            )
        matrix = self.build_matrix(generator)  # (C, K)
        mixed = torch.zeros(len(ARTIFACT_MIXING_KINDS), self.channels, sources.size(-1))
        for index in range(len(ARTIFACT_MIXING_KINDS)):
            delays = self.build_delays(generator)
            delayed = self._apply_delay(sources[index], delays)
            mixed[index] = matrix[:, index : index + 1] * delayed
        if mask is not None:
            mixed = mixed * mask.float().view(1, -1, 1)
        return mixed, matrix

    def mix_sources(
        self,
        sources: Tensor,
        generator: torch.Generator,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """按真混音模型合成多通道伪迹（6 族求和）。

        ``sources`` 形状 ``(K, T)``（顺序同 :data:`ARTIFACT_MIXING_KINDS`）。
        返回 ``(mixed (C, T), matrix (C, K))``——矩阵作为真值记录进样本字典。
        """
        mixed, matrix = self.mix_by_family(sources, generator, mask=mask)
        return mixed.sum(dim=0), matrix

    def mix_clean(
        self,
        clean_source: Tensor,
        generator: torch.Generator,
        *,
        gain: float = 1.0,
    ) -> Tensor:
        """清洁源的空间混合：逐通道幅度与相位都不同，不能是同一源复制。

        幅度按枕区优势（alpha 节律）加权，并带逐通道随机增益；
        相位差异由 ``mix_sources`` 同款的有限延迟实现。
        """
        distance = self._distance_to(self._posterior_index)
        prior = torch.exp(-0.5 * (distance / 0.6).square())
        weights = prior * self._jitter(generator, 0.5, 1.2, self.channels)
        delays = self.build_delays(generator)
        return gain * weights[:, None] * self._apply_delay(clean_source, delays)


def build_mixing_matrix(
    kind: str,
    channels: int,
    coords: Tensor | None,
    generator: torch.Generator,
    *,
    sample_rate: int = 500,
    mask: Tensor | None = None,
    frame_axes: Mapping[str, Sequence[float]] | None = None,
) -> Tensor:
    """手册 T1.4 要求的便捷入口：``build_mixing_matrix(kind, C, coords, generator) -> (C,)``。"""
    if kind not in ARTIFACT_MIXING_KINDS:
        raise ValueError(f"未知的伪迹类型 {kind!r}；可用：{ARTIFACT_MIXING_KINDS}")
    if coords is None:
        coords = canonical_head_layout(channels)
    if coords.size(0) != channels:
        raise ValueError(f"coords 有 {coords.size(0)} 个通道，与 channels={channels} 不一致")
    mixer = SpatialMixer(coords, sample_rate=sample_rate, mask=mask, frame_axes=frame_axes)
    return mixer.build_mixing_matrix(kind, generator)
