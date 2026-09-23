from __future__ import annotations

import contextlib
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ARTIFACT_NAMES = (
    "harmonic",
    "ocular_drift",
    "myogenic",
    "cardiac",
    "motion_transient",
    "unknown",
)


@dataclass
class UniCOREEGConfig:
    # Kept for config/checkpoint compatibility. The current architecture shares
    # all learned channel operations, so this value does not size any parameter.
    in_channels: int = 1
    sample_rate: int = 500
    window_size: int = 1000
    base_channels: int = 48
    token_dim: int = 64
    artifact_count: int = 6
    metadata_dim: int = 8
    max_active_experts: int = 4
    probability_threshold: float = 0.35
    probability_thresholds: tuple[float, ...] | list[float] | None = None
    clean_threshold: float = 0.50
    bypass_source: str = "presence_head"
    router_temperature: float = 1.0
    domain_calibration_enabled: bool = False
    domain_calibration_hidden: int = 32
    eps: float = 1e-5
    # ---- 空间投影（T1.3）----
    use_spatial: bool = True
    spatial_pca_rank: int = 4
    spatial_modulation_alpha_scale: float = 1.0
    spatial_coord_dim: int = 2
    spatial_power_iterations: int = 4
    spatial_hidden: int = 32
    spatial_pca_mode: str = "power"
    #: "raw"：直接取坐标的前 coord_dim 个分量（对任何坐标系都成立）；
    #: "disk"：用 montage 的等距方位投影压到头部圆盘（需要坐标系的"上"轴已知）。
    spatial_coord_space: str = "raw"

    def __post_init__(self) -> None:
        if self.artifact_count != len(ARTIFACT_NAMES):
            raise ValueError(f"artifact_count must be {len(ARTIFACT_NAMES)}")
        if self.bypass_source not in {"presence_head", "max_probability"}:
            raise ValueError("bypass_source must be 'presence_head' or 'max_probability'")
        if self.probability_thresholds is not None:
            values = tuple(float(value) for value in self.probability_thresholds)
            if len(values) != len(ARTIFACT_NAMES):
                raise ValueError(f"probability_thresholds must have {len(ARTIFACT_NAMES)} values")
            self.probability_thresholds = values


def _groups(channels: int) -> int:
    for group_count in (16, 8, 4, 2):
        if channels % group_count == 0:
            return group_count
    return 1


def _moving_average(x: Tensor, kernel_size: int) -> Tensor:
    kernel_size = min(kernel_size, x.size(-1) - (1 - x.size(-1) % 2))
    kernel_size = max(kernel_size, 1)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    return F.avg_pool1d(x, kernel_size, stride=1, padding=kernel_size // 2)


@contextlib.contextmanager
def no_autocast(y: Tensor):
    """在观测抽取里**关掉 autocast**。

    只把张量转成 float32 是不够的：``@`` 是 autocast 的降精度算子，
    即使输入已经是 float32，autocast 也会把它降回 bf16，随后
    ``torch.linalg.solve`` 就会报
    ``NotImplementedError: "lu_factor_cublas" not implemented for 'BFloat16'``
    （CPU 后端是 ``lu_cpu``）。所以必须显式退出 autocast，而不是改 dtype。
    """
    with torch.autocast(device_type=y.device.type, enabled=False):
        yield


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SharedStem(nn.Module):
    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # C is folded into the batch dimension before entering the stem.
            ConvGNAct(1, 32, 7),
            ConvGNAct(32, 32, 15, groups=32),
            ConvGNAct(32, base_channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MSBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.short = ConvGNAct(channels, channels, 3)
        self.medium = ConvGNAct(channels, channels, 7)
        self.long = ConvGNAct(channels, channels, 15, dilation=dilation)
        self.out = nn.Conv1d(channels * 3, channels, 1)
        self.norm = nn.GroupNorm(_groups(channels), channels)

    def forward(self, x: Tensor) -> Tensor:
        y = torch.cat((self.short(x), self.medium(x), self.long(x)), dim=1)
        return F.silu(self.norm(x + self.out(y)))


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, token_dim: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = MSBlock(channels, dilation)
        self.to_gamma_beta = nn.Linear(token_dim, channels * 2)

    def forward(self, x: Tensor, token: Tensor) -> Tensor:
        gamma, beta = self.to_gamma_beta(token).chunk(2, dim=-1)
        return self.block(x) * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)


class ArtifactObservationExtractor(nn.Module):
    """把原始波形拆成 6 类"伪迹观测"，供各专家使用（T1.2 多通道化）。

    输入 ``y: (B, C, T)``，输出 ``list[Tensor]`` 长度 6，每个 ``(B, C, T)``，
    顺序与 :data:`ARTIFACT_NAMES` 一致：

    =====  ==================  ==============================================
    序号   专家                观测口径
    =====  ==================  ==============================================
    0      harmonic            窄带正弦基拟合；**频率估计共享**（谱在全通道上平均后找峰），
                              **幅相系数逐通道独立求解**，输出逐通道拟合曲线
    1      ocular_drift        慢趋势（逐通道移动平均）
    2      myogenic            高频分量（去局部均值）
    3      cardiac             准周期脉冲；**逐通道自相关寻峰**得到各自的滞后
    4      motion_transient    一阶/二阶差分组合
    5      unknown             已知解释后的残差
    =====  ==================  ==============================================

    改造前的实现有两处把多通道压成单通道（``y.mean(dim=1)`` 后
    ``expand_as(y)``），会让所有导联拿到完全相同的谐波基与心电滞后，
    空间信息在进入专家之前就被抹掉了。这里全部改为逐通道。
    """

    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        self.sample_rate = sample_rate

    def peak_frequency(self, y: Tensor) -> Tensor:
        """对全通道平均后的谱找峰（8–80 Hz），返回 ``(B,)``。

        频率是共享的：同一次记录里的线噪基频对所有导联相同，只有幅相不同。
        """
        work = y.float()
        length = work.size(-1)
        with no_autocast(y):
            spectrum = torch.fft.rfft(work, dim=-1).abs().mean(dim=1)
        frequencies = torch.fft.rfftfreq(length, 1.0 / self.sample_rate).to(work.device)
        valid = (frequencies >= 8.0) & (frequencies <= min(80.0, self.sample_rate / 2.0 - 1.0))
        masked = spectrum.masked_fill(~valid.unsqueeze(0), -1.0)
        return frequencies[masked.argmax(dim=-1)]

    def _harmonic_basis(self, y: Tensor) -> Tensor:
        """逐通道窄带拟合，返回 ``(B, C, T)``（dtype 与输入一致）。

        频率仍然只估一次（对通道平均后的谱找峰，8–80 Hz）——同一个记录里的
        线噪基频是共同的；但幅相系数按通道各解各的，保留导联间的位置差异。
        """
        work = y.float()
        length = work.size(-1)
        with no_autocast(y):
            peak_frequency = self.peak_frequency(work)
            time = torch.arange(length, device=work.device, dtype=torch.float32) / self.sample_rate
            basis = []
            for harmonic in (1.0, 2.0, 3.0):
                phase = 2.0 * math.pi * harmonic * peak_frequency[:, None] * time[None]
                basis.extend((torch.sin(phase), torch.cos(phase)))
            design = torch.stack(basis, dim=-1)  # (B, T, 6)

        # 逐通道最小二乘解析求解。用带极小岭项的正规方程解 (6,6) 线性系统，
        # 比 torch.linalg.lstsq 快得多，且在基函数退化（峰频落到 0）时不会发散；
        # 岭项 1e-6 相对设计矩阵的 O(1) 量级可忽略，数值上等价于无正则最小二乘。
        with no_autocast(y):
            signal = work.transpose(1, 2)  # (B, T, C)
            gram = design.transpose(1, 2) @ design  # (B, 6, 6)
            eye = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype) * 1e-6
            coefficients = torch.linalg.solve(gram + eye, design.transpose(1, 2) @ signal)
            fitted = (design @ coefficients).transpose(1, 2)  # (B, C, T)
        return fitted.to(y.dtype)

    def cardiac_lags(self, y: Tensor) -> Tensor:
        """逐通道自相关寻峰得到的滞后（采样点），形状 ``(B, C)``。

        各导联的心电到达时间与形态不同，共享一个滞后会丢掉这个差异。
        """
        work = y.float()
        with no_autocast(y):
            autocorrelation = torch.fft.irfft(
                torch.fft.rfft(work, dim=-1).abs().square(), n=work.size(-1), dim=-1
            )
        lag_start = max(int(0.4 * self.sample_rate), 1)
        lag_end = min(int(1.5 * self.sample_rate), work.size(-1) - 1)
        if lag_end <= lag_start:  # 窗口过短时退化为全窗搜索，避免空切片
            lag_start, lag_end = 1, work.size(-1) - 1
        return autocorrelation[..., lag_start:lag_end].argmax(dim=-1) + lag_start

    def _cardiac(self, y: Tensor, lags: Tensor | None = None) -> Tensor:
        """逐通道构造准周期脉冲，返回 ``(B, C, T)``（dtype 与输入一致）。"""
        work = y.float()
        if lags is None:
            lags = self.cardiac_lags(work)
        length = work.size(-1)
        index = torch.arange(length, device=work.device)
        # roll(sample, +lag) 等价于在 i 处取 sample[i - lag]；逐通道滞后用 gather 一次算完，
        # 不用对 batch 做 Python 循环。
        minus = (index[None, None, :] - lags[:, :, None]) % length
        plus = (index[None, None, :] + lags[:, :, None]) % length
        periodic = (work + torch.gather(work, 2, minus) + torch.gather(work, 2, plus)) / 3.0
        first_diff = F.pad(work[..., 1:] - work[..., :-1], (1, 0))
        return (periodic + 0.25 * first_diff.abs() * torch.tanh(work)).to(y.dtype)

    def forward(self, y: Tensor) -> list[Tensor]:
        """``y: (B, C, T)`` → 6 个 ``(B, C, T)`` 观测，顺序同 :data:`ARTIFACT_NAMES`。

        **内部一律用 float32 计算**，出入口再转换 dtype。原因有二：

        * 这些算子里既有 FFT 又有线性求解，半精度不会带来任何收益；
        * 半精度还会引入**跨设备差异**：CUDA 的 FFT 支持 bf16 而 CPU 不支持
          （``RuntimeError: Unsupported dtype BFloat16``），bf16 的 LU 分解则
          两个后端都没有实现（``lu_factor_cublas`` / ``lu_cpu`` not implemented for 'BFloat16'）。
          训练时 ``robust_normalize`` 会被 autocast 推成 bf16，所以这条路径必然被走到。

        输出保持输入的 dtype，以免改变下游 autocast 的行为。
        """
        if y.dim() != 3:
            raise ValueError(f"expected (batch, channels, time), got {tuple(y.shape)}")
        work = y.float()
        with no_autocast(y):
            slow = _moving_average(work, max(int(self.sample_rate * 0.25) | 1, 3))
            local = _moving_average(work, max(int(self.sample_rate * 0.04) | 1, 3))
            high = work - local
            first_diff = F.pad(work[..., 1:] - work[..., :-1], (1, 0))
            second_diff = F.pad(first_diff[..., 1:] - first_diff[..., :-1], (1, 0))
            pulse = self._cardiac(work)
            motion = torch.tanh(2.0 * first_diff) + 0.5 * torch.tanh(second_diff)
            unexplained = work - slow - high
            observations = [self._harmonic_basis(work), slow, high, pulse, motion, unexplained]
        if y.dtype != torch.float32:
            observations = [item.to(y.dtype) for item in observations]
        return observations


#: 只有这两路观测接受空间权重调制（手册 §6.4 步骤 6）。
SPATIAL_MODULATED_ARTIFACTS: tuple[str, ...] = ("ocular_drift", "cardiac")
SPATIAL_MODULATED_INDICES: tuple[int, ...] = tuple(
    ARTIFACT_NAMES.index(name) for name in SPATIAL_MODULATED_ARTIFACTS
)


def _orthonormalize(block: Tensor, eps: float = 1e-6) -> Tensor:
    """把 ``(B, T, k)`` 的列组正交化并归一化。

    **必须"投影完立刻归一化"再进入下一列**：如果保留未归一化的列当参考向量，
    后面每次投影减掉的都是范数暴涨的向量，残差会被指数放大
    （实测 4 列就会从 337 涨到 3.4e22，随即归一化除以 inf 得到 NaN）。
    第一遍构造正交基，第二遍用已归一化的列再扫一遍以提升正交性。
    """
    columns: list[Tensor] = []
    for index in range(block.size(-1)):
        vector = block[:, :, index]
        for previous in columns:
            vector = vector - (previous * vector).sum(dim=-1, keepdim=True) * previous
        columns.append(vector / vector.norm(dim=1, keepdim=True).clamp_min(eps))

    refined: list[Tensor] = []
    for vector in columns:
        for previous in refined:
            vector = vector - (previous * vector).sum(dim=-1, keepdim=True) * previous
        refined.append(vector / vector.norm(dim=1, keepdim=True).clamp_min(eps))
    return torch.stack(refined, dim=-1)


class SpatialProjectionHead(nn.Module):
    """显式空间投影（手册 T1.3、§6.4）。

    设计依据：设计文档 §3.4.4「多通道模式利用不同电极上的空间投影一致性」、
    §3.4.2「空间前额优势」、§3.7「源时程 + 空间投影因子化」。
    位置在观测抽取之后、专家调用之前，**只调制 cardiac 与 ocular_drift 两路**：
    ``obs_k ← obs_k * (1 + α·w_s)``，其余四路不动。

    为什么是 DeepSets（逐通道 MLP + 注意力池化）而不是 ``nn.Linear``：
    ``nn.Linear(C, ·)`` 会把通道数写死，无法让同一权重直接处理不同 C。
    本模块参数量与 ``C`` **严格无关**（有单测钉住）。

    两处与 §6.4 字面的差异（都记在附录 C）：

    1. §6.4 步骤 2 算出了逐通道载荷 ``(B, C, rank)``，但步骤 3–5 没有再用它。
       逐通道载荷恰恰是"空间投影一致性"的载体，所以这里把它作为 MLP 的输入特征之一。
       特征维度 = 坐标(coord_dim) + RMS(1) + 有坐标标志(1) + 载荷(rank)，与 C 无关。
    2. §6.4 说"对 ``y_norm`` 做 SVD"。逐 batch 精确 SVD 在本机代价过大
       （``C=128`` 时每 batch 要 64 次 128×1000 分解），默认改用**确定性幂迭代**
       求同一子空间（``spatial_pca_mode="power"``）。精确路径保留为 ``"svd"``，
       两者子空间一致性有单测断言（主角度）。
    """

    def __init__(self, config: UniCOREEGConfig) -> None:
        super().__init__()
        self.rank = max(int(config.spatial_pca_rank), 1)
        self.coord_dim = max(int(config.spatial_coord_dim), 1)
        self.iterations = max(int(config.spatial_power_iterations), 1)
        self.mode = str(config.spatial_pca_mode)
        self.coord_space = str(config.spatial_coord_space)
        self.alpha_scale = float(config.spatial_modulation_alpha_scale)
        self.eps = float(config.eps)

        hidden = int(config.spatial_hidden)
        feature_dim = self.coord_dim + 2 + self.rank
        self.per_channel = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.attention = nn.Linear(hidden, 1)
        self.to_logit = nn.Linear(hidden, 1)
        self.from_context = nn.Linear(hidden, 1)
        # α 用 tanh 有界化、初值取 0：初始化时整个模块是恒等映射，
        # 于是"打开 use_spatial"不会在训练一开始就扰动既有行为（单通道回归因此更干净）。
        self.alpha_raw = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def component_directions(self, y: Tensor) -> Tensor:
        """前 ``rank`` 个空间主成分的时序，形状 ``(B, rank, T)``。

        ``power``（默认）：确定性幂迭代。``svd``：精确 SVD，用于复核。

        通道数少于 ``rank`` 时用零列补齐，保证输出维度恒为 ``rank``——
        否则 MLP 的输入维度会随 ``C`` 变化，违反"空间模块参数量与 C 无关"。
        """
        batch, channels, length = y.shape
        usable = min(self.rank, channels)

        if self.mode == "svd":
            # 必须 full_matrices=False，否则会产出 T×T 的 Vh（C=128/T=1000 时 256 MB）
            _, _, vh = torch.linalg.svd(y, full_matrices=False)
            directions = vh[:, :usable, :]
        elif self.mode == "power":
            # 用"通道前若干行的时序"做确定性初值，再反复做 v ← yᵀ(y·v)，中间正交化。
            # 每轮开销 O(B·C·T·rank)，比 B 次 SVD 小几个量级。
            basis = y[:, :usable, :].transpose(1, 2).contiguous()
            for _ in range(self.iterations):
                basis = _orthonormalize(basis, eps=self.eps)
                projected = torch.matmul(y, basis)  # (B, C, usable)
                basis = torch.matmul(y.transpose(1, 2), projected)  # (B, T, usable)
            basis = _orthonormalize(basis, eps=self.eps)
            directions = basis.transpose(1, 2)
        else:
            raise ValueError(f"未知的 spatial_pca_mode={self.mode!r}（可用 power / svd）")

        if usable < self.rank:
            pad = directions.new_zeros(batch, self.rank - usable, length)
            directions = torch.cat((directions, pad), dim=1)
        return directions

    def channel_loadings(self, y: Tensor, directions: Tensor, rms: Tensor) -> Tensor:
        """逐通道载荷 ``(B, C, rank)``：通道时序与主成分的相关（按 RMS 归一）。"""
        loadings = torch.matmul(y, directions.transpose(1, 2)) / y.size(-1)
        return loadings / rms.unsqueeze(-1).clamp_min(self.eps)

    def project_coordinates(self, coords: Tensor, coords_mask: Tensor | None = None) -> Tensor:
        """把 ``(B, C, 3)`` 坐标压成 MLP 能吃的 ``(B, C, coord_dim)``。

        ``raw``（默认）：直接取前 ``coord_dim`` 个分量。对任何坐标系都成立——
        各数据集的坐标轴含义不同（参考表是 RAS，ds004784 是 x=前/y=左右），
        但"前两个分量"在各自坐标系里都是稳定的二维编码，不需要额外约定。

        ``disk``：用 montage 的等距方位投影压到头部圆盘。语义更接近"头地图"，
        但要求坐标系的"上"轴确实是 +z（参考表与 ds004784 都满足）。
        """
        if self.coord_space == "disk":
            from .montage import project_to_disk

            base = coords[0] if coords.dim() == 3 else coords
            projected = project_to_disk(base, coords_mask)
            projected = projected.unsqueeze(0).expand(coords.size(0), -1, -1)
        elif self.coord_space == "raw":
            projected = coords
        else:
            raise ValueError(
                f"未知的 spatial_coord_space={self.coord_space!r}（可用 raw / disk）"
            )

        out = projected[..., : self.coord_dim]
        if out.size(-1) < self.coord_dim:  # 坐标维度不足时补零
            pad = out.new_zeros(out.size(0), out.size(1), self.coord_dim - out.size(-1))
            out = torch.cat((out, pad), dim=-1)
        return out

    def forward(
        self,
        y_norm: Tensor,
        coords: Tensor | None = None,
        coords_mask: Tensor | None = None,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        """返回逐通道空间权重 ``w_s: (B, C, 1)``。"""
        batch, channels, _ = y_norm.shape
        if channel_mask is not None:
            if channel_mask.dim() == 1:
                channel_mask = channel_mask.unsqueeze(0).expand(batch, -1)
            coords_mask = channel_mask if coords_mask is None else coords_mask
            if coords_mask.dim() == 1:
                coords_mask = coords_mask.unsqueeze(0).expand(batch, -1)
            coords_mask = coords_mask.bool() & channel_mask.bool()
        rms = y_norm.square().mean(dim=-1).sqrt()  # (B, C)
        directions = self.component_directions(y_norm)
        loadings = self.channel_loadings(y_norm, directions, rms)  # (B, C, rank)

        if coords is None:
            projected = y_norm.new_zeros(batch, channels, self.coord_dim)
            has_coords = y_norm.new_zeros(batch, channels, 1)
        else:
            if coords.dim() == 2:
                coords = coords.unsqueeze(0).expand(batch, -1, -1)
            if coords.size(1) != channels:
                raise ValueError(
                    f"coords 有 {coords.size(1)} 个通道，与输入通道数 {channels} 不一致"
                )
            projected = self.project_coordinates(coords, coords_mask)
            if coords_mask is None:
                has_coords = y_norm.new_ones(batch, channels, 1)
            else:
                if coords_mask.dim() == 1:
                    coords_mask = coords_mask.unsqueeze(0).expand(batch, -1)
                has_coords = coords_mask.to(y_norm.dtype).unsqueeze(-1)

        feature = torch.cat(
            (projected * has_coords, rms.unsqueeze(-1), has_coords, loadings), dim=-1
        )
        hidden = self.per_channel(feature)  # (B, C, H)，逐通道独立作用

        # 注意力池化：没有坐标的通道不参与注意力（否则它们会把上下文带偏）
        scores = self.attention(hidden).squeeze(-1)
        scores = scores.masked_fill(has_coords.squeeze(-1) < 0.5, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=1.0 / max(channels, 1))
        context = torch.einsum("bc,bch->bh", weights, hidden)  # (B, H)

        logit = self.to_logit(hidden) + self.from_context(context).unsqueeze(1)
        return torch.sigmoid(logit)

    def build_modulation(self, w_s: Tensor, reference: Tensor) -> Tensor:
        """``1 + α·w_s``，形状 ``(B, C, 1)``。α 有界且初值为 0（恒等映射）。"""
        alpha = self.alpha_scale * torch.tanh(self.alpha_raw).to(reference.dtype)
        return 1.0 + alpha * w_s.to(reference.dtype)


class ArtifactTokenizer(nn.Module):
    def __init__(self, config: UniCOREEGConfig) -> None:
        super().__init__()
        self.metadata_dim = config.metadata_dim
        self.base_channels = config.base_channels
        self.time_branch = nn.Sequential(
            ConvGNAct(config.base_channels, 96, 5, stride=2),
            ConvGNAct(96, 128, 5, stride=2),
        )
        self.freq_branch = nn.Sequential(
            ConvGNAct(1, 32, 7, stride=2),
            ConvGNAct(32, 64, 7, stride=2),
        )
        feature_dim = 128 * 2 + 64 * 2 + 3 + 4
        self.per_channel = nn.Sequential(nn.Linear(feature_dim, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.attention = nn.Linear(128, 1)
        self.fuse = nn.Sequential(nn.Linear(128 + config.metadata_dim, 192), nn.SiLU(), nn.Linear(192, 128), nn.SiLU())
        self.prob = nn.Linear(128, config.artifact_count)
        self.artifact_presence = nn.Linear(128, 1)
        self.severity = nn.Linear(128, config.artifact_count)
        self.priority = nn.Linear(128, config.artifact_count)
        self.token = nn.Linear(128, config.artifact_count * config.token_dim)

    @staticmethod
    def _pool(x: Tensor) -> Tensor:
        return torch.cat((x.mean(dim=-1), x.std(dim=-1, unbiased=False)), dim=-1)

    def forward(
        self,
        y: Tensor,
        h0: Tensor,
        stats: Tensor,
        metadata: Tensor | None,
        coords: Tensor | None = None,
        coords_mask: Tensor | None = None,
        channel_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch, channels, length = y.shape
        if channel_mask is None:
            channel_mask = torch.ones(batch, channels, dtype=torch.bool, device=y.device)
        if metadata is None:
            metadata = y.new_zeros((y.size(0), self.metadata_dim))
        elif metadata.shape != (y.size(0), self.metadata_dim):
            raise ValueError(f"metadata must have shape (batch, {self.metadata_dim})")
        if coords is None:
            coords = y.new_zeros(batch, channels, 3)
            coords_mask = torch.zeros(batch, channels, dtype=torch.bool, device=y.device)
        else:
            if coords.dim() == 2:
                coords = coords.unsqueeze(0).expand(batch, -1, -1)
            if coords_mask is None:
                coords_mask = channel_mask
            elif coords_mask.dim() == 1:
                coords_mask = coords_mask.unsqueeze(0).expand(batch, -1)
            coords_mask = coords_mask.bool() & channel_mask.bool()
        channel_mask = channel_mask.bool()
        flat_y = y.reshape(batch * channels, 1, length)
        flat_h = h0.reshape(batch * channels, self.base_channels, length)
        spectrum = torch.log1p(torch.fft.rfft(flat_y, dim=-1).abs())
        channel_features = torch.cat((
            self._pool(self.time_branch(flat_h)),
            self._pool(self.freq_branch(spectrum)),
            stats.reshape(batch * channels, 3),
            coords.reshape(batch * channels, 3),
            coords_mask.reshape(batch * channels, 1).to(y.dtype),
        ), dim=-1).reshape(batch, channels, -1)
        encoded = self.per_channel(channel_features)
        attention = self.attention(encoded).squeeze(-1).masked_fill(~channel_mask, float("-inf"))
        weights = torch.softmax(attention, dim=-1)
        pooled = torch.einsum("bc,bch->bh", weights, encoded)
        fused = self.fuse(torch.cat((pooled, metadata), dim=-1))
        logits = self.prob(fused)
        artifact_presence_logit = self.artifact_presence(fused).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        severities = F.softplus(self.severity(fused))
        priorities = self.priority(fused)
        tokens = self.token(fused).view(y.size(0), len(ARTIFACT_NAMES), -1)
        aggregate = (probabilities.unsqueeze(-1) * tokens).sum(dim=1)
        aggregate = aggregate / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-4)
        return {
            "probabilities": probabilities,
            "probability_logits": logits,
            "artifact_presence_logit": artifact_presence_logit,
            "artifact_presence_probability": torch.sigmoid(artifact_presence_logit),
            "severities": severities,
            "priorities": priorities,
            "tokens": tokens,
            "aggregate": aggregate,
        }


class SparseRouter(nn.Module):
    def __init__(self, config: UniCOREEGConfig) -> None:
        super().__init__()
        self.temperature = config.router_temperature
        self.max_active = config.max_active_experts
        self.probability_threshold = config.probability_threshold
        self.probability_thresholds = config.probability_thresholds
        self.clean_threshold = config.clean_threshold
        self.bypass_source = config.bypass_source

    def forward(
        self,
        tokens: dict[str, Tensor],
        mode: str = "learned",
        oracle_labels: Tensor | None = None,
        disabled_experts: Tensor | None = None,
        temperature_override: Tensor | None = None,
    ) -> dict[str, Tensor]:
        probabilities = tokens["probabilities"]
        if self.bypass_source == "presence_head":
            artifact_presence = tokens.get("artifact_presence_probability")
            if artifact_presence is None:
                # Direct router callers and legacy checkpoints may omit the head.
                artifact_presence = probabilities.masked_fill(
                    disabled_experts.bool(), 0.0
                ) if disabled_experts is not None else probabilities
                artifact_presence = artifact_presence.amax(dim=-1)
        else:
            masked_probabilities = probabilities.masked_fill(
                disabled_experts.bool(), 0.0
            ) if disabled_experts is not None else probabilities
            artifact_presence = masked_probabilities.amax(dim=-1)
        temperature = max(self.temperature, 1e-4) if temperature_override is None else temperature_override
        if isinstance(temperature, Tensor):
            temperature = temperature.view(-1, 1, 1).clamp_min(1e-4)
        scores = probabilities * torch.sigmoid(tokens["severities"]) * F.softmax(
            tokens["priorities"] / temperature, dim=-1
        )
        enabled = torch.ones_like(scores, dtype=torch.bool)
        if disabled_experts is not None:
            enabled &= ~disabled_experts.bool()
        scores = scores * enabled

        if mode == "oracle":
            if oracle_labels is None:
                raise ValueError("oracle route requires oracle_labels")
            active = oracle_labels.bool() & enabled
            bypass = ~active.any(dim=-1)
            sparse = active.to(scores.dtype)
        elif mode == "all":
            active = enabled
            bypass = artifact_presence < self.clean_threshold
            active &= ~bypass.unsqueeze(-1)
            sparse = scores * active
        elif mode == "learned":
            bypass = artifact_presence < self.clean_threshold
            if self.probability_thresholds is None:
                thresholds = torch.full_like(probabilities, self.probability_threshold)
            else:
                thresholds = probabilities.new_tensor(self.probability_thresholds).view(1, -1)
            active = (probabilities >= thresholds) & enabled & ~bypass.unsqueeze(-1)
            fallback = scores.argmax(dim=-1, keepdim=True)
            needs_fallback = ~active.any(dim=-1, keepdim=True) & ~bypass.unsqueeze(-1)
            active |= torch.zeros_like(active).scatter(-1, fallback, needs_fallback)
            if self.max_active < scores.size(-1):
                top_indices = scores.topk(self.max_active, dim=-1).indices
                top_mask = torch.zeros_like(active).scatter(-1, top_indices, True)
                active &= top_mask
            sparse = scores * active
        else:
            raise ValueError(f"unsupported route mode: {mode}")

        route = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        route = route.masked_fill(bypass.unsqueeze(-1), 0.0)
        return {"route": route, "route_scores": scores, "active_mask": active, "bypass": bypass}


class ContentEncoder(nn.Module):
    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.dims = (base_channels, 96, 160, 256)
        self.blocks = nn.ModuleList([MSBlock(dim, dilation) for dim, dilation in zip(self.dims, (1, 2, 4, 8))])
        self.downs = nn.ModuleList([ConvGNAct(self.dims[i], self.dims[i + 1], 5, stride=2) for i in range(3)])
        self.bottleneck = nn.Sequential(MSBlock(256, 2), MSBlock(256, 4))

    def forward(self, h0: Tensor) -> list[Tensor]:
        features = []
        x = h0
        for index, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if index < len(self.downs):
                x = self.downs[index](x)
        features[-1] = self.bottleneck(features[-1])
        return features


class ArtifactExpert(nn.Module):
    def __init__(self, dims: tuple[int, ...], token_dim: int, kind: str) -> None:
        super().__init__()
        settings = {
            "harmonic": ((31, 15, 9, 7), (1, 2, 3, 4)),
            "ocular_drift": ((63, 31, 15, 9), (1, 2, 4, 8)),
            "myogenic": ((5, 5, 3, 3), (1, 1, 2, 2)),
            "cardiac": ((17, 11, 7, 5), (1, 2, 4, 4)),
            "motion_transient": ((9, 7, 5, 3), (1, 2, 4, 8)),
            "unknown": ((7, 5, 3, 3), (1, 1, 2, 2)),
        }
        kernels, dilations = settings[kind]
        hidden_scale = 0.5 if kind == "unknown" else 1.0
        self.adapters = nn.ModuleList()
        self.observation_proj = nn.ModuleList()
        self.film = nn.ModuleList()
        for dim, kernel, dilation in zip(dims, kernels, dilations):
            hidden = max(int(dim * hidden_scale), 16)
            self.adapters.append(nn.Sequential(ConvGNAct(dim, hidden, kernel, dilation=dilation), nn.Conv1d(hidden, dim, 1)))
            self.observation_proj.append(nn.Conv1d(1, dim, 1))
            self.film.append(nn.Linear(token_dim, dim * 2))

    def forward(self, features: list[Tensor], token: Tensor, observation: Tensor) -> list[Tensor]:
        outputs = []
        for feature, adapter, obs_proj, film in zip(features, self.adapters, self.observation_proj, self.film):
            batch = None
            obs_input, film_token = observation, token
            if feature.dim() == 4:
                batch, channels, width, length = feature.shape
                feature = feature.reshape(batch * channels, width, length)
                obs_input = observation.reshape(batch * channels, 1, observation.size(-1))
                film_token = token[:, None, :].expand(batch, channels, -1).reshape(batch * channels, -1)
            obs = F.interpolate(obs_input, size=feature.size(-1), mode="linear", align_corners=False)
            gamma, beta = film(film_token).chunk(2, dim=-1)
            adapted = adapter(feature) + obs_proj(obs)
            result = adapted * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
            if batch is not None:
                result = result.reshape(batch, channels, result.size(1), result.size(-1))
            outputs.append(result)
        return outputs


class UnknownDetector(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def forward(self, residual_feature: Tensor, channel_mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if residual_feature.dim() == 4:
            batch, channels, _, _ = residual_feature.shape
            if channel_mask is None:
                channel_mask = residual_feature.new_ones(batch, channels)
            weights = channel_mask.to(residual_feature.dtype).view(batch, channels, 1, 1)
            residual_feature = (residual_feature * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = torch.cat((residual_feature.mean(dim=-1), residual_feature.std(dim=-1, unbiased=False)), dim=-1)
        logit, severity, priority = self.net(pooled).unbind(dim=-1)
        return logit, F.softplus(severity), priority


class CrossStreamGating(nn.Module):
    def __init__(self, dims: tuple[int, ...], token_dim: int) -> None:
        super().__init__()
        self.token_proj = nn.ModuleList([nn.Linear(token_dim, dim) for dim in dims])
        self.artifact_proj = nn.ModuleList([nn.Conv1d(dim, dim, 1) for dim in dims])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Conv1d(dim * 3, dim, 1), nn.GroupNorm(_groups(dim), dim), nn.SiLU(), nn.Conv1d(dim, dim, 1), nn.Sigmoid())
            for dim in dims
        ])

    def forward(self, neural: list[Tensor], experts: list[list[Tensor]], route: Tensor, token: Tensor, channel_mask: Tensor | None = None) -> tuple[list[Tensor], Tensor]:
        purified, gate_values = [], []
        for level, neural_level in enumerate(neural):
            if neural_level.dim() == 4:
                batch, channels, width, length = neural_level.shape
                base = neural_level.reshape(batch * channels, width, length)
                expert_level = [expert[level].reshape(batch * channels, width, length) for expert in experts]
                level_route = route[:, None, :].expand(batch, channels, -1).reshape(batch * channels, -1)
                level_token = token[:, None, :].expand(batch, channels, -1).reshape(batch * channels, -1)
            else:
                batch, channels = neural_level.size(0), 1
                base, expert_level, level_route, level_token = neural_level, [expert[level] for expert in experts], route, token
            stacked = torch.stack(expert_level, dim=1)
            mixed = (level_route[:, :, None, None] * stacked).sum(dim=1)
            token_map = self.token_proj[level](level_token).unsqueeze(-1).expand_as(base)
            gate = self.gates[level](torch.cat((base, mixed, token_map), dim=1))
            result = base - gate * self.artifact_proj[level](mixed)
            if channel_mask is not None and neural_level.dim() == 4:
                valid = channel_mask.reshape(batch * channels, 1, 1).to(gate.dtype)
                gate_values.append((gate * valid).sum() / (valid.sum() * gate.size(1) * gate.size(2)).clamp_min(1.0))
            else:
                gate_values.append(gate.mean())
            if neural_level.dim() == 4:
                result = result.reshape(batch, channels, width, length)
            purified.append(result)
        return purified, torch.stack(gate_values).mean()


class CoarseDecoder(nn.Module):
    def __init__(self, config: UniCOREEGConfig, dims: tuple[int, ...]) -> None:
        super().__init__()
        reversed_dims = tuple(reversed(dims))
        self.up_blocks = nn.ModuleList()
        in_dim = reversed_dims[0]
        for skip_dim in reversed_dims[1:]:
            self.up_blocks.append(nn.Sequential(ConvGNAct(in_dim + skip_dim, skip_dim, 3), MSBlock(skip_dim, 1)))
            in_dim = skip_dim
        self.clean_head = nn.Conv1d(dims[0], 1, 1)
        self.artifact_heads = nn.ModuleList([
            nn.Sequential(ConvGNAct(dims[0] * 2, dims[0], 7), nn.Conv1d(dims[0], 1, 1))
            for _ in ARTIFACT_NAMES
        ])

    def forward(self, features: list[Tensor], experts: list[list[Tensor]], route: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, channels = features[-1].shape[:2] if features[-1].dim() == 4 else (features[-1].size(0), 1)
        if features[-1].dim() == 4:
            features = [item.reshape(batch * channels, item.size(2), item.size(3)) for item in features]
            experts = [[item.reshape(batch * channels, item.size(2), item.size(3)) for item in expert] for expert in experts]
            route = route[:, None, :].expand(batch, channels, -1).reshape(batch * channels, -1)
        x = features[-1]
        for block, skip in zip(self.up_blocks, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        coarse_clean = self.clean_head(x)
        components = torch.stack([
            head(torch.cat((x, expert[0]), dim=1)) for head, expert in zip(self.artifact_heads, experts)
        ], dim=1)
        artifact_sum = (route[:, :, None, None] * components).sum(dim=1)
        coarse_clean = coarse_clean.reshape(batch, channels, -1)
        components = components.reshape(batch, channels, len(ARTIFACT_NAMES), -1).permute(0, 2, 1, 3)
        artifact_sum = artifact_sum.reshape(batch, channels, -1)
        return coarse_clean, components, artifact_sum


class ResidualRefiner(nn.Module):
    def __init__(self, config: UniCOREEGConfig) -> None:
        super().__init__()
        dims = (48, 96, 160)
        self.in_proj = ConvGNAct(4, dims[0], 7)
        self.down1 = ConvGNAct(dims[0], dims[1], 5, stride=2)
        self.down2 = ConvGNAct(dims[1], dims[2], 5, stride=2)
        self.block0 = FiLMBlock(dims[0], config.token_dim, 1)
        self.block1 = FiLMBlock(dims[1], config.token_dim, 2)
        self.block2 = FiLMBlock(dims[2], config.token_dim, 4)
        self.up1 = ConvGNAct(dims[2] + dims[1], dims[1], 3)
        self.up0 = ConvGNAct(dims[1] + dims[0], dims[0], 3)
        self.out = nn.Conv1d(dims[0], 1, 1)
        self.rho = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, inputs: Tensor, token: Tensor, max_probability: Tensor, mean_severity: Tensor) -> Tensor:
        if inputs.dim() == 4:
            batch, channels, _, length = inputs.shape
            inputs = inputs.reshape(batch * channels, 4, length)
            token = token[:, None, :].expand(batch, channels, -1).reshape(batch * channels, -1)
            max_probability = max_probability[:, None].expand(batch, channels).reshape(-1)
            mean_severity = mean_severity[:, None].expand(batch, channels).reshape(-1)
        else:
            batch, channels = inputs.size(0), 1
        x0 = self.block0(self.in_proj(inputs), token)
        x1 = self.block1(self.down1(x0), token)
        x2 = self.block2(self.down2(x1), token)
        y = self.up1(torch.cat((F.interpolate(x2, size=x1.size(-1), mode="linear", align_corners=False), x1), dim=1))
        y = self.up0(torch.cat((F.interpolate(y, size=x0.size(-1), mode="linear", align_corners=False), x0), dim=1))
        rho = 0.05 + 0.95 * self.rho(torch.stack((max_probability, mean_severity), dim=-1))
        return (rho.unsqueeze(-1) * torch.tanh(self.out(y))).reshape(batch, channels, -1)


class IdentityGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)
        nn.init.constant_(self.linear.bias, -2.0)

    def forward(self, y: Tensor, probabilities: Tensor, artifact_sum: Tensor, residual: Tensor, bypass: Tensor, channel_mask: Tensor | None = None) -> Tensor:
        if channel_mask is None:
            channel_mask = y.new_ones(y.shape[:2])
        weights = channel_mask.to(y.dtype).unsqueeze(-1)
        count = (weights.sum(dim=(1, 2)) * y.size(-1)).clamp_min(1.0)
        denom = ((y.square() * weights).sum(dim=(1, 2)) / count + 1e-8).sqrt()
        artifact_ratio = (((artifact_sum.square() * weights).sum(dim=(1, 2)) / count + 1e-8).sqrt()) / denom
        residual_ratio = (((residual.square() * weights).sum(dim=(1, 2)) / count + 1e-8).sqrt()) / denom
        features = torch.stack((probabilities.max(dim=-1).values, artifact_ratio, residual_ratio), dim=-1)
        strength = torch.sigmoid(self.linear(features)).view(-1, 1, 1)
        return strength.masked_fill(bypass.view(-1, 1, 1), 0.0)


class UniCOREEG(nn.Module):
    def __init__(self, config: UniCOREEGConfig | None = None) -> None:
        super().__init__()
        self.config = config or UniCOREEGConfig()
        dims = (self.config.base_channels, 96, 160, 256)
        self.observations = ArtifactObservationExtractor(self.config.sample_rate)
        self.spatial = SpatialProjectionHead(self.config)
        self.stem = SharedStem(self.config.base_channels)
        self.tokenizer = ArtifactTokenizer(self.config)
        self.router = SparseRouter(self.config)
        self.domain_calibration = nn.Sequential(
            nn.Linear(self.config.token_dim + self.config.metadata_dim, self.config.domain_calibration_hidden),
            nn.SiLU(),
            nn.Linear(self.config.domain_calibration_hidden, 1),
        )
        self.content_encoder = ContentEncoder(self.config.base_channels)
        self.experts = nn.ModuleList([
            ArtifactExpert(dims, self.config.token_dim, name) for name in ARTIFACT_NAMES
        ])
        self.unknown_detector = UnknownDetector(dims[0])
        self.gating = CrossStreamGating(dims, self.config.token_dim)
        self.coarse_decoder = CoarseDecoder(self.config, dims)
        self.residual_refiner = ResidualRefiner(self.config)
        self.identity_gate = IdentityGate()

    def robust_normalize(self, y: Tensor, channel_mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if channel_mask is None:
            channel_mask = torch.ones(y.shape[:2], dtype=torch.bool, device=y.device)
        channel_mask = channel_mask.bool()
        y = y.masked_fill(~channel_mask.unsqueeze(-1), 0.0)
        median = y.median(dim=-1, keepdim=True).values
        centered = y - median
        mad = centered.abs().median(dim=-1, keepdim=True).values.clamp_min(self.config.eps)
        y_norm = centered / mad
        stats = torch.log1p(torch.stack((y.square().mean(dim=-1).sqrt(), mad.squeeze(-1), y.amax(dim=-1) - y.amin(dim=-1)), dim=-1).clamp_min(0.0))
        y_norm = y_norm.masked_fill(~channel_mask.unsqueeze(-1), 0.0)
        stats = stats.masked_fill(~channel_mask.unsqueeze(-1), 0.0)
        return y_norm, median, mad, stats

    def forward(
        self,
        y: Tensor,
        metadata: Tensor | None = None,
        route_mode: str = "learned",
        oracle_labels: Tensor | None = None,
        disabled_experts: Tensor | None = None,
        enable_residual: bool = True,
        coords: Tensor | None = None,
        coords_mask: Tensor | None = None,
        channel_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if y.ndim != 3:
            raise ValueError(f"y must have shape (batch, channels, time), got {tuple(y.shape)}")
        batch, channels, length = y.shape
        if channel_mask is None:
            channel_mask = torch.ones(batch, channels, dtype=torch.bool, device=y.device)
        elif channel_mask.shape != (batch, channels):
            raise ValueError(f"channel_mask must have shape {(batch, channels)}, got {tuple(channel_mask.shape)}")
        if not channel_mask.bool().any(dim=-1).all():
            raise ValueError("每个样本至少要有一个有效 EEG 通道")
        channel_mask = channel_mask.bool()
        y_norm, median, mad, stats = self.robust_normalize(y, channel_mask)
        flat_y = y_norm.reshape(batch * channels, 1, length)
        h0 = self.stem(flat_y).reshape(batch, channels, self.config.base_channels, length)
        token_outputs = self.tokenizer(y_norm, h0, stats, metadata, coords, coords_mask, channel_mask)
        flat_features = self.content_encoder(h0.reshape(batch * channels, self.config.base_channels, length))
        neural_features = [item.reshape(batch, channels, item.size(1), item.size(-1)) for item in flat_features]
        observations = self.observations(y_norm)

        # --- 空间投影：只调制 cardiac 与 ocular_drift 两路观测（T1.3）---
        spatial_weights = None
        if self.config.use_spatial:
            spatial_weights = self.spatial(y_norm, coords, coords_mask, channel_mask)  # (B, C, 1)
            modulation = self.spatial.build_modulation(spatial_weights, observations[0])
            observations = list(observations)
            for index in SPATIAL_MODULATED_INDICES:
                observations[index] = observations[index] * modulation
        spatial_modulation = (
            None if spatial_weights is None else self.spatial.build_modulation(spatial_weights, y_norm)
        )

        known_features = [
            expert(neural_features, token_outputs["tokens"][:, index], observations[index])
            for index, expert in enumerate(self.experts[:-1])
        ]
        known_weights = token_outputs["probabilities"][:, :-1]
        if disabled_experts is not None:
            known_weights = known_weights * (~disabled_experts[:, :-1].bool()).to(known_weights.dtype)
        known_weights = known_weights / known_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        unknown_inputs = []
        for level, feature in enumerate(neural_features):
            stacked = torch.stack([expert[level] for expert in known_features], dim=2)
            weights = known_weights[:, None, :, None, None]
            unknown_inputs.append(feature - (weights * stacked).sum(dim=2))
        unknown_features = self.experts[-1](unknown_inputs, token_outputs["tokens"][:, -1], observations[-1])
        expert_features = known_features + [unknown_features]
        unknown_logit, unknown_severity, unknown_priority = self.unknown_detector(unknown_inputs[0], channel_mask)
        token_outputs["probability_logits"] = torch.cat((token_outputs["probability_logits"][:, :-1], unknown_logit[:, None]), dim=-1)
        token_outputs["probabilities"] = torch.sigmoid(token_outputs["probability_logits"])
        token_outputs["severities"] = torch.cat((token_outputs["severities"][:, :-1], unknown_severity[:, None]), dim=-1)
        token_outputs["priorities"] = torch.cat((token_outputs["priorities"][:, :-1], unknown_priority[:, None]), dim=-1)
        token_outputs["aggregate"] = (
            token_outputs["probabilities"].unsqueeze(-1) * token_outputs["tokens"]
        ).sum(dim=1) / token_outputs["probabilities"].sum(dim=1, keepdim=True).clamp_min(1e-4)
        if self.config.domain_calibration_enabled and metadata is not None:
            domain_input = torch.cat((token_outputs["aggregate"], metadata), dim=-1)
            domain_temperature = (1.0 + 0.5 * torch.tanh(self.domain_calibration(domain_input).squeeze(-1)))
        else:
            domain_temperature = None
        routing = self.router(token_outputs, route_mode, oracle_labels, disabled_experts, domain_temperature)
        purified, gate_mean = self.gating(neural_features, expert_features, routing["route"], token_outputs["aggregate"], channel_mask)
        coarse_clean, components, artifact_sum = self.coarse_decoder(purified, expert_features, routing["route"])
        decomp_error = y_norm - coarse_clean - artifact_sum
        if enable_residual:
            residual = self.residual_refiner(
                torch.stack((y_norm, coarse_clean, artifact_sum, decomp_error), dim=2),
                token_outputs["aggregate"],
                token_outputs["probabilities"].max(dim=-1).values,
                token_outputs["severities"].mean(dim=-1),
            )
        else:
            residual = torch.zeros_like(y_norm)
        identity_strength = self.identity_gate(y_norm, token_outputs["probabilities"], artifact_sum, residual, routing["bypass"], channel_mask)
        clean_norm = y_norm + identity_strength * (coarse_clean + residual - y_norm)
        valid = channel_mask.unsqueeze(-1)
        clean_norm = clean_norm.masked_fill(~valid, 0.0)
        coarse_clean = coarse_clean.masked_fill(~valid, 0.0)
        residual = residual.masked_fill(~valid, 0.0)
        artifact_sum = artifact_sum.masked_fill(~valid, 0.0)
        components = components.masked_fill(~valid.unsqueeze(1), 0.0)
        outputs: dict[str, Tensor] = {
            "clean": clean_norm * mad + median,
            "clean_norm": clean_norm,
            "coarse_clean_norm": coarse_clean,
            "residual_norm": residual,
            "artifact_components_norm": components,
            "artifact_sum_norm": artifact_sum,
            "identity_strength": identity_strength.flatten(),
            "gate_mean": gate_mean,
            "median": median,
            "mad": mad,
            "stats": stats,
            "channel_mask": channel_mask,
            "domain_temperature": domain_temperature if domain_temperature is not None else token_outputs["probabilities"].new_ones(token_outputs["probabilities"].size(0)),
            "expert_feature_vectors": torch.stack(
                [levels[-1].mean(dim=(1, 3)) for levels in expert_features], dim=1
            ),
            **token_outputs,
            **routing,
        }
        if spatial_weights is not None:
            # 仅在启用空间投影时出现这两个键：use_spatial=False 的返回字典与改造前逐键一致，
            # 使等价性回归可以直接比对整个字典。
            outputs["spatial_weights"] = spatial_weights
            outputs["spatial_modulation"] = spatial_modulation
        return outputs


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
