from __future__ import annotations

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
    in_channels: int = 1
    sample_rate: int = 500
    window_size: int = 1000
    base_channels: int = 48
    token_dim: int = 64
    artifact_count: int = 6
    metadata_dim: int = 8
    max_active_experts: int = 4
    probability_threshold: float = 0.35
    clean_threshold: float = 0.50
    router_temperature: float = 1.0
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.artifact_count != len(ARTIFACT_NAMES):
            raise ValueError(f"artifact_count must be {len(ARTIFACT_NAMES)}")


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
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_channels, 32, 7),
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
    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        self.sample_rate = sample_rate

    def _harmonic_basis(self, y: Tensor) -> Tensor:
        length = y.size(-1)
        spectrum = torch.fft.rfft(y, dim=-1).abs().mean(dim=1)
        frequencies = torch.fft.rfftfreq(length, 1.0 / self.sample_rate).to(y.device)
        valid = (frequencies >= 8.0) & (frequencies <= min(80.0, self.sample_rate / 2.0 - 1.0))
        masked = spectrum.masked_fill(~valid.unsqueeze(0), -1.0)
        peak_frequency = frequencies[masked.argmax(dim=-1)]
        time = torch.arange(length, device=y.device, dtype=y.dtype) / self.sample_rate
        basis = []
        for harmonic in (1.0, 2.0, 3.0):
            phase = 2.0 * math.pi * harmonic * peak_frequency[:, None] * time[None]
            basis.extend((torch.sin(phase), torch.cos(phase)))
        design = torch.stack(basis, dim=-1)
        coefficients = torch.linalg.lstsq(design.float(), y.mean(dim=1).float().unsqueeze(-1)).solution
        fitted = torch.matmul(design.float(), coefficients).squeeze(-1).to(y.dtype)
        return fitted.unsqueeze(1).expand_as(y)

    def forward(self, y: Tensor) -> list[Tensor]:
        slow = _moving_average(y, max(int(self.sample_rate * 0.25) | 1, 3))
        local = _moving_average(y, max(int(self.sample_rate * 0.04) | 1, 3))
        high = y - local
        first_diff = F.pad(y[..., 1:] - y[..., :-1], (1, 0))
        second_diff = F.pad(first_diff[..., 1:] - first_diff[..., :-1], (1, 0))
        mono = y.mean(dim=1)
        autocorrelation = torch.fft.irfft(torch.fft.rfft(mono, dim=-1).abs().square(), n=y.size(-1), dim=-1)
        lag_start = max(int(0.4 * self.sample_rate), 1)
        lag_end = min(int(1.5 * self.sample_rate), y.size(-1) - 1)
        cardiac_lags = autocorrelation[:, lag_start:lag_end].argmax(dim=-1) + lag_start
        periodic = torch.stack([
            (sample + torch.roll(sample, int(lag), dims=-1) + torch.roll(sample, -int(lag), dims=-1)) / 3.0
            for sample, lag in zip(y, cardiac_lags)
        ])
        pulse = periodic + 0.25 * first_diff.abs() * torch.tanh(y)
        motion = torch.tanh(2.0 * first_diff) + 0.5 * torch.tanh(second_diff)
        unexplained = y - slow - high
        return [self._harmonic_basis(y), slow, high, pulse, motion, unexplained]


class ArtifactTokenizer(nn.Module):
    def __init__(self, config: UniCOREEGConfig) -> None:
        super().__init__()
        stat_dim = config.in_channels * 3
        self.metadata_dim = config.metadata_dim
        self.time_branch = nn.Sequential(
            ConvGNAct(config.base_channels, 96, 5, stride=2),
            ConvGNAct(96, 128, 5, stride=2),
        )
        self.freq_branch = nn.Sequential(
            ConvGNAct(config.in_channels, 32, 7, stride=2),
            ConvGNAct(32, 64, 7, stride=2),
        )
        fused_dim = 128 * 2 + 64 * 2 + stat_dim + config.metadata_dim
        self.fuse = nn.Sequential(nn.Linear(fused_dim, 192), nn.SiLU(), nn.Linear(192, 128), nn.SiLU())
        self.prob = nn.Linear(128, config.artifact_count)
        self.artifact_presence = nn.Linear(128, 1)
        self.severity = nn.Linear(128, config.artifact_count)
        self.priority = nn.Linear(128, config.artifact_count)
        self.token = nn.Linear(128, config.artifact_count * config.token_dim)

    @staticmethod
    def _pool(x: Tensor) -> Tensor:
        return torch.cat((x.mean(dim=-1), x.std(dim=-1, unbiased=False)), dim=-1)

    def forward(self, y: Tensor, h0: Tensor, stats: Tensor, metadata: Tensor | None) -> dict[str, Tensor]:
        if metadata is None:
            metadata = y.new_zeros((y.size(0), self.metadata_dim))
        elif metadata.shape != (y.size(0), self.metadata_dim):
            raise ValueError(f"metadata must have shape (batch, {self.metadata_dim})")
        spectrum = torch.log1p(torch.fft.rfft(y, dim=-1).abs())
        fused = self.fuse(
            torch.cat((self._pool(self.time_branch(h0)), self._pool(self.freq_branch(spectrum)), stats.flatten(1), metadata), dim=-1)
        )
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
        self.clean_threshold = config.clean_threshold

    def forward(
        self,
        tokens: dict[str, Tensor],
        mode: str = "learned",
        oracle_labels: Tensor | None = None,
        disabled_experts: Tensor | None = None,
    ) -> dict[str, Tensor]:
        probabilities = tokens["probabilities"]
        artifact_presence = tokens.get("artifact_presence_probability", probabilities.masked_fill(disabled_experts.bool(), 0.0).amax(dim=-1) if disabled_experts is not None else probabilities.amax(dim=-1))
        scores = probabilities * torch.sigmoid(tokens["severities"]) * F.softmax(
            tokens["priorities"] / max(self.temperature, 1e-4), dim=-1
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
            active = (probabilities >= self.probability_threshold) & enabled & ~bypass.unsqueeze(-1)
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
    def __init__(self, dims: tuple[int, ...], token_dim: int, kind: str, in_channels: int) -> None:
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
            self.observation_proj.append(nn.Conv1d(in_channels, dim, 1))
            self.film.append(nn.Linear(token_dim, dim * 2))

    def forward(self, features: list[Tensor], token: Tensor, observation: Tensor) -> list[Tensor]:
        outputs = []
        for feature, adapter, obs_proj, film in zip(features, self.adapters, self.observation_proj, self.film):
            obs = F.interpolate(observation, size=feature.size(-1), mode="linear", align_corners=False)
            gamma, beta = film(token).chunk(2, dim=-1)
            adapted = adapter(feature) + obs_proj(obs)
            outputs.append(adapted * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1))
        return outputs


class UnknownDetector(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def forward(self, residual_feature: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pooled = torch.cat(
            (residual_feature.mean(dim=-1), residual_feature.std(dim=-1, unbiased=False)),
            dim=-1,
        )
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

    def forward(self, neural: list[Tensor], experts: list[list[Tensor]], route: Tensor, token: Tensor) -> tuple[list[Tensor], Tensor]:
        purified, gate_values = [], []
        for level, neural_level in enumerate(neural):
            stacked = torch.stack([expert[level] for expert in experts], dim=1)
            mixed = (route[:, :, None, None] * stacked).sum(dim=1)
            token_map = self.token_proj[level](token).unsqueeze(-1).expand_as(neural_level)
            gate = self.gates[level](torch.cat((neural_level, mixed, token_map), dim=1))
            purified.append(neural_level - gate * self.artifact_proj[level](mixed))
            gate_values.append(gate.mean())
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
        self.clean_head = nn.Conv1d(dims[0], config.in_channels, 1)
        self.artifact_heads = nn.ModuleList([
            nn.Sequential(ConvGNAct(dims[0] * 2, dims[0], 7), nn.Conv1d(dims[0], config.in_channels, 1))
            for _ in ARTIFACT_NAMES
        ])

    def forward(self, features: list[Tensor], experts: list[list[Tensor]], route: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = features[-1]
        for block, skip in zip(self.up_blocks, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        coarse_clean = self.clean_head(x)
        components = torch.stack([
            head(torch.cat((x, expert[0]), dim=1)) for head, expert in zip(self.artifact_heads, experts)
        ], dim=1)
        artifact_sum = (route[:, :, None, None] * components).sum(dim=1)
        return coarse_clean, components, artifact_sum


class ResidualRefiner(nn.Module):
    def __init__(self, config: UniCOREEGConfig) -> None:
        super().__init__()
        dims = (48, 96, 160)
        self.in_proj = ConvGNAct(config.in_channels * 4, dims[0], 7)
        self.down1 = ConvGNAct(dims[0], dims[1], 5, stride=2)
        self.down2 = ConvGNAct(dims[1], dims[2], 5, stride=2)
        self.block0 = FiLMBlock(dims[0], config.token_dim, 1)
        self.block1 = FiLMBlock(dims[1], config.token_dim, 2)
        self.block2 = FiLMBlock(dims[2], config.token_dim, 4)
        self.up1 = ConvGNAct(dims[2] + dims[1], dims[1], 3)
        self.up0 = ConvGNAct(dims[1] + dims[0], dims[0], 3)
        self.out = nn.Conv1d(dims[0], config.in_channels, 1)
        self.rho = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, inputs: Tensor, token: Tensor, max_probability: Tensor, mean_severity: Tensor) -> Tensor:
        x0 = self.block0(self.in_proj(inputs), token)
        x1 = self.block1(self.down1(x0), token)
        x2 = self.block2(self.down2(x1), token)
        y = self.up1(torch.cat((F.interpolate(x2, size=x1.size(-1), mode="linear", align_corners=False), x1), dim=1))
        y = self.up0(torch.cat((F.interpolate(y, size=x0.size(-1), mode="linear", align_corners=False), x0), dim=1))
        rho = 0.05 + 0.95 * self.rho(torch.stack((max_probability, mean_severity), dim=-1))
        return rho.unsqueeze(-1) * torch.tanh(self.out(y))


class IdentityGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)
        nn.init.constant_(self.linear.bias, -2.0)

    def forward(self, y: Tensor, probabilities: Tensor, artifact_sum: Tensor, residual: Tensor, bypass: Tensor) -> Tensor:
        denom = (y.square().mean(dim=(1, 2)) + 1e-8).sqrt()
        artifact_ratio = (artifact_sum.square().mean(dim=(1, 2)) + 1e-8).sqrt() / denom
        residual_ratio = (residual.square().mean(dim=(1, 2)) + 1e-8).sqrt() / denom
        features = torch.stack((probabilities.max(dim=-1).values, artifact_ratio, residual_ratio), dim=-1)
        strength = torch.sigmoid(self.linear(features)).view(-1, 1, 1)
        return strength.masked_fill(bypass.view(-1, 1, 1), 0.0)


class UniCOREEG(nn.Module):
    def __init__(self, config: UniCOREEGConfig | None = None) -> None:
        super().__init__()
        self.config = config or UniCOREEGConfig()
        dims = (self.config.base_channels, 96, 160, 256)
        self.observations = ArtifactObservationExtractor(self.config.sample_rate)
        self.stem = SharedStem(self.config.in_channels, self.config.base_channels)
        self.tokenizer = ArtifactTokenizer(self.config)
        self.router = SparseRouter(self.config)
        self.content_encoder = ContentEncoder(self.config.base_channels)
        self.experts = nn.ModuleList([
            ArtifactExpert(dims, self.config.token_dim, name, self.config.in_channels) for name in ARTIFACT_NAMES
        ])
        self.unknown_detector = UnknownDetector(dims[0])
        self.gating = CrossStreamGating(dims, self.config.token_dim)
        self.coarse_decoder = CoarseDecoder(self.config, dims)
        self.residual_refiner = ResidualRefiner(self.config)
        self.identity_gate = IdentityGate()

    def robust_normalize(self, y: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        median = y.median(dim=-1, keepdim=True).values
        centered = y - median
        mad = centered.abs().median(dim=-1, keepdim=True).values.clamp_min(self.config.eps)
        y_norm = centered / mad
        stats = torch.log1p(torch.stack((y.square().mean(dim=-1).sqrt(), mad.squeeze(-1), y.amax(dim=-1) - y.amin(dim=-1)), dim=-1).clamp_min(0.0))
        return y_norm, median, mad, stats

    def forward(
        self,
        y: Tensor,
        metadata: Tensor | None = None,
        route_mode: str = "learned",
        oracle_labels: Tensor | None = None,
        disabled_experts: Tensor | None = None,
        enable_residual: bool = True,
    ) -> dict[str, Tensor]:
        y_norm, median, mad, stats = self.robust_normalize(y)
        h0 = self.stem(y_norm)
        token_outputs = self.tokenizer(y_norm, h0, stats, metadata)
        neural_features = self.content_encoder(h0)
        observations = self.observations(y_norm)
        known_features = [
            expert(neural_features, token_outputs["tokens"][:, index], observations[index])
            for index, expert in enumerate(self.experts[:-1])
        ]
        known_weights = token_outputs["probabilities"][:, :-1]
        if disabled_experts is not None:
            known_weights = known_weights * (~disabled_experts[:, :-1].bool()).to(known_weights.dtype)
        known_weights = known_weights / known_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        unknown_inputs = [
            feature - (known_weights[:, :, None, None] * torch.stack([expert[level] for expert in known_features], dim=1)).sum(dim=1)
            for level, feature in enumerate(neural_features)
        ]
        unknown_features = self.experts[-1](unknown_inputs, token_outputs["tokens"][:, -1], observations[-1])
        expert_features = known_features + [unknown_features]
        unknown_logit, unknown_severity, unknown_priority = self.unknown_detector(unknown_inputs[0])
        token_outputs["probability_logits"] = torch.cat((token_outputs["probability_logits"][:, :-1], unknown_logit[:, None]), dim=-1)
        token_outputs["probabilities"] = torch.sigmoid(token_outputs["probability_logits"])
        token_outputs["severities"] = torch.cat((token_outputs["severities"][:, :-1], unknown_severity[:, None]), dim=-1)
        token_outputs["priorities"] = torch.cat((token_outputs["priorities"][:, :-1], unknown_priority[:, None]), dim=-1)
        token_outputs["aggregate"] = (
            token_outputs["probabilities"].unsqueeze(-1) * token_outputs["tokens"]
        ).sum(dim=1) / token_outputs["probabilities"].sum(dim=1, keepdim=True).clamp_min(1e-4)
        routing = self.router(token_outputs, route_mode, oracle_labels, disabled_experts)
        purified, gate_mean = self.gating(neural_features, expert_features, routing["route"], token_outputs["aggregate"])
        coarse_clean, components, artifact_sum = self.coarse_decoder(purified, expert_features, routing["route"])
        decomp_error = y_norm - coarse_clean - artifact_sum
        if enable_residual:
            residual = self.residual_refiner(
                torch.cat((y_norm, coarse_clean, artifact_sum, decomp_error), dim=1),
                token_outputs["aggregate"],
                token_outputs["probabilities"].max(dim=-1).values,
                token_outputs["severities"].mean(dim=-1),
            )
        else:
            residual = torch.zeros_like(y_norm)
        identity_strength = self.identity_gate(y_norm, token_outputs["probabilities"], artifact_sum, residual, routing["bypass"])
        clean_norm = y_norm + identity_strength * (coarse_clean + residual - y_norm)
        return {
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
            **token_outputs,
            **routing,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
