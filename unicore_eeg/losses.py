from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class UniCORELossWeights:
    final_charbonnier: float = 1.0
    final_corr: float = 0.2
    final_diff: float = 0.2
    final_stft: float = 0.05
    coarse: float = 0.5
    artifact: float = 0.35
    decomp: float = 0.3
    route: float = 0.7
    identity: float = 0.35
    clean_presence: float = 0.6
    gate_sparse: float = 0.01
    unknown_energy: float = 0.02
    unknown_replacement: float = 0.05
    diversity: float = 0.05
    load_balance: float = 0.02
    hard_negative: float = 0.05


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)


def charbonnier(pred: Tensor, target: Tensor, eps: float = 1e-3) -> Tensor:
    return torch.sqrt((pred - target).square() + eps * eps).mean()


def _channel_mean(values: Tensor, channel_mask: Tensor | None, channel_axis: int = 1) -> Tensor:
    """Mean over valid electrode entries, ignoring batch padding."""
    if channel_mask is None:
        return values.mean()
    mask = channel_mask.to(values.dtype)
    if channel_axis == 2:
        mask = mask[:, None, :, None]
    else:
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def correlation_loss(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    pred_centered = pred - pred.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    return 1.0 - F.cosine_similarity(pred_centered, target_centered, dim=-1, eps=eps).mean()


def multi_resolution_stft_loss(pred: Tensor, target: Tensor, channel_mask: Tensor | None = None) -> Tensor:
    batch, channels, _ = pred.shape
    pred = pred.reshape(batch * channels, -1)
    target = target.reshape(batch * channels, -1)
    losses = []
    for n_fft in (64, 128, 256):
        window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        pred_spec = torch.stft(pred, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
        target_spec = torch.stft(target, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
        per_channel = (torch.log1p(pred_spec.abs()) - torch.log1p(target_spec.abs())).abs().mean(dim=(1, 2)).reshape(batch, channels)
        if channel_mask is None:
            losses.append(per_channel.mean())
        else:
            losses.append(_channel_mean(per_channel, channel_mask))
    return torch.stack(losses).mean()


class UniCORELoss(nn.Module):
    def __init__(self, weights: UniCORELossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or UniCORELossWeights()

    def forward(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        mad = outputs["mad"].clamp_min(1e-5)
        channel_mask = batch.get("channel_mask", outputs.get("channel_mask"))
        y_norm = (batch["noisy"] - outputs["median"]) / mad
        clean_norm = (batch["clean"] - outputs["median"]) / mad
        artifact_norm = batch["artifacts"] / mad.unsqueeze(1)

        final = (
            self.weights.final_charbonnier * _channel_mean(torch.sqrt((outputs["clean_norm"] - clean_norm).square() + 1e-6), channel_mask)
            + self.weights.final_corr * _channel_mean(1.0 - F.cosine_similarity(
                outputs["clean_norm"] - outputs["clean_norm"].mean(-1, keepdim=True),
                clean_norm - clean_norm.mean(-1, keepdim=True), dim=-1
            ), channel_mask)
            + self.weights.final_diff * _channel_mean(torch.diff(outputs["clean_norm"], dim=-1).sub(torch.diff(clean_norm, dim=-1)).abs(), channel_mask)
            + self.weights.final_stft * multi_resolution_stft_loss(outputs["clean_norm"], clean_norm, channel_mask)
        )
        coarse = _channel_mean(F.smooth_l1_loss(outputs["coarse_clean_norm"], clean_norm, reduction="none"), channel_mask)
        component_mask = batch.get("component_mask", batch["labels"]).bool()
        artifact_error = F.smooth_l1_loss(outputs["artifact_components_norm"], artifact_norm, reduction="none")
        if channel_mask is not None:
            component_mask = component_mask.to(artifact_error.dtype)[:, :, None, None] * channel_mask.to(artifact_error.dtype)[:, None, :, None]
            artifact_mask = component_mask.expand_as(artifact_error)
            artifact = (artifact_error * artifact_mask).sum() / artifact_mask.sum().clamp_min(1.0)
        else:
            artifact = _masked_mean(artifact_error, component_mask)
        decomp = _channel_mean((y_norm - outputs["coarse_clean_norm"] - outputs["artifact_sum_norm"]).abs(), channel_mask)

        label_mask = batch.get("label_mask", torch.ones_like(batch["labels"])).bool()
        route_pos_weight = batch["labels"].new_tensor([3.0, 3.0, 3.0, 3.5, 3.5, 5.0])
        route_bce = _masked_mean(
            F.binary_cross_entropy_with_logits(
                outputs["probability_logits"].float(),
                batch["labels"].float(),
                pos_weight=route_pos_weight,
                reduction="none",
            ),
            label_mask,
        )
        severity_mask = label_mask & batch["labels"].bool()
        severity = _masked_mean(
            F.smooth_l1_loss(outputs["severities"].float(), batch["severity"].float(), reduction="none"),
            severity_mask,
        )
        pre_route_sparse = outputs["route_scores"].mean()
        active_count = outputs["active_mask"].float().sum(dim=-1).mean() / outputs["active_mask"].size(-1)
        route_loss = route_bce + 0.2 * severity + 0.01 * pre_route_sparse + 0.01 * active_count

        is_clean = batch["is_clean"].bool()
        artifact_present = ~is_clean
        artifact_target = artifact_present.float()
        artifact_count = artifact_target.sum()
        clean_count = artifact_target.numel() - artifact_count
        presence_pos_weight = (clean_count / artifact_count.clamp_min(1.0)).clamp(0.5, 4.0)
        clean_presence = F.binary_cross_entropy_with_logits(
            outputs["artifact_presence_logit"].float(),
            artifact_target,
            pos_weight=presence_pos_weight,
        )
        clean_false_alarm = _masked_mean(outputs["probabilities"].amax(dim=-1), is_clean)
        artifact_miss = _masked_mean(F.relu(0.65 - outputs["artifact_presence_probability"]), artifact_present)
        if channel_mask is None:
            identity_error = (outputs["clean_norm"] - y_norm).abs().mean(dim=(1, 2))
        else:
            identity_values = (outputs["clean_norm"] - y_norm).abs() * channel_mask.to(y_norm.dtype).unsqueeze(-1)
            identity_error = identity_values.sum(dim=(1, 2)) / (channel_mask.sum(dim=1).clamp_min(1) * y_norm.size(-1))
        identity_target = torch.where(is_clean, torch.zeros_like(outputs["identity_strength"]), torch.full_like(outputs["identity_strength"], 0.75))
        identity = (
            _masked_mean(identity_error, is_clean)
            + F.smooth_l1_loss(outputs["identity_strength"], identity_target)
            + 0.1 * clean_false_alarm
            + 0.1 * artifact_miss
        )

        unknown_energy_tensor = outputs["artifact_components_norm"][:, -1].square()
        if channel_mask is None:
            unknown_energy_per_sample = (unknown_energy_tensor.mean(dim=(1, 2)) + 1e-8).sqrt()
        else:
            unknown_weight = channel_mask.to(unknown_energy_tensor.dtype).unsqueeze(-1)
            unknown_energy_per_sample = ((unknown_energy_tensor * unknown_weight).sum(dim=(1, 2)) / (channel_mask.sum(dim=1).clamp_min(1) * unknown_energy_tensor.size(-1)) + 1e-8).sqrt()
        unknown_target = batch["labels"][:, -1].bool()
        unknown_energy = _masked_mean(unknown_energy_per_sample, ~unknown_target)
        known_present = batch["labels"][:, :-1].bool().any(dim=-1) & ~unknown_target
        known_route = outputs["route"][:, :-1].amax(dim=-1)
        replacement = _masked_mean(F.relu(outputs["route"][:, -1] - known_route + 0.05), known_present)

        # Expert diversity uses the pooled feature emitted by each expert.  A
        # positive cosine similarity is penalized; anti-correlated experts are
        # allowed and are useful for complementary artifact mechanisms.
        expert_vectors = outputs.get("expert_feature_vectors")
        if expert_vectors is not None:
            stacked = F.normalize(expert_vectors.float(), dim=-1)
            similarities = torch.matmul(stacked, stacked.transpose(1, 2))
            off_diagonal = ~torch.eye(stacked.size(1), device=stacked.device, dtype=torch.bool)
            diversity = F.relu(similarities[:, off_diagonal]).mean()
        else:
            diversity = outputs["probabilities"].new_zeros(())

        usage = outputs["route"].float().mean(dim=0)
        target_usage = torch.full_like(usage, 1.0 / usage.numel())
        load_balance = (usage - target_usage).square().mean()

        cardiac = batch["labels"][:, 3].bool()
        hard_negative = _masked_mean(
            F.relu(outputs["route"][:, 1] - outputs["route"][:, 3]), cardiac
        )

        total = (
            final
            + self.weights.coarse * coarse
            + self.weights.artifact * artifact
            + self.weights.decomp * decomp
            + self.weights.route * route_loss
            + self.weights.identity * identity
            + self.weights.clean_presence * clean_presence
            + self.weights.gate_sparse * outputs["gate_mean"]
            + self.weights.unknown_energy * unknown_energy
            + self.weights.unknown_replacement * replacement
            + self.weights.diversity * diversity
            + self.weights.load_balance * load_balance
            + self.weights.hard_negative * hard_negative
        )
        logs = {
            "loss": total.detach(),
            "final": final.detach(),
            "coarse": coarse.detach(),
            "artifact": artifact.detach(),
            "decomp": decomp.detach(),
            "route": route_loss.detach(),
            "identity": identity.detach(),
            "clean_presence": clean_presence.detach(),
            "unknown_energy": unknown_energy.detach(),
            "unknown_replacement": replacement.detach(),
            "diversity": diversity.detach(),
            "load_balance": load_balance.detach(),
            "hard_negative": hard_negative.detach(),
            "expert_usage": usage.detach(),
            "routing_confusion_before": (outputs["route_scores"].argmax(dim=-1) == 1).float().masked_select(cardiac).mean().detach(),
            "routing_confusion_after": (outputs["route"].argmax(dim=-1) == 1).float().masked_select(cardiac).mean().detach(),
        }
        return total, logs
