"""Batch EEG records with different channel counts by padding and masking."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate


# Field name -> channel axis in one unbatched sample.
_CHANNEL_AXES = {
    "noisy": 0,
    "clean": 0,
    "artifacts": 1,
    "mixing_matrix": 1,
    "coords": 0,
    "coords_mask": 0,
}


def collate_variable_channels(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    """Stack a batch whose records have different C, padding only the channel axis.

    The returned ``channel_mask`` is true for observed electrodes and false for
    padding. Non-channel targets such as labels, severity, and metadata are
    collated normally.
    """
    if not samples:
        raise ValueError("cannot collate an empty batch")
    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples):
        raise ValueError("all samples in a batch must expose the same fields")

    channel_counts = [int(torch.as_tensor(sample["noisy"]).shape[0]) for sample in samples]
    max_channels = max(channel_counts)
    result: dict[str, object] = {}
    for key in samples[0]:
        if key == "channel_mask":
            continue
        values = [sample[key] for sample in samples]
        if key not in _CHANNEL_AXES or not all(isinstance(value, Tensor) for value in values):
            result[key] = default_collate(values)
            continue
        axis = _CHANNEL_AXES[key]
        padded: list[Tensor] = []
        for value, count in zip(values, channel_counts):
            assert isinstance(value, Tensor)
            if value.shape[axis] != count:
                raise ValueError(f"{key} channel axis has {value.shape[axis]} entries, expected {count}")
            shape = list(value.shape)
            shape[axis] = max_channels
            target = value.new_zeros(shape)
            slices = [slice(None)] * value.ndim
            slices[axis] = slice(0, count)
            target[tuple(slices)] = value
            padded.append(target)
        result[key] = torch.stack(padded)

    channel_mask = torch.zeros(len(samples), max_channels, dtype=torch.bool)
    for index, count in enumerate(channel_counts):
        channel_mask[index, :count] = True
    result["channel_mask"] = channel_mask
    return result
