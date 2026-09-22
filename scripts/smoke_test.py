from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from unicore_eeg import UniCOREEG, UniCOREEGConfig
from unicore_eeg.losses import UniCORELoss
from unicore_eeg.model import count_parameters
from unicore_eeg.synthetic import SyntheticEEGDataset


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    config = UniCOREEGConfig(in_channels=1, window_size=1000)
    model = UniCOREEG(config).to(device)
    dataset = SyntheticEEGDataset(samples=8, channels=1, length=1000)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    batch = {key: value.to(device) for key, value in batch.items()}
    loss_fn = UniCORELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = model(
            batch["noisy"],
            metadata=batch["metadata"],
            disabled_experts=batch["disabled_experts"],
        )
        loss, logs = loss_fn(outputs, batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"device={device}")
    print(f"params={count_parameters(model):,}")
    print(f"loss={logs['loss'].item():.4f}")
    print(f"clean_shape={tuple(outputs['clean'].shape)}")
    print(f"route_mean={outputs['route'].mean().item():.4f}")
    print(f"route_shape={tuple(outputs['route'].shape)}")
    print(f"bypass_rate={outputs['bypass'].float().mean().item():.4f}")


if __name__ == "__main__":
    main()
