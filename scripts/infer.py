from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from unicore_eeg import UniCOREEG, UniCOREEGConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UniCORE-EEG inference on a .npy EEG window.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-mode", choices=("learned", "all"), default="learned")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = UniCOREEGConfig(**checkpoint["config"])
    model = UniCOREEG(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    array = np.load(args.input).astype("float32")
    if array.ndim == 2:
        array = array[None, ...]
    eeg = torch.from_numpy(array).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = model(eeg, route_mode=args.route_mode)
    cleaned = outputs["clean"].detach().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, cleaned)
    print(f"saved={args.output}")
    print(f"probabilities={outputs['probabilities'].detach().cpu().numpy()}")


if __name__ == "__main__":
    main()
