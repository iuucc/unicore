"""在 validation 上冻结路由阈值，并可对冻结后的 test 运行一次诊断。

该脚本不读取或调整 test 指标；``--split val`` 只产生校准结果，``--split test``
必须传入 validation JSON 中冻结的阈值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unicore_eeg import ARTIFACT_NAMES, UniCOREEG, UniCOREEGConfig
from unicore_eeg.batching import collate_variable_channels
from unicore_eeg.routing_metrics import masked_routing_metrics, masked_threshold_calibration
from unicore_eeg.synthetic import SyntheticEEGDataset


def _collect(
    model: UniCOREEG,
    loader: DataLoader,
    device: torch.device,
    *,
    use_spatial: bool,
    coords_mask: torch.Tensor | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels, masks, probs, logits, presence, routes, disabled, sample_ids = [], [], [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            spatial_kwargs = {}
            if use_spatial:
                spatial_kwargs = {"coords": batch["coords"], "coords_mask": coords_mask}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch["noisy"], metadata=batch["metadata"],
                    **spatial_kwargs, disabled_experts=batch["disabled_experts"], route_mode="learned")
            labels.append(batch["labels"].float().cpu().numpy())
            masks.append(batch["label_mask"].float().cpu().numpy())
            probs.append(output["probabilities"].float().cpu().numpy())
            logits.append(output["probability_logits"].float().cpu().numpy())
            presence.append(output["artifact_presence_probability"].float().cpu().numpy())
            routes.append(output["route"].float().cpu().numpy())
            disabled.append(batch["disabled_experts"].float().cpu().numpy())
            sample_ids.append(batch["sample_id"].cpu().numpy())
    return (
        np.concatenate(labels),
        np.concatenate(masks),
        np.concatenate(probs),
        np.concatenate(logits),
        np.concatenate(presence),
        np.concatenate(routes),
        np.concatenate(disabled),
        np.concatenate(sample_ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--thresholds", type=Path, help="validation routing JSON；test 模式必须提供")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--use-public-sources", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--montage")
    parser.add_argument("--montage-channels", nargs="+")
    parser.add_argument("--bypass-source", choices=("presence_head", "max_probability"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing CPU calibration")
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = UniCOREEGConfig(**checkpoint["config"])
    if args.bypass_source:
        model_config.bypass_source = args.bypass_source
    model = UniCOREEG(model_config).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    coords = None
    coords_mask = None
    if args.montage:
        from unicore_eeg.montage import load_montage, resolve_channels
        spec = load_montage(args.montage)
        names = list(args.montage_channels or spec.channels)
        _, coords, coords_mask = resolve_channels(names, spec)
        coords = spec.to_head_ras(coords.float())
        args.channels = len(names)
    dataset = SyntheticEEGDataset(samples=args.samples, channels=args.channels, seed=args.seed,
        mode="first_experiment", split=args.split, use_public_sources=args.use_public_sources,
        data_root=args.data_root, montage_coords=coords, montage_mask=coords_mask)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_variable_channels)
    device_coords_mask = None if coords_mask is None or bool(coords_mask.all()) else coords_mask.to(device)
    labels, masks, probs, logits, presence, routes, disabled, sample_ids = _collect(
        model, loader, device, use_spatial=args.montage is not None, coords_mask=device_coords_mask
    )
    if args.split == "val":
        thresholds = masked_threshold_calibration(labels, probs, masks)
    else:
        if args.thresholds is None:
            raise SystemExit("test evaluation requires --thresholds from a completed validation calibration")
        frozen = json.loads(args.thresholds.read_text(encoding="utf-8"))["thresholds"]
        thresholds = np.asarray([float(frozen[name]) for name in ARTIFACT_NAMES])
    rows, summary = masked_routing_metrics(labels, probs, thresholds, masks)
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    payload = {"split": args.split, "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "git_sha": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv, "data_config": {"use_public_sources": args.use_public_sources, "data_root": str(args.data_root), "montage": args.montage, "montage_channels": args.montage_channels},
        "seed": args.seed, "samples": args.samples, "bypass_source": model_config.bypass_source,
        "thresholds": dict(zip(ARTIFACT_NAMES, thresholds.tolist())),
        "metrics": rows, "summary": summary, "artifact_presence_mean": float(presence.mean()),
        "source_split_manifest": dataset.public.split_manifest() if dataset.public else None,
        "per_sample": {
            "sample_id": sample_ids.tolist(),
            "labels": labels.tolist(),
            "label_mask": masks.tolist(),
            "probabilities": probs.tolist(),
            "logits": logits.tolist(),
            "disabled_experts": disabled.tolist(),
        }}
    payload["expert_activation_matrix"] = routes.mean(axis=0).tolist()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"routing_{args.split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
