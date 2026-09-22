from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests
from tqdm import tqdm


EEGDENOISENET_FILES = {
    "EEG_all_epochs_512hz.npy": "https://gin.g-node.org/NCClab/EEGdenoiseNet/raw/master/data/EEG_all_epochs_512hz.npy",
    "EOG_all_epochs.npy": "https://gin.g-node.org/NCClab/EEGdenoiseNet/raw/master/data/EOG_all_epochs.npy",
    "EMG_all_epochs_512hz.npy": "https://gin.g-node.org/NCClab/EEGdenoiseNet/raw/master/data/EMG_all_epochs_512hz.npy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        remote_size = int(requests.head(url, allow_redirects=True, timeout=60).headers.get("content-length", 0))
        if remote_size and target.stat().st_size == remote_size:
            print(f"exists={target}")
            return
    existing = target.stat().st_size if target.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    mode = "ab" if existing and response.status_code == 206 else "wb"
    if mode == "wb":
        existing = 0
    total = int(response.headers.get("content-length", 0)) + existing
    with target.open(mode) as handle, tqdm(
        total=total,
        initial=existing,
        unit="B",
        unit_scale=True,
        desc=target.name,
    ) as progress:
        for block in response.iter_content(1024 * 1024):
            if block:
                handle.write(block)
                progress.update(len(block))


def download_eegdenoisenet(root: Path) -> list[dict[str, object]]:
    target_dir = root / "eegdenoisenet"
    records = []
    for filename, url in EEGDENOISENET_FILES.items():
        target = target_dir / filename
        download_file(url, target)
        records.append({"file": str(target), "bytes": target.stat().st_size, "sha256": sha256(target), "url": url})
    return records


def download_mitdb(root: Path, records: list[str]) -> list[dict[str, object]]:
    import wfdb

    target = root / "mitdb"
    target.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", str(target), records=records, keep_subdirs=False, overwrite=False)
    return [
        {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(target.iterdir())
        if path.is_file()
    ]


def download_physiomotion(root: Path, subjects: list[str], metadata_only: bool) -> list[dict[str, object]]:
    from openneuro import download

    target = root / "physiomotion"
    include = ["/*.json", "/*.tsv", "README", "CHANGES"]
    if not metadata_only:
        include.extend(subjects)
    download(dataset="ds006386", tag="1.0.1", target_dir=target, include=include, max_concurrent_downloads=4)
    return [
        {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(target.rglob("*"))
        if path.is_file()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public source data used by UniCORE-EEG.")
    parser.add_argument("--dataset", choices=("all", "eegdenoisenet", "mitdb", "physiomotion"), default="all")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--mitdb-records", nargs="+", default=["100", "101", "102", "103", "105"])
    parser.add_argument("--motion-subjects", nargs="+", default=["sub-1"])
    parser.add_argument("--motion-metadata-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    selected = ("eegdenoisenet", "mitdb", "physiomotion") if args.dataset == "all" else (args.dataset,)
    manifest_path = args.root.parent / "download_manifest.json"
    existing_files: dict[str, object] = {}
    if manifest_path.exists():
        existing_files = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
    manifest: dict[str, object] = {
        "schema_version": 1,
        "sources": {
            "eegdenoisenet": {
                "homepage": "https://gin.g-node.org/NCClab/EEGdenoiseNet",
                "license": "CC0-1.0",
                "purpose": "clean EEG, EOG and EMG source epochs",
            },
            "mitdb": {
                "homepage": "https://physionet.org/content/mitdb/1.0.0/",
                "license": "ODC-By-1.0",
                "purpose": "controlled cardiac artifact source waveforms",
            },
            "physiomotion": {
                "homepage": "https://openneuro.org/datasets/ds006386/versions/1.0.1",
                "license": "CC0-1.0",
                "purpose": "real EEG motion artifacts with point-level labels",
                "download_scope": "metadata" if args.motion_metadata_only else args.motion_subjects,
            },
        },
        "files": existing_files,
    }
    if "eegdenoisenet" in selected:
        manifest["files"]["eegdenoisenet"] = download_eegdenoisenet(args.root)
    if "mitdb" in selected:
        manifest["files"]["mitdb"] = download_mitdb(args.root, args.mitdb_records)
    if "physiomotion" in selected:
        manifest["files"]["physiomotion"] = download_physiomotion(args.root, args.motion_subjects, args.motion_metadata_only)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
