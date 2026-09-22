"""下载 / 校验 OpenNeuro ds004784（体模数据集）。

T0.2 起本脚本承担两件事：

1. ``--download``（默认）：从 OpenNeuro 公开 S3 拉取缺失对象；
2. ``--verify``：**不下载**，只遍历本地已有文件，逐个记录 ``bytes`` 与 ``sha256``，
   写 ``data/raw/ds004784/download_manifest.json``。

校验强度与 ``data/download_manifest.json`` 保持一致（逐文件 SHA-256）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unicore_eeg import paths


BUCKET_URL = "https://s3.amazonaws.com/openneuro.org/"
PREFIX = "ds004784/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
DEFAULT_ROOT = paths.RAW_ROOT / "ds004784"
CHUNK_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or verify OpenNeuro ds004784 from public S3."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--verify", action="store_true", help="不下载，只为本地文件生成含 sha256 的清单。")
    parser.add_argument("--aria2-input", type=Path, help="Write an aria2 input file for the missing objects and exit.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def list_objects() -> list[dict[str, object]]:
    """列出 S3 上 ``ds004784/`` 前缀下的全部对象（键、字节数、ETag）。"""
    objects: list[dict[str, object]] = []
    token: str | None = None
    while True:
        query = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = BUCKET_URL + "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        for item in root.findall("s3:Contents", NS):
            key = item.findtext("s3:Key", default="", namespaces=NS)
            size = int(item.findtext("s3:Size", default="0", namespaces=NS))
            etag = item.findtext("s3:ETag", default="", namespaces=NS).strip('"')
            if key and not key.endswith("/"):
                objects.append({"key": key, "size": size, "etag": etag})
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=NS) == "true"
        token = root.findtext("s3:NextContinuationToken", default="", namespaces=NS)
        if not truncated:
            break
    return objects


def relative_path(key: str) -> Path:
    if not key.startswith(PREFIX):
        raise ValueError(f"unexpected key outside prefix: {key}")
    return Path(key[len(PREFIX) :])


def object_url(key: str) -> str:
    return BUCKET_URL + urllib.parse.quote(key, safe="/")


def sha256_of(path: Path) -> str:
    """分块计算 SHA-256；11 GB 级别也可安全用于后台运行。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def manifest_record(root: Path, obj: dict[str, object], status: str) -> dict[str, object]:
    """统一的清单条目格式：file / bytes / sha256 / status（+ key、url）。"""
    key = str(obj["key"])
    target = root / relative_path(key)
    record: dict[str, object] = {
        "file": str(target.relative_to(paths.PROJECT_ROOT)),
        "bytes": target.stat().st_size if target.exists() else 0,
        "sha256": "",
        "status": status,
        "key": key,
        "url": object_url(key),
    }
    if status == "verified":
        record["sha256"] = sha256_of(target)
    return record


def verify_local(root: Path, objects: list[dict[str, object]], workers: int) -> list[dict[str, object]]:
    """逐个核对本地文件的大小与 SHA-256，不发起任何网络下载。"""
    def check(obj: dict[str, object]) -> dict[str, object]:
        target = root / relative_path(str(obj["key"]))
        expected = int(obj["size"])
        if not target.exists():
            return manifest_record(root, obj, "missing")
        actual = target.stat().st_size
        if actual != expected:
            return manifest_record(root, obj, f"size_mismatch({actual}!={expected})")
        return manifest_record(root, obj, "verified")

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(check, obj) for obj in objects]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: str(item["key"]))
    return records


def download_one(obj: dict[str, object], out: Path, retries: int) -> dict[str, object]:
    key = str(obj["key"])
    size = int(obj["size"])
    target = out / relative_path(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == size:
        return {**obj, "path": str(target), "status": "skipped"}

    for attempt in range(1, retries + 1):
        try:
            existing = target.stat().st_size if target.exists() else 0
            mode = "ab" if 0 < existing < size else "wb"
            headers = {}
            if mode == "ab":
                headers["Range"] = f"bytes={existing}-"
            request = urllib.request.Request(object_url(key), headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response, target.open(mode) as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if target.stat().st_size == size:
                return {**obj, "path": str(target), "status": "downloaded"}
            raise IOError(f"incomplete download: {target.stat().st_size} != {size}")
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            if attempt == retries:
                return {**obj, "path": str(target), "status": "failed", "error": repr(exc)}
            time.sleep(min(2**attempt, 30))
    return {**obj, "path": str(target), "status": "failed", "error": "unknown"}


def main() -> None:
    args = parse_args()
    paths.assert_readonly_pool(args.out, operation="write ds004784 files into")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "download_manifest.json"

    if args.verify:
        objects = list_objects()
        print(f"verify mode: objects={len(objects)} root={args.out}")
        records = verify_local(args.out, objects, args.workers)
        manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        verified = sum(1 for item in records if item["status"] == "verified")
        total_bytes = sum(int(item["bytes"]) for item in records)
        print(f"verified={verified}/{len(records)} bytes={total_bytes} manifest={manifest_path}")
        bad = [item for item in records if item["status"] != "verified"]
        if bad:
            print(f"non-verified entries: {len(bad)}")
            for item in bad[:20]:
                print(f"  {item['status']:>28s}  {item['file']}")
            raise SystemExit(1)
        return

    objects = list_objects()
    total_size = sum(int(obj["size"]) for obj in objects)
    print(f"objects={len(objects)} size_gib={total_size / 1024**3:.2f} out={args.out}")
    if args.dry_run:
        return
    if args.aria2_input:
        lines = []
        missing = 0
        for obj in objects:
            key = str(obj["key"])
            size = int(obj["size"])
            target = args.out / relative_path(key)
            if target.exists() and target.stat().st_size == size:
                continue
            missing += 1
            lines.extend([
                object_url(key),
                f"  dir={target.parent.resolve()}",
                f"  out={target.name}",
            ])
        args.aria2_input.parent.mkdir(parents=True, exist_ok=True)
        args.aria2_input.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"aria2_input={args.aria2_input} missing={missing}")
        return

    results: list[dict[str, object]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, obj, args.out, args.retries) for obj in objects]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1
            print(f"[{done}/{len(objects)}] {result['status']} {result['key']}")

    status_by_key = {str(item["key"]): str(item["status"]) for item in results}
    records = [
        manifest_record(args.out, obj, "verified" if status_by_key.get(str(obj["key"])) in {"downloaded", "skipped"} else status_by_key.get(str(obj["key"]), "unknown"))
        for obj in objects
    ]
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    failed = [item for item in records if item["status"] != "verified"]
    if failed:
        raise SystemExit(f"failed files: {len(failed)}")
    print(f"download complete: {args.out}")


if __name__ == "__main__":
    main()
