from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BUCKET_URL = "https://s3.amazonaws.com/openneuro.org/"
PREFIX = "ds004784/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenNeuro ds004784 / NEMAR on004784 from public S3.")
    parser.add_argument("--out", type=Path, default=Path("data/raw/on004784"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--aria2-input", type=Path, help="Write an aria2 input file for the missing objects and exit.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def list_objects() -> list[dict[str, object]]:
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
    args.out.mkdir(parents=True, exist_ok=True)
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

    manifest: list[dict[str, object]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, obj, args.out, args.retries) for obj in objects]
        for future in as_completed(futures):
            result = future.result()
            manifest.append(result)
            done += 1
            print(f"[{done}/{len(objects)}] {result['status']} {result['key']}")

    manifest.sort(key=lambda item: str(item["key"]))
    (args.out / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    failed = [item for item in manifest if item["status"] == "failed"]
    if failed:
        raise SystemExit(f"failed files: {len(failed)}")
    print(f"download complete: {args.out}")


if __name__ == "__main__":
    main()
