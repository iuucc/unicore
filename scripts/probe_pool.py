"""外部数据池（POOL）可达性与 schema 探测（手册 T0.3）。

对 POOL 下 12 个数据集逐个实测并写出 ``runs/_pool_probe/pool_probe.json``：

* 实际文件数与总字节；按扩展名统计；一级子目录布局；
* EDF / GDF / BDF / BrainVision(.vhdr) / EEGLAB(.set) 用 ``mne.io.read_raw_*``
  以 ``preload=False`` 读第一个文件的 ``sfreq`` / ``n_channels`` / ``n_times`` / ``ch_names``；
* ``.npz`` 用 ``np.load(...).files`` dump 每个键的 shape 与 dtype（**不读全部数据**）；
* ``.mat`` 用 ``scipy.io.loadmat(..., simplify_cells=True)`` 打印顶层键。

本脚本**只读**：所有目标路径都被 ``paths.assert_readonly_pool`` 校验，输出目录不得落在池内。
手册 §3.2 纪律 1：表中的"待读"字段必须由本脚本实测回填，不得凭记忆填写。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import warnings
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from unicore_eeg import paths


#: 采样率可能的键名（用于从 npz/mat 里挖频率元信息）。
RATE_KEYS = (
    "sfreq",
    "srate",
    "fs",
    "sample_rate",
    "sampling_rate",
    "sampling_frequency",
    "rate",
    "SampleRate",
    "samp_freq",
)

#: 以 mne 可读的原始格式为优先顺序。
MNE_SUFFIXES = (".edf", ".gdf", ".bdf", ".vhdr", ".set")

#: 每个数据集最多 dump 多少个 npz / mat 的 schema。
SCHEMA_SAMPLES = 2

#: 每个数据集最多读多少条原始记录用于核对采样率/通道数是否整库一致。
RAW_SAMPLES = 3

#: `scipy.io.loadmat` 会把整个文件读进内存，本机系统内存紧张（31.8 GiB，可用常低于 20 GiB），
#: 因此对超过该阈值的 .mat 只记录大小与跳过原因，不做 loadmat。
MAX_MAT_BYTES = 200 * 1024 * 1024

#: 字节统计时忽略的目录（缓存与元数据不计入"数据体积"判断，但仍会单独报告）。
SKIP_DIR_NAMES = {"__pycache__"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the read-only external EEG pool (T0.3).")
    parser.add_argument("--out", type=Path, default=paths.RUNS_ROOT / "_pool_probe" / "pool_probe.json")
    parser.add_argument("--datasets", nargs="+", default=list(paths.POOL_DATASET_NAMES))
    parser.add_argument("--schema-samples", type=int, default=SCHEMA_SAMPLES)
    return parser.parse_args()


# --------------------------------------------------------------------------
# 文件遍历
# --------------------------------------------------------------------------


def walk_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        base = Path(dirpath)
        files.extend(base / name for name in filenames)
    return files


def human_bytes(value: int) -> str:
    step = 1024.0
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < step:
            return f"{size:.2f} {unit}"
        size /= step
    return f"{size:.2f} PiB"


# --------------------------------------------------------------------------
# 各类文件的 schema 提取
# --------------------------------------------------------------------------


def probe_mne_file(path: Path) -> dict[str, Any]:
    """用 mne 读一个原始文件的头部信息（preload=False，不加载数据）。"""
    import mne

    warnings.filterwarnings("ignore")
    readers = {
        ".edf": mne.io.read_raw_edf,
        ".gdf": mne.io.read_raw_gdf,
        ".bdf": mne.io.read_raw_bdf,
        ".vhdr": mne.io.read_raw_brainvision,
        ".set": mne.io.read_raw_eeglab,
    }
    reader = readers[path.suffix.lower()]
    raw = reader(path, preload=False, verbose="ERROR")
    return {
        "file": str(path),
        "reader": reader.__name__,
        "sfreq": float(raw.info["sfreq"]),
        "n_channels": int(len(raw.ch_names)),
        "n_times": int(raw.n_times),
        "duration_s": round(float(raw.n_times / raw.info["sfreq"]), 3),
        "channel_types": dict(Counter(str(kind) for kind in raw.get_channel_types())),
        "ch_names": list(raw.ch_names),
        "highpass": float(raw.info.get("highpass", 0.0) or 0.0),
        "lowpass": float(raw.info.get("lowpass", 0.0) or 0.0),
    }


def describe_array(array: Any) -> dict[str, Any]:
    """把任意 mat/npz 值归纳成可序列化的小字典。

    注意：**不要**用 ``isinstance(x, np.ndarray.dtype)`` 判断 dtype——NumPy 2.x 里
    ``np.ndarray.dtype`` 是 getset 描述符不是类型，会抛
    ``TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union``。
    """
    if isinstance(array, np.ndarray):
        info: dict[str, Any] = {
            "kind": "ndarray",
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
        }
        # 0 维数组就是标量：把值带出来，否则 sample_rate 这类元信息会被丢掉。
        if array.shape == ():
            info["value"] = array.item()
        # 小体量字符串数组通常是通道名 / 标签名，直接带出来供 §3.1 回填与 montage 使用。
        elif array.dtype.kind in "US" and array.ndim == 1 and array.size <= 128:
            info["values"] = [str(item) for item in array.tolist()]
        # 小体量整数数组可能是标签或索引，同理带出来。
        elif array.dtype.kind in "iu" and array.ndim == 1 and array.size <= 16:
            info["values"] = [int(item) for item in array.tolist()]
        return info
    if isinstance(array, (bool, np.bool_, int, float, np.integer, np.floating, str)):
        return {"kind": type(array).__name__, "value": array.item() if hasattr(array, "item") else array}
    if isinstance(array, Mapping):
        keys = [str(key) for key in array.keys()]
        return {"kind": "mapping", "n_keys": len(keys), "keys": keys[:20]}
    if isinstance(array, (list, tuple)):
        return {"kind": "sequence", "length": len(array), "item_kinds": sorted({type(item).__name__ for item in array})}
    return {"kind": type(array).__name__, "repr": repr(array)[:120]}


def probe_npz_file(path: Path) -> dict[str, Any]:
    """dump npz 的键名与各键 shape/dtype；只读元信息，不物化整个数组。"""
    record: dict[str, Any] = {"file": str(path)}
    with np.load(path, mmap_mode="r", allow_pickle=False) as archive:
        keys = list(archive.files)
        record["keys"] = keys
        schema: dict[str, Any] = {}
        for key in keys:
            try:
                schema[key] = describe_array(archive[key])
            except Exception as error:  # 对象数组等无法 mmap 的键
                schema[key] = {"kind": "unreadable", "error": repr(error)}
        record["schema"] = schema
        record["rate_like_keys"] = {
            key: schema[key]["value"]
            for key in keys
            if key.lower() in {name.lower() for name in RATE_KEYS} and "value" in schema[key]
        }
    record["size_bytes"] = path.stat().st_size
    return record


def probe_mat_file(path: Path) -> dict[str, Any]:
    """打印 mat 的顶层键；仅保留标量与形状，避免把大矩阵读进内存。

    ``simplify_cells=True`` 在 NumPy 2.4 + SciPy 1.17 下对部分 v7 文件会抛
    ``TypeError: isinstance() arg 2 must be a type...``，因此保留一次不带
    simplify 的兜底重试，并记录实际生效的模式。
    """
    from scipy.io import loadmat

    record: dict[str, Any] = {"file": str(path), "size_bytes": path.stat().st_size}
    if path.stat().st_size > MAX_MAT_BYTES:
        record["skipped"] = f"larger than MAX_MAT_BYTES ({MAX_MAT_BYTES}); loadmat would materialize it"
        record["top_level"] = []
        return record

    payload = None
    for simplify in (True, False):
        try:
            payload = loadmat(path, simplify_cells=simplify)
            record["simplify_cells"] = simplify
            break
        except NotImplementedError as error:
            record["error"] = f"loadmat unsupported (v7.3/HDF5?): {error}"
            record["top_level"] = []
            return record
        except Exception as error:  # noqa: BLE001 - 记录后换模式重试
            record.setdefault("attempt_errors", []).append(
                f"simplify_cells={simplify}: {type(error).__name__}: {error}"
            )
    if payload is None:
        record["top_level"] = []
        return record

    top_level = [key for key in payload if not key.startswith("__")]
    record["top_level"] = top_level
    record["fields"] = {key: describe_array(payload[key]) for key in top_level[:20]}
    return record


# --------------------------------------------------------------------------
# 单数据集探测
# --------------------------------------------------------------------------


def first_matching(files: Iterable[Path], patterns: Iterable[str]) -> Path | None:
    """按 pattern 顺序找第一个匹配文件；pattern 支持 ``**/`` 递归。"""
    materialized = list(files)
    for pattern in patterns:
        matches = sorted(path for path in materialized if path.match(pattern))
        if matches:
            return matches[0]
    return None


def probe_dataset(name: str, schema_samples: int) -> dict[str, Any]:
    entry = paths.pool_entry(name)
    record: dict[str, Any] = {
        "name": name,
        "root": str(entry.root),
        "declared": str(entry.declared),
        "roles": list(entry.roles),
        "overridden": entry.overridden,
        "exists": entry.exists,
    }
    # 只读审计分两半：
    #   1) 正面断言探测目标确实落在只读池内（用 is_in_pool，而不是写守卫）；
    #   2) 记录基线指纹（文件数 + 最大 mtime），供 main() 事后复核池未被改动。
    # 写守卫 assert_readonly_pool 只用于**输出**路径（见 main()），因为它的语义是
    # "禁止写入池内"，对本函数的读操作不适用。
    if not paths.is_in_pool(entry.root):
        raise RuntimeError(f"{name} resolves outside the read-only pool: {entry.root}")
    if not entry.root.is_dir():
        record.update(
            {
                "status": "empty" if entry.name == "tuar" else "missing",
                "file_count": 0,
                "total_bytes": 0,
                "readonly_audit": {"root_in_pool": True, "writes_attempted": 0},
            }
        )
        return record

    files = walk_files(entry.root)
    if not files:
        record.update(
            {
                "status": "empty",
                "file_count": 0,
                "total_bytes": 0,
                "readonly_audit": {"root_in_pool": True, "writes_attempted": 0},
            }
        )
        return record

    sizes = {path: path.stat() for path in files}
    suffixes = Counter(path.suffix.lower() for path in files)
    record.update(
        {
            "file_count": len(files),
            "total_bytes": sum(stat.st_size for stat in sizes.values()),
            "by_extension": {suffix or "<none>": count for suffix, count in suffixes.most_common()},
            "top_level_dirs": sorted(child.name for child in entry.root.iterdir() if child.is_dir()),
            "top_level_files": sorted(child.name for child in entry.root.iterdir() if child.is_file())[:40],
            "fingerprint": {
                "file_count": len(files),
                "total_bytes": sum(stat.st_size for stat in sizes.values()),
                "max_mtime_ns": max(stat.st_mtime_ns for stat in sizes.values()),
            },
            "readonly_audit": {"root_in_pool": True, "writes_attempted": 0},
        }
    )

    errors: list[str] = []
    notes: list[str] = []

    # --- 原始信号文件：读前若干条，检查采样率/通道数是否整库一致 ---
    mne_candidates = [path for path in files if path.suffix.lower() in MNE_SUFFIXES]
    if mne_candidates:
        for suffix in MNE_SUFFIXES:
            same_kind = sorted(path for path in mne_candidates if path.suffix.lower() == suffix)
            if not same_kind:
                continue
            record.setdefault("raw_file_counts", {})[suffix] = len(same_kind)
            samples = []
            for path in same_kind[:RAW_SAMPLES]:
                try:
                    samples.append(probe_mne_file(path))
                except Exception as error:
                    message = f"{suffix} {path.name}: {type(error).__name__}: {error}"
                    # 标注文件（如 Sleep-EDF 的 Hypnogram）本身不含信号通道，
                    # mne 会报 "yielded no channels"。这不是库缺陷，记为 note。
                    if "yielded no channels" in str(error):
                        notes.append(message)
                    else:
                        errors.append(message)
            if samples:
                record["primary_raw"] = samples[0]
                record["raw_samples_probed"] = len(samples)
                record["raw_variants"] = [
                    {
                        "file": item["file"],
                        "sfreq": item["sfreq"],
                        "n_channels": item["n_channels"],
                        "n_times": item["n_times"],
                        "duration_s": item["duration_s"],
                    }
                    for item in samples
                ]
                distinct_rates = sorted({item["sfreq"] for item in samples})
                distinct_channels = sorted({item["n_channels"] for item in samples})
                record["raw_uniform"] = {
                    "sfreq": len(distinct_rates) == 1,
                    "n_channels": len(distinct_channels) == 1,
                }
                break

    # --- npz schema ---
    npz_files = sorted(path for path in files if path.suffix.lower() == ".npz")
    if npz_files:
        record["npz_file_count"] = len(npz_files)
        schemas = []
        for path in npz_files[:schema_samples]:
            try:
                schemas.append(probe_npz_file(path))
            except Exception as error:
                errors.append(f"npz {path.name}: {type(error).__name__}: {error}")
        record["npz_schemas"] = schemas

    # --- mat 顶层键 ---
    mat_files = sorted(path for path in files if path.suffix.lower() == ".mat")
    if mat_files:
        record["mat_file_count"] = len(mat_files)
        mat_records = []
        for path in mat_files[:schema_samples]:
            try:
                mat_records.append(probe_mat_file(path))
            except Exception as error:
                errors.append(f"mat {path.name}: {type(error).__name__}: {error}")
        record["mat_schemas"] = mat_records

    # --- 从 npz/mat 里挖采样率，供 §3.1 回填 ---
    rates: dict[str, Any] = {}
    for schema in record.get("npz_schemas", []):
        for key, value in schema.get("rate_like_keys", {}).items():
            rates[f"npz:{Path(schema['file']).name}:{key}"] = value
    for schema in record.get("mat_schemas", []):
        for key, info in list(schema.get("fields", {}).items()):
            if key.lower() in {name.lower() for name in RATE_KEYS} and info.get("kind") in {
                "int",
                "float",
                "float64",
                "float32",
                "int64",
                "int32",
            }:
                rates[f"mat:{Path(schema['file']).name}:{key}"] = info.get("value")
    if rates:
        record["rate_candidates"] = rates

    if notes:
        record["notes"] = notes
    if errors:
        record["errors"] = errors
    record["status"] = "ok"
    return record


def verify_pool_untouched(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """事后复核：重走一遍池，比对文件数 / 总字节 / 最大 mtime，证明探测没改动池。

    注：读取文件只更新 atime，不更新 mtime（且 Windows 默认关闭 last-access
    update），所以 mtime 指纹足以发现任何写入。
    """
    report: dict[str, Any] = {}
    for record in datasets:
        base = record.get("fingerprint")
        if not base:
            continue
        name = str(record["name"])
        root = Path(str(record["root"]))
        if not root.is_dir():
            report[name] = {"untouched": False, "reason": "root disappeared on re-check"}
            continue
        files = walk_files(root)
        stats = [path.stat() for path in files]
        current = {
            "file_count": len(files),
            "total_bytes": sum(stat.st_size for stat in stats),
            "max_mtime_ns": max((stat.st_mtime_ns for stat in stats), default=0),
        }
        report[name] = {"untouched": current == base, "before": base, "after": current}
    return report


def main() -> None:
    args = parse_args()
    paths.assert_readonly_pool(args.out, operation="write pool probe into")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    datasets = []
    for name in args.datasets:
        try:
            datasets.append(probe_dataset(name, args.schema_samples))
        except Exception:  # 单库失败不应中断整轮探测
            datasets.append(
                {
                    "name": name,
                    "status": "error",
                    "traceback": traceback.format_exc(limit=4),
                }
            )

    untouched = verify_pool_untouched(datasets)
    payload = {
        "schema_version": 1,
        "probe": "T0.3 pool accessibility and schema probe",
        "external_pool": str(paths.EXTERNAL_POOL),
        "registry": str(paths.POOL_REGISTRY),
        "output_path": str(args.out),
        "output_in_pool": paths.is_in_pool(args.out),
        "datasets_requested": list(args.datasets),
        "dataset_count": len(datasets),
        "datasets": datasets,
        "pool_untouched_report": untouched,
        "pool_untouched_all": all(item.get("untouched") is not False for item in untouched.values()),
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"{'dataset':16s} {'status':8s} {'files':>8s} {'size':>12s}  primary")
    for record in datasets:
        primary = record.get("primary_raw", {})
        detail = ""
        if primary:
            detail = (
                f"{primary['sfreq']:.1f} Hz, {primary['n_channels']} ch, "
                f"{primary['duration_s']:.0f} s [{Path(primary['file']).suffix}]"
            )
        elif record.get("npz_schemas"):
            keys = record["npz_schemas"][0].get("keys", [])
            detail = f"npz keys={keys[:6]}"
        elif record.get("mat_schemas"):
            detail = f"mat top={record['mat_schemas'][0].get('top_level', [])[:6]}"
        print(
            f"{record['name']:16s} {record.get('status', '?'):8s} "
            f"{record.get('file_count', 0):8d} {human_bytes(int(record.get('total_bytes', 0))):>12s}  {detail}"
        )


if __name__ == "__main__":
    main()
