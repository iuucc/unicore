"""生成 `configs/montages/*.yaml`（手册 T1.1）。

三类产物：

1. `standard_reference.yaml`　10-20/10-5 参考坐标表。
   来源：MNE `standard_1020`（94 点）∪ `standard_1005`（343 点，只补 1020 缺的名字，
   实测需要补的是 `FTT9h TTP7h TPP9h FTT10h TPP8h TPP10h`）。
   归一化：全部坐标除以 19 导标准 10-20 电极的平均模长，使平均半径为 1。

2. `<dataset>.yaml`　各外部池数据集的通道布局。
   通道名**全部来自实测来源**（T0.3 的 `runs/_pool_probe/pool_probe.json`、
   数据集自带的 `.mat` / `.npz`、POOL 旁的旧项目常量），不手抄、不猜测。

3. `ds004784.yaml`　128 导体模坐标，来自 `sub-001_task-*_electrodes.tsv`（只读）。

纪律：只写 `configs/montages/`，绝不写入外部池（手册 §0.2）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unicore_eeg import paths  # noqa: E402
from unicore_eeg.montage import classify_channel  # noqa: E402

MONTAGE_DIR = PROJECT_ROOT / "configs" / "montages"
PROBE_JSON = PROJECT_ROOT / "runs" / "_pool_probe" / "pool_probe.json"

#: 归一到"平均半径 1"时使用的参考电极集合（标准 19 导）。
REFERENCE_19 = (
    "Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz", "C3", "C4", "Cz",
    "T3", "T4", "T5", "T6", "P3", "P4", "Pz", "O1", "O2",
)

#: 需要从 10-5 系统补进参考表的额外名字（仅这些在 standard_1020 中缺失）。
EXTRA_FROM_1005 = ("FTT9h", "TTP7h", "TPP9h", "FTT10h", "TPP8h", "TPP10h")

#: 旧项目常量文件（只读参考，手册 §4）。仅用于取已验证的通道名清单，运行时不依赖它。
LEGACY_CONSTANTS = Path("D:/codexwork/eeg/src/metacorrseg/data/constants.py")
LEGACY_CHO = Path("D:/codexwork/eeg/src/metacorrseg/data/cho.py")


# --------------------------------------------------------------------------
# 参考坐标表
# --------------------------------------------------------------------------
def load_standard_1005_names() -> dict[str, list[float]]:
    """返回 {名字: [x, y, z]}（米，MNE head RAS：x=右、y=前、z=上）。"""
    import mne

    coordinates: dict[str, list[float]] = {}
    m1020 = mne.channels.make_standard_montage("standard_1020")
    for name, xyz in m1020.get_positions()["ch_pos"].items():
        coordinates[name] = [float(v) for v in xyz]

    m1005 = mne.channels.make_standard_montage("standard_1005")
    pos1005 = m1005.get_positions()["ch_pos"]
    missing = [name for name in EXTRA_FROM_1005 if name not in pos1005]
    if missing:
        raise SystemExit(f"standard_1005 缺少预期存在的名字：{missing}")
    for name in EXTRA_FROM_1005:
        coordinates[name] = [float(v) for v in pos1005[name]]

    return coordinates


def build_standard_reference() -> dict[str, Any]:
    coordinates = load_standard_1005_names()
    array = np.array([coordinates[name] for name in sorted(coordinates)], dtype=np.float64)
    scales = {name: float(np.linalg.norm(coordinates[name])) for name in REFERENCE_19}
    scale = float(np.mean(list(scales.values())))

    normalized = {
        name: [round(float(v) / scale, 8) for v in coordinates[name]] for name in sorted(coordinates)
    }
    radii = np.linalg.norm(np.array(list(normalized.values())), axis=1)
    return {
        "name": "standard_reference",
        "kind": "reference",
        "description": (
            "10-20 / 10-5 参考坐标表。standard_1020 提供全部 10-20 与常见扩展导联；"
            "standard_1005 只补 standard_1020 缺失的 6 个 10-5 名字"
            "（FTT9h TTP7h TPP9h FTT10h TPP8h TPP10h，openbmi 的 62 导用到）。"
        ),
        "generated_by": "scripts/build_montages.py --emit-standard",
        "source": {
            "primary": "mne.channels.make_standard_montage('standard_1020')",
            "extension": "mne.channels.make_standard_montage('standard_1005')",
            "mne_version": _mne_version(),
        },
        "coord_frame": "MNE head RAS：x=右，y=前，z=上",
        "frame_axes": {"up": [0.0, 0.0, 1.0], "anterior": [0.0, 1.0, 0.0]},
        "unit": "head radius（平均半径 = 1）",
        "normalization": (
            "除以标准 19 导的平均模长。平均半径精确为 1；因为 MNE 头模型不是正球，"
            f"单点半径范围为 {radii.min():.3f}–{radii.max():.3f}（保留真实头形差异）。"
        ),
        "reference_19": list(REFERENCE_19),
        "coordinate_count": len(normalized),
        # 参考表同时是一个合法的 montage（100 个位置全部自带坐标），
        # 这样 load_montage("standard_reference") 也能用，便于读取 frame_axes 与做覆盖率自查。
        "channel_count": len(normalized),
        "channels": sorted(normalized),
        "coordinates": normalized,
    }


def _mne_version() -> str:
    try:
        import mne

        return str(mne.__version__)
    except Exception:  # pragma: no cover - 仅在缺 mne 时触发
        return "unknown"


# --------------------------------------------------------------------------
# 数据集通道名来源
# --------------------------------------------------------------------------
def probe_entry(dataset: str) -> dict[str, Any]:
    payload = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    for entry in payload["datasets"]:
        if entry["name"] == dataset:
            return entry
    raise SystemExit(f"{PROBE_JSON} 中没有 {dataset}（先跑 scripts/probe_pool.py）")


def probe_raw_channels(dataset: str) -> list[str]:
    entry = probe_entry(dataset)
    primary = entry.get("primary_raw")
    if not primary:
        raise SystemExit(f"{dataset} 的探测结果缺少 primary_raw")
    return list(primary["ch_names"])


def probe_npz_channels(dataset: str) -> list[str]:
    entry = probe_entry(dataset)
    for schema in entry.get("npz_schemas", []):
        values = schema.get("schema", {}).get("channels", {}).get("values")
        if values:
            return list(values)
    raise SystemExit(f"{dataset} 的探测结果没有 npz 通道名")


def legacy_constant_tuple(path: Path, name: str) -> list[str]:
    """从只读参考文件里取出一个字符串常量的列表（不 import 那个包）。"""
    text = path.read_text(encoding="utf-8")
    if name not in text:
        raise SystemExit(f"{path} 中没有常量 {name}")
    block = text[text.index(name) :]
    start = block.index("= (") + 3
    end = block.index(")", start)
    items = []
    for line in block[start:end].strip().splitlines():
        item = line.strip().rstrip(",").strip().strip('"').strip("'")
        if item:
            items.append(item)
    return items


def openbmi_raw_channels() -> list[str]:
    """openbmi 原始 .mat 的 `chan` 字段（62 导）；只读，不复制数据。"""
    from scipy.io import loadmat

    match = sorted(paths.pool_path("openbmi").rglob("sess01_subj01_EEG_MI.mat"))
    if not match:
        raise SystemExit("找不到 openbmi 的 sess01_subj01_EEG_MI.mat")
    loaded = loadmat(str(match[0]), variable_names=["EEG_MI_train"])
    struct = loaded["EEG_MI_train"][0, 0]
    flat = np.asarray(struct["chan"]).ravel()
    names = []
    for item in flat:
        value = item[0] if isinstance(item, np.ndarray) else item
        names.append(str(value))
    return names


# --------------------------------------------------------------------------
# 配置构造
# --------------------------------------------------------------------------
def montage_payload(
    name: str,
    *,
    channels: Sequence[str],
    bipolar: bool,
    resolve: str,
    description: str,
    source: str,
    sample_rate: float | None = None,
    available_layouts: Sequence[str] = (),
    aux: Sequence[dict[str, str]] = (),
    anchors: dict[int, str] | None = None,
    raw_layout: Sequence[str] = (),
    unresolved: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "description": description,
        "resolve": resolve,
        "bipolar": bipolar,
        "channel_count": len(channels),
        "source": source,
        "channels": list(channels),
    }
    if sample_rate is not None:
        payload["sample_rate"] = sample_rate
    if available_layouts:
        payload["available_layouts"] = list(available_layouts)
    if raw_layout:
        payload["raw_layout"] = list(raw_layout)
    if anchors:
        payload["anchors"] = {str(index): raw for index, raw in sorted(anchors.items())}
    if aux:
        payload["aux"] = [dict(item) for item in aux]
    if unresolved:
        payload["unresolved"] = list(unresolved)
    if notes:
        payload["notes"] = list(notes)
    return payload


def dataset_payloads() -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}

    # --- bci2a：EDF 通道名是匿名的 EEG-0..EEG-16，只能按位置对应（锚点做完整性校验） ---
    bci2a_raw = probe_raw_channels("bci2a")
    payloads["bci2a"] = montage_payload(
        "bci2a",
        channels=legacy_constant_tuple(LEGACY_CONSTANTS, "BCI2A_EEG_CHANNELS"),
        bipolar=False,
        resolve="positional",
        description="BCI Competition IV 2a，22 导国际 10-20 系统（其余 3 路为 EOG）",
        source=(
            "POOL 旁旧项目 constants.py::BCI2A_EEG_CHANNELS（顺序与原始 EDF 一致，已用锚点校验）"
        ),
        sample_rate=250.0,
        raw_layout=bci2a_raw,
        aux=(
            {"raw": "EOG-left", "kind": "eog"},
            {"raw": "EOG-central", "kind": "eog"},
            {"raw": "EOG-right", "kind": "eog"},
        ),
        anchors={0: "EEG-Fz", 7: "EEG-C3", 9: "EEG-Cz", 11: "EEG-C4", 19: "EEG-Pz"},
        notes=(
            "原始 EDF 只有 4 个带名通道（EEG-Fz / EEG-C3 / EEG-Cz / EEG-C4 / EEG-Pz），"
            "其余是 EEG-0..EEG-16。五个锚点在实测通道表里的位置与 BCI2A_EEG_CHANNELS 完全一致，"
            f"因此按位置映射成立。实测通道表：{bci2a_raw}",
        ),
    )

    # --- bci2b：3 路 EEG（无第二电极，按标准位，不做中点） ---
    payloads["bci2b"] = montage_payload(
        "bci2b",
        channels=legacy_constant_tuple(LEGACY_CONSTANTS, "BCI2B_EEG_CHANNELS"),
        bipolar=False,
        resolve="names",
        description="BCI Competition IV 2b，3 导（C3/Cz/C4）",
        source="原始 EDF 实测通道名 EEG:C3 / EEG:Cz / EEG:C4（T0.3 探测）",
        sample_rate=250.0,
        aux=(
            {"raw": "EOG:ch01", "kind": "eog"},
            {"raw": "EOG:ch02", "kind": "eog"},
            {"raw": "EOG:ch03", "kind": "eog"},
        ),
        notes=(
            "手册 T1.1 写作\"3 双极\"，但原始通道是无第二电极的标准位名（EEG:C3/Cz/C4），"
            "旧项目 constants.py::BCI2B_EEG_CHANNELS 也把它们列为标准名，"
            "故按标准位取坐标、不做中点（bipolar: false）。见附录 C。",
        ),
    )

    # --- physiomotion：34 双极，照抄旧项目常量（与实测 EDF 通道名逐字一致） ---
    physiomotion_raw = probe_raw_channels("physiomotion")
    declared = legacy_constant_tuple(LEGACY_CONSTANTS, "PHYSIOMOTION_BIPOLAR_CHANNELS")
    if declared != physiomotion_raw:
        raise SystemExit(
            "PhysioMotion 的实测通道名与旧项目常量不一致，需要人工核对：\n"
            f"  实测 {physiomotion_raw}\n  常量 {declared}"
        )
    payloads["physiomotion"] = montage_payload(
        "physiomotion",
        channels=declared,
        bipolar=True,
        resolve="names",
        description="PhysioMotion，34 路双极导联（坐标取两端电极中点）",
        source="原始 EDF 实测通道名，与 POOL 旁旧项目 constants.py::PHYSIOMOTION_BIPOLAR_CHANNELS 逐字一致",
        sample_rate=1000.0,
        notes=("实测采样率 1000 Hz（非手册早期写的 500 Hz）。双极导联坐标 = 两端电极坐标中点。",),
    )

    # --- chbmit：23 双极（含 FT9/FT10 与两条 T8-P8-*，末位数字会被解析器剥掉） ---
    chbmit_raw = probe_raw_channels("chbmit")
    payloads["chbmit"] = montage_payload(
        "chbmit",
        channels=chbmit_raw,
        bipolar=True,
        resolve="names",
        description="CHB-MIT 头皮 EEG，23 路双极导联（坐标取两端电极中点）",
        source="原始 EDF 实测通道名（T0.3 探测）",
        sample_rate=256.0,
        notes=(
            "名称里的 T8-P8-0 / T8-P8-1 是同名导联的两个重名通道，解析时剥掉末位序号后"
            "都落到 T8–P8 的中点（两者坐标相同，符合数据实际）。"
            "T9/FT9/FT10/T10 均在参考表中，不需要近似。",
        ),
    )

    # --- cap_sleep：18 路里 13 路是 EEG 导联，其余 5 路是生理/辅助通道 ---
    cap_raw = probe_raw_channels("cap_sleep")
    cap_eeg, cap_aux, cap_unresolved = _split_by_resolvability(cap_raw)
    cache_channels = probe_npz_channels("cap_sleep")
    payloads["cap_sleep"] = montage_payload(
        "cap_sleep",
        channels=cap_eeg,
        bipolar=True,
        resolve="names",
        description="CAP Sleep Database，13 路双极 EEG 导联（原记录 18 路，其余为 ROC/EMG/ECG/DX/SX）",
        source="原始 EDF 实测通道名（T0.3 探测）",
        sample_rate=512.0,
        aux=tuple({"raw": name, "kind": "physiological"} for name in cap_aux),
        unresolved=cap_unresolved,
        available_layouts=(
            "cap_sleep 原始 EDF：18 路（本配置取其中 13 路 EEG 导联）",
            f"cap_sleep/feature_cache/*.npz：{len(cache_channels)} 路，全部是本配置通道的子集，"
            "按名解析即可，无需另建 montage",
            "整库不统一（实测 512Hz/18、256Hz/18、512Hz/22 三种），另两种布局的通道名尚未探测到，待阶段 2 遇到时补",
        ),
        notes=(
            "C4-A1 里的 A1 是乳突参考位，在参考表中有坐标。"
            f"feature_cache 实测通道：{cache_channels}",
        ),
    )

    # --- sleep_edfx：2 路双极 EEG ---
    sleep_raw = probe_raw_channels("sleep_edfx")
    sleep_eeg, sleep_aux, sleep_unresolved = _split_by_resolvability(sleep_raw)
    payloads["sleep_edfx"] = montage_payload(
        "sleep_edfx",
        channels=sleep_eeg,
        bipolar=True,
        resolve="names",
        description="Sleep-EDF，2 路双极 EEG（Fpz-Cz、Pz-Oz），其余为呼吸/肌电/体温/事件标记",
        source="原始 EDF 实测通道名（T0.3 探测）；两个名字的写法需去掉前缀 'EEG '",
        sample_rate=100.0,
        aux=tuple({"raw": name, "kind": "physiological"} for name in sleep_aux),
        unresolved=sleep_unresolved,
        notes=(
            "缓存 npz（cache20/*.npz）没有 channels 字段，无法核实缓存里的 2 个通道身份，"
            "阶段 2 加载时应断言其与 (Fpz-Cz, Pz-Oz) 的对应关系。",
        ),
    )

    # --- ds002094：30 路，标准位名（TP9/TP10/IZ 走别名） ---
    payloads["ds002094"] = montage_payload(
        "ds002094",
        channels=probe_raw_channels("ds002094"),
        bipolar=False,
        resolve="names",
        description="ds002094（TMS-EEG），30 路标准位 EEG",
        source="BrainVision .vhdr 实测通道名（T0.3 探测）",
        sample_rate=5000.0,
        notes=("TP9/TP10 已并入 T9/T10，IZ 走别名 IZ→Iz。实测采样率 5000 Hz，降采样需注意抗混叠。",),
    )

    # --- faced：32 路（含 A1/A2 乳突） ---
    payloads["faced"] = montage_payload(
        "faced",
        channels=probe_raw_channels("faced"),
        bipolar=False,
        resolve="names",
        description="FACED，32 路标准位 EEG（含 A1/A2 乳突参考位）",
        source="BDF 实测通道名（T0.3 探测）",
        sample_rate=250.0,
    )

    # --- openbmi：62 路原始；缓存 npz 的 20 路是其子集 ---
    openbmi_channels = openbmi_raw_channels()
    cache20 = probe_npz_channels("openbmi")
    payloads["openbmi"] = montage_payload(
        "openbmi",
        channels=openbmi_channels,
        bipolar=False,
        resolve="names",
        description="OpenBMI，62 路原始 EEG（g.USBamp 64 导布局去掉 2 个参考通道）",
        source="原始 .mat 的 struct 字段 `chan`（62 项，T0.3 补充探测）",
        available_layouts=(
            f"原始 .mat：{len(openbmi_channels)} 路",
            f"cache/subj*_session*_mi.npz：{len(cache20)} 路，是本配置通道的子集，按名解析即可",
        ),
        notes=(
            "FTT9h/TTP7h/TPP9h/FTT10h/TPP8h/TPP10h 六个名字不在 standard_1020，"
            "由 standard_1005 提供精确坐标。"
            f"缓存通道：{cache20}",
        ),
    )

    # --- physionet_mi：原始 64 路；缓存 21 路是其子集（且是前 21 个） ---
    physionet_raw = probe_raw_channels("physionet_mi")
    cache21 = probe_npz_channels("physionet_mi")
    payloads["physionet_mi"] = montage_payload(
        "physionet_mi",
        channels=physionet_raw,
        bipolar=False,
        resolve="names",
        description="PhysioNet MI（EEGMMIDB），原始 64 路标准 10-10 布局",
        source="EDF 实测通道名（T0.3 探测，形如 'Fc5.'/'C3..'，解析时去掉点号）",
        sample_rate=160.0,
        available_layouts=(
            f"原始 EDF：{len(physionet_raw)} 路",
            f"four_class_cache/*.npz：{len(cache21)} 路，等于原始 64 路的前 {len(cache21)} 个，按名解析即可",
        ),
        notes=(f"缓存通道：{cache21}",),
    )

    # --- cho_gigadb：64 路（62 导 + 2 个参考通道） ---
    payloads["cho_gigadb"] = montage_payload(
        "cho_gigadb",
        channels=legacy_constant_tuple(LEGACY_CHO, "CHO_EEG_CHANNELS"),
        bipolar=False,
        resolve="names",
        description="Cho2017 (GigaDB)，64 路，末两路 FCz_ref / Oz_ref 是参考通道",
        source="POOL 旁旧项目 cho.py::CHO_EEG_CHANNELS（64 名，与缓存 npz 的 21 路逐字一致）",
        sample_rate=512.0,
        notes=(
            "数据集自带 eeg.psenloc（64×3，单位球）与 eeg.senloc（cm）。已实测拒绝使用："
            "在已知通道对应下做 Kabsch 刚性对齐，最优平均弦长 0.90（完全随机约 1.27），"
            "说明其数组顺序与 CHO_EEG_CHANNELS 不对应或不是头部坐标系，故改用按名解析。"
            "见附录 C。",
        ),
    )

    return payloads


def _split_by_resolvability(
    names: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    """按"能否解析出坐标"把一个记录的通道列表拆成 (eeg, aux, unresolved)。

    用能否解析来判断，而不是"名字里有没有 '-'"——后者会把 ``ROC-LOC``、``ECG1-ECG2``
    这类辅助通道误判成 EEG 双极导联（这是第一版实现踩过的坑）。
    """
    eeg: list[str] = []
    aux: list[str] = []
    unresolved: list[str] = []
    for name in names:
        kind = classify_channel(name).kind
        if kind in {"eeg", "bipolar"}:
            eeg.append(name)
        elif kind == "aux":
            aux.append(name)
        else:
            unresolved.append(name)
    return eeg, aux, unresolved


# --------------------------------------------------------------------------
# 覆盖率报告（生成后的自查）
# --------------------------------------------------------------------------
def report_coverage() -> int:
    """逐个 montage 报告解析覆盖率；有任何未解析通道时返回非零。"""
    from unicore_eeg.montage import MONTAGE_DIR as MONTAGE_DIR_RUNTIME
    from unicore_eeg.montage import MontageError, coverage_report, describe_montage, list_montages

    names = list_montages()
    if not names:
        print(f"没有找到任何 montage 配置（{MONTAGE_DIR_RUNTIME}）")
        return 1
    print(f"montage 目录：{MONTAGE_DIR_RUNTIME}")
    print(f"{'montage':16s} {'resolve':11s} {'通道':>5s} {'解析':>5s} {'双极':>5s}  未解析通道")
    failures = 0
    for name in names:
        try:
            info = coverage_report(name)
        except MontageError as error:
            print(f"{name:16s} 配置错误：{error}")
            failures += 1
            continue
        bad = "、".join(info["unresolved"]) if info["unresolved"] else "—"
        bipolar = info["kinds"].get("bipolar", 0)
        print(
            f"{info['montage']:16s} {info['resolve']:11s} {info['channel_count']:5d} "
            f"{info['resolved']:5d} {bipolar:5d}  {bad}"
        )
        if info["unresolved"] or (info["bipolar"] and bipolar != info["channel_count"] and info["montage"] not in {"cap_sleep"}):
            failures += 1
    print()
    for name in names:
        print("  " + describe_montage(name))
    return 1 if failures else 0


# --------------------------------------------------------------------------
# ds004784
# --------------------------------------------------------------------------
def ds004784_payload() -> dict[str, Any]:
    import csv

    root = PROJECT_ROOT / "data" / "raw" / "ds004784"
    electrode_files = sorted(root.glob("sub-*/eeg/sub-*_task-*_electrodes.tsv"))
    if not electrode_files:
        raise SystemExit(f"找不到 ds004784 的 electrodes.tsv（在 {root}）")

    per_file: dict[str, list[tuple[str, list[float]]]] = {}
    for path in electrode_files:
        rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
        pairs = [
            (row["name"], [float(row["x"]), float(row["y"]), float(row["z"])])
            for row in rows
            if row.get("x") not in (None, "", "n/a", "NA")
        ]
        per_file[path.name] = pairs

    reference = per_file[electrode_files[0].name]
    for name, pairs in per_file.items():
        if pairs != reference:
            raise SystemExit(f"{name} 的电极坐标与 {electrode_files[0].name} 不一致，不能合成一份 montage")

    names = [name for name, _ in reference]
    raw = np.array([xyz for _, xyz in reference], dtype=np.float64)
    radii = np.linalg.norm(raw, axis=1)
    if float(radii.std()) > 1e-3:
        raise SystemExit("ds004784 电极不在等半径球面上，原先的归一化假设失效，需要重新核查")
    scale = float(radii.mean())

    coordinates = {
        name: [round(float(value) / scale, 8) for value in xyz] for name, xyz in reference
    }
    return {
        "name": "ds004784",
        "description": (
            "ds004784 导电球体模，128 路 EEG 电极。坐标来自数据集自带 electrodes.tsv。"
        ),
        "resolve": "names",
        "bipolar": False,
        "generated_by": "scripts/build_montages.py --emit-ds004784",
        "channel_count": len(names),
        "source": f"{electrode_files[0].relative_to(PROJECT_ROOT).as_posix()}（六个任务文件内容一致）",
        "coord_frame": (
            "数据集原始坐标系：x=前后（x<0 为枕/颈侧）、y=左右（严格镜像对称）、z=上"
            "（顶点 A1=(0,0,1)）；不是 MNE 的 RAS 朝向"
        ),
        # 空间先验需要知道"前"和"上"是哪个轴；这个数据集的坐标不是 RAS，必须显式声明。
        "frame_axes": {"up": [0.0, 0.0, 1.0], "anterior": [1.0, 0.0, 0.0]},
        "unit": "head radius（平均半径 = 1）",
        "normalization": (
            f"除以球面半径 {scale:.1f} mm。全部 {len(names)} 个电极半径恒为 "
            f"{radii.min():.1f}–{radii.max():.1f} mm（标准差 {radii.std():.2e}），"
            "归一化后精确落在单位球面上。"
        ),
        "channels": names,
        "coordinates": coordinates,
        "notes": (
            "通道名与 channels.tsv 完全一致（A1–A32 / B1–B32 / C1–C32 / D1–D32 各 32 个）。"
            "上一轮记录的\"A1..A128\"是错的，已在附录 C 更正。",
            "该数据集不生成标准 19 导子集：128 个电极是均匀铺满球冠的阵列而非 10-20 头皮帽，"
            "实测最优刚性对齐（ICP，8 种初始朝向）后标准 19 导到最近电极的平均距离仍有 "
            "0.1485 半径 ≈ 8.3 mm，且左右完全镜像对称导致面内朝向不可解。经用户确认放弃该文件。",
        ),
    }


# --------------------------------------------------------------------------
# 写出与查验
# --------------------------------------------------------------------------
def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100)
    path.write_text(text, encoding="utf-8")
    print(f"  written {path.relative_to(PROJECT_ROOT)}  ({len(text)} bytes)")


def emit_standard() -> None:
    print("[standard_reference]")
    write_yaml(MONTAGE_DIR / "standard_reference.yaml", build_standard_reference())


def emit_datasets() -> None:
    print("[datasets]")
    for name, payload in dataset_payloads().items():
        write_yaml(MONTAGE_DIR / f"{name}.yaml", payload)


def emit_ds004784() -> None:
    print("[ds004784]")
    write_yaml(MONTAGE_DIR / "ds004784.yaml", ds004784_payload())


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 configs/montages/*.yaml（手册 T1.1）")
    parser.add_argument("--emit-standard", action="store_true", help="生成 10-20/10-5 参考坐标表")
    parser.add_argument("--emit-datasets", action="store_true", help="生成各数据集通道布局")
    parser.add_argument("--emit-ds004784", action="store_true", help="生成 ds004784 的 128 导坐标")
    parser.add_argument("--all", action="store_true", help="等价于三个 --emit-* 全开")
    parser.add_argument("--report", action="store_true", help="逐 montage 报告解析覆盖率")
    args = parser.parse_args()

    if not any((args.emit_standard, args.emit_datasets, args.emit_ds004784, args.report, args.all)):
        parser.error("至少给一个 --emit-* / --report / --all")

    if args.all or args.emit_standard:
        emit_standard()
    if args.all or args.emit_datasets:
        emit_datasets()
    if args.all or args.emit_ds004784:
        emit_ds004784()
    if args.report:
        raise SystemExit(report_coverage())
    print("done")


if __name__ == "__main__":
    main()
