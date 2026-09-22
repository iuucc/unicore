"""montage 与坐标表（手册 T1.1）。

职责：

- 提供 10-20 / 10-5 参考坐标表（从 `configs/montages/standard_reference.yaml` 读取，
  该文件由 `scripts/build_montages.py --emit-standard` 从 MNE 生成，数值可复核）。
- 把数据集的原始通道名解析成三维坐标，返回 ``(coords, mask)``。
- 双极导联（如 ``Fp1-F7``）取两端电极坐标的中点，并记录两端的标准名。
- 提供把三维坐标压到二维头部圆盘的正投影，供空间投影模块使用。

约定
----
坐标单位是**头部半径**（归一化到平均半径 1）。参考表用 MNE head RAS
（x=右、y=前、z=上）；各数据集自带的坐标系可能不同（见 ds004784），
本模块只保证**同一 montage 内部**的几何自洽，不承诺跨数据集朝向一致。

无法解析的通道：坐标置零、``mask=False``，**不抛异常**（手册 T1.1 验收第 2 条）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import yaml
from torch import Tensor

from . import paths

__all__ = [
    "MONTAGE_DIR",
    "REFERENCE_MONTAGE",
    "REFERENCE_19_NAMES",
    "MontageError",
    "Montage",
    "ChannelResolution",
    "canonical",
    "clean_label",
    "split_bipolar",
    "list_montages",
    "load_montage",
    "reference_table",
    "resolve_electrode",
    "resolve_montage",
    "resolve_channels",
    "classify_channel",
    "coverage_report",
    "project_to_disk",
    "describe_montage",
]

MONTAGE_DIR: Path = paths.CONFIG_ROOT / "montages"

#: 参考坐标表所在的 montage 名。
REFERENCE_MONTAGE = "standard_reference"

#: 标准 19 导。参考表的归一化尺度就定义在这组电极上（平均半径 = 1）。
REFERENCE_19_NAMES: tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz", "C3", "C4", "Cz",
    "T3", "T4", "T5", "T6", "P3", "P4", "Pz", "O1", "O2",
)

#: 非参考表、也不是数据集配置的文件名（不作为数据集 montage 暴露）。
_NON_DATASET_FILES = (REFERENCE_MONTAGE,)

#: 坐标半径低于该值的通道，位置信息量很低（典型来源：跨半球双极导联的中点落在近头心）。
LOW_INFORMATION_RADIUS = 0.6

#: 少量规范名之外的写法。键与值都是 ``canonical()`` 之后的形态。
#: 目前为空表：参考表（standard_1020 ∪ standard_1005）已覆盖全部实测通道名，
#: 保留这张表是为了让后续遇到新写法时有统一的落点，而不用散落 if 分支。
ALIASES: dict[str, str] = {}


class MontageError(ValueError):
    """montage 配置缺失或自相矛盾时抛出。"""


# --------------------------------------------------------------------------
# 名字规范化
# --------------------------------------------------------------------------
_PREFIX_RE = re.compile(r"^\s*(?:eeg|ecg|emg|eog|exg)\s*[:：_\-]\s*", re.IGNORECASE)
_PREFIX_SPACE_RE = re.compile(r"^\s*(?:eeg|ecg|emg|eog|exg)\s+", re.IGNORECASE)
_SUFFIX_RE = re.compile(r"[\s:：_\-]*r(?:ef|eference)\s*$", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")

#: 数据分布里出现的非电极名（分类用）：肌电/眼电/心电/呼吸等辅助通道。
AUX_HINT_RE = re.compile(
    r"^(?:roc|loc|emg|ecg|eog|resp|temp|event|marker|dx|sx|status|trigger|imu|acc)",
    re.IGNORECASE,
)


def clean_label(name: str) -> str:
    """去掉常见的前后缀（``EEG``/``EOG``/``-Ref``）与首尾点号，保留大小写与连字符。

    例：``"EEG Fp1-Ref"`` → ``"Fp1"``；``"EEG Fpz-Cz"`` → ``"Fpz-Cz"``；
    ``"C3.."`` → ``"C3"``；``"EEG:C3"`` → ``"C3"``；``"FCz_ref"`` → ``"FCz"``。
    """
    text = str(name).strip()
    text = _PREFIX_RE.sub("", text)
    text = _PREFIX_SPACE_RE.sub("", text)
    text = _SUFFIX_RE.sub("", text)
    return text.strip().strip(".").strip()


def canonical(name: str) -> str:
    """把名字压成用于查表的键：只保留字母数字并小写。

    例：``"Fp1"``→``"fp1"``、``"FCZ"``→``"fcz"``、``"Cz.."``→``"cz"``。
    """
    return _NON_ALNUM_RE.sub("", clean_label(name).lower())


def split_bipolar(name: str) -> tuple[str, str] | None:
    """判断是否为双极导联名，是则返回两端电极名（未规范化）。

    ``"Fp1-F7"``→``("Fp1", "F7")``；``"T8-P8-0"``→``("T8", "P8")``（剥掉末位序号）；
    ``"EEG Fpz-Cz"``→``("Fpz", "Cz")``；``"C3"``→``None``。
    """
    text = clean_label(name)
    if "-" not in text:
        return None
    parts = [segment.strip() for segment in text.split("-")]
    if len(parts) == 3 and parts[-1].isdigit():
        parts = parts[:2]
    if len(parts) != 2 or not all(parts):
        return None
    return parts[0], parts[1]


# --------------------------------------------------------------------------
# 参考坐标表
# --------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _reference_payload() -> dict[str, Any]:
    path = MONTAGE_DIR / f"{REFERENCE_MONTAGE}.yaml"
    if not path.is_file():
        raise MontageError(
            f"缺少参考坐标表 {path}；先运行 "
            "`python scripts/build_montages.py --emit-standard`"
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "coordinates" not in payload:
        raise MontageError(f"{path} 格式不对，缺少 coordinates 段")
    return payload


@lru_cache(maxsize=1)
def reference_table() -> dict[str, dict[str, Any]]:
    """返回 ``{规范键: {"name": 标准名, "coord": [x, y, z]}}``。"""
    table: dict[str, dict[str, Any]] = {}
    for name, coord in _reference_payload()["coordinates"].items():
        key = canonical(name)
        if key in table:
            raise MontageError(f"参考表里 {name} 与 {table[key]['name']} 规范化后重名（{key}）")
        table[key] = {"name": name, "coord": [float(v) for v in coord]}
    for key, target in ALIASES.items():
        if key not in table:
            raise MontageError(f"ALIASES 里的键 {key!r} 不在参考表中")
        if target not in table:
            raise MontageError(f"ALIASES 里的目标 {target!r} 不在参考表中")
        table[key] = table[target]
    return table


def resolve_electrode(name: str) -> tuple[str, Tensor] | None:
    """单个电极名 → ``(标准名, (3,) 坐标)``；无法解析返回 ``None``。"""
    entry = reference_table().get(canonical(name))
    if entry is None:
        return None
    return entry["name"], torch.tensor(entry["coord"], dtype=torch.float32)


# --------------------------------------------------------------------------
# montage 对象
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelResolution:
    """一个原始通道的解析结果。"""

    raw: str
    standard: str | None
    kind: str  # "eeg" | "bipolar" | "aux" | "unresolved"
    link: tuple[str, str] | None = None
    coord: Tensor | None = None

    @property
    def resolved(self) -> bool:
        return self.coord is not None


@dataclass(frozen=True)
class Montage:
    """一个数据集的通道布局与坐标表。"""

    name: str
    resolve: str
    bipolar: bool
    channels: tuple[str, ...]
    coords: Tensor  # (C, 3) float32
    mask: Tensor  # (C,) bool
    links: tuple[tuple[str, str] | None, ...]
    aux: tuple[dict[str, Any], ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    #: montage 自带的坐标表（如 ds004784 的 128 个体模电极）；没有则为 None。
    coordinates: dict[str, list[float]] | None = None
    #: ``positional`` 模式下 原始名(规范键) → 标准名 的映射。
    raw_map: dict[str, str] = field(default_factory=dict)

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def resolved_count(self) -> int:
        return int(self.mask.sum().item())

    def index_of(self, channel: str) -> int | None:
        """按原始名或标准名找通道序号（找不到返回 ``None``）。"""
        target = canonical(channel)
        for index, raw in enumerate(self.channels):
            if canonical(raw) == target:
                return index
            link = self.links[index]
            if link and target in {canonical(part) for part in link}:
                return index
        return None

    @property
    def frame_axes(self) -> dict[str, list[float]]:
        """坐标系的解剖方向基（单位向量）。

        参考表用 MNE head RAS，因此默认值就是 (0,0,1) / (0,1,0)；
        坐标系不同的数据集（如 ds004784）必须在配置里用 ``frame_axes`` 显式声明，
        否则依赖"前/上"的空间先验会算错方向。
        """
        declared = self.meta.get("frame_axes") or {}
        axes: dict[str, list[float]] = {
            "up": [0.0, 0.0, 1.0],
            "anterior": [0.0, 1.0, 0.0],
        }
        for key, value in declared.items():
            axes[str(key)] = [float(item) for item in value]
        if "right" not in axes:
            anterior = torch.tensor(axes["anterior"], dtype=torch.float32)
            up = torch.tensor(axes["up"], dtype=torch.float32)
            axes["right"] = torch.linalg.cross(anterior, up).tolist()
        return axes

    def to_head_ras(self, coords: Tensor | None = None) -> Tensor:
        """把 montage 坐标转成统一的 (right, anterior, superior) 轴顺序。

        返回新张量，不改写配置里保留的原始坐标。模型跨 montage 共用权重时，
        必须让“前额方向”在每个数据集里落到同一维度。
        """
        source = self.coords if coords is None else torch.as_tensor(coords, dtype=torch.float32)
        axes = self.frame_axes
        basis = torch.tensor([axes["right"], axes["anterior"], axes["up"]], dtype=source.dtype, device=source.device)
        gram = basis @ basis.T
        if not torch.allclose(gram, torch.eye(3, dtype=gram.dtype, device=gram.device), atol=1e-3):
            raise MontageError(f"{self.name}: frame_axes 必须是正交单位向量，得到 {basis.tolist()}")
        return source @ basis.T


def list_montages() -> tuple[str, ...]:
    """``configs/montages/`` 下所有数据集 montage 名（不含参考表）。"""
    if not MONTAGE_DIR.is_dir():
        return ()
    names = []
    for path in sorted(MONTAGE_DIR.glob("*.yaml")):
        if path.stem in _NON_DATASET_FILES:
            continue
        names.append(path.stem)
    return tuple(names)


def _build_raw_map(payload: dict[str, Any]) -> dict[str, str]:
    """``positional`` 模式：原始通道名 → 标准名 的映射。

    由 ``raw_layout``（记录的原始通道顺序）与 ``channels``（同序的标准名）逐位配对得到。
    ``anchors`` 用于校验这条知识是否与记录一致——不一致说明数据集某个文件的通道顺序变了。
    """
    raw_layout = payload.get("raw_layout")
    channels = [str(item) for item in payload.get("channels", ())]
    if not raw_layout:
        return {}
    raw_layout = [str(item) for item in raw_layout]
    if len(raw_layout) < len(channels):
        raise MontageError(
            f"{payload.get('name')}: raw_layout 只有 {len(raw_layout)} 项，少于 {len(channels)} 个 EEG 通道"
        )
    mapping = {
        canonical(raw): standard for raw, standard in zip(raw_layout[: len(channels)], channels)
    }
    for index_text, expected in (payload.get("anchors") or {}).items():
        index = int(index_text)
        actual = raw_layout[index] if index < len(raw_layout) else None
        if actual != expected:
            raise MontageError(
                f"{payload.get('name')}: 锚点校验失败，位置 {index} 记录为 {actual!r}，"
                f"配置期望 {expected!r}。原始通道顺序可能已变，需要重新核对。"
            )
    return mapping


@lru_cache(maxsize=64)
def load_montage(name: str) -> Montage:
    """读取 montage 配置并把全部通道解析成坐标。结果缓存。"""
    path = MONTAGE_DIR / f"{name}.yaml"
    if not path.is_file():
        known = ", ".join(list_montages()) or "（空）"
        raise MontageError(f"没有 montage 配置 {path}；已有：{known}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MontageError(f"{path} 不是一个 YAML 映射")

    resolve_mode = str(payload.get("resolve", "names"))
    if resolve_mode not in {"names", "positional"}:
        raise MontageError(f"{name}: 不支持的 resolve 模式 {resolve_mode!r}")

    coordinates = payload.get("coordinates")
    channels = [str(item) for item in payload.get("channels", ())]
    if not channels:
        raise MontageError(f"{name}: channels 为空")
    raw_map = _build_raw_map(payload) if resolve_mode == "positional" else {}

    coords: list[Tensor] = []
    mask: list[bool] = []
    links: list[tuple[str, str] | None] = []
    standards: list[str | None] = []

    for raw in channels:
        resolution = classify_channel(raw, coordinates, raw_map)
        standards.append(resolution.standard)
        links.append(resolution.link)
        if resolution.coord is None:
            coords.append(torch.zeros(3, dtype=torch.float32))
            mask.append(False)
        else:
            coords.append(resolution.coord)
            mask.append(True)

    if resolve_mode == "positional":
        if not raw_map:
            raise MontageError(
                f"{name}: resolve=positional 但没有可用的 raw_layout，无法确定原始名与标准名的对应"
            )
        for raw, standard in raw_map.items():
            if resolve_electrode(standard) is None:
                raise MontageError(
                    f"{name}: raw_layout 里的 {raw!r} 映射到标准名 {standard!r}，但该名字不在参考表中"
                )

    expected = payload.get("channel_count")
    if expected is not None and int(expected) != len(channels):
        raise MontageError(
            f"{name}: channel_count 声明 {expected}，实际 channels 长度 {len(channels)}，配置自相矛盾"
        )

    meta = {key: value for key, value in payload.items() if key not in {"channels", "coordinates"}}
    return Montage(
        name=name,
        resolve=resolve_mode,
        bipolar=bool(payload.get("bipolar", False)),
        channels=tuple(channels),
        coords=torch.stack(coords),
        mask=torch.tensor(mask, dtype=torch.bool),
        links=tuple(links),
        aux=tuple(dict(item) for item in payload.get("aux", ())),
        meta=meta,
        coordinates=dict(coordinates) if coordinates else None,
        raw_map=dict(raw_map),
    )


def classify_channel(
    raw: str,
    coordinates: dict[str, Any] | None = None,
    raw_map: dict[str, str] | None = None,
) -> ChannelResolution:
    """判断一个原始通道名属于哪一类，并给出坐标。分类优先级：

    1. montage 自带的坐标表（如 ds004784 的 128 个体模电极）
    2. ``positional`` 映射（原始名 → 标准名）
    3. 双极导联（两端电极坐标的中点）
    4. 单个标准位电极
    5. 无法解析 → ``kind="unresolved"``、``coord=None``
    """
    raw_map = raw_map or {}
    # 1) montage 自带坐标表（如 ds004784 的 128 个体模电极）
    if coordinates:
        for key in (raw, clean_label(raw)):
            if key in coordinates:
                return ChannelResolution(
                    raw=raw,
                    standard=str(key),
                    kind="eeg",
                    coord=torch.tensor([float(v) for v in coordinates[key]], dtype=torch.float32),
                )
        lowered = {str(k).lower(): k for k in coordinates}
        key = lowered.get(clean_label(raw).lower())
        if key is not None:
            return ChannelResolution(
                raw=raw,
                standard=str(key),
                kind="eeg",
                coord=torch.tensor([float(v) for v in coordinates[key]], dtype=torch.float32),
            )

    # 2) positional 映射：原始名 → 标准名，再查参考表
    mapped = raw_map.get(canonical(raw))
    if mapped is not None:
        found = resolve_electrode(mapped)
        if found is not None:
            standard, coord = found
            return ChannelResolution(raw=raw, standard=standard, kind="eeg", coord=coord)

    # 3) 双极导联：两端电极的中点
    pair = split_bipolar(raw)
    if pair is not None:
        first, second = (resolve_electrode(part) for part in pair)
        if first is not None and second is not None:
            return ChannelResolution(
                raw=raw,
                standard=f"{first[0]}-{second[0]}",
                kind="bipolar",
                link=(first[0], second[0]),
                coord=(first[1] + second[1]) / 2.0,
            )

    # 4) 单个标准位电极
    found = resolve_electrode(raw)
    if found is not None:
        standard, coord = found
        kind = "aux" if AUX_HINT_RE.match(clean_label(raw)) else "eeg"
        return ChannelResolution(raw=raw, standard=standard, kind=kind, coord=coord)

    return ChannelResolution(raw=raw, standard=None, kind="unresolved", coord=None)


# --------------------------------------------------------------------------
# 对外解析接口
# --------------------------------------------------------------------------
def resolve_channels(
    channel_names: Sequence[str],
    montage: str | Montage | None = None,
) -> tuple[list[ChannelResolution], Tensor, Tensor]:
    """解析任意通道名列表，返回 ``(逐通道明细, coords (C,3), mask (C,))``。

    ``montage=None`` 表示只按参考坐标表解析（不与任何数据集的通道布局绑定）。
    """
    coordinates, raw_map = _resolution_context(montage)
    details: list[ChannelResolution] = []
    coords: list[Tensor] = []
    mask: list[bool] = []
    for raw in channel_names:
        item = classify_channel(str(raw), coordinates, raw_map)
        details.append(item)
        if item.coord is None:
            coords.append(torch.zeros(3, dtype=torch.float32))
            mask.append(False)
        else:
            coords.append(item.coord)
            mask.append(True)
    return details, torch.stack(coords), torch.tensor(mask, dtype=torch.bool)


def _resolution_context(montage: str | Montage | None) -> tuple[dict[str, Any] | None, dict[str, str]]:
    """返回值决定解析时能用到哪些额外信息（自带坐标表 / positional 映射）。"""
    if montage is None:
        return None, {}
    spec = _as_montage(montage)
    return spec.coordinates, spec.raw_map


def resolve_montage(
    channel_names: Sequence[str],
    montage: str | Montage | None = None,
) -> tuple[Tensor, Tensor]:
    """按手册 T1.1 的签名：``resolve_montage(channel_names) -> (coords, mask)``。

    ``coords`` 形状 ``(C, 3)``，无法解析的通道坐标为零、``mask`` 为 False。
    ``montage=None`` 时只按参考坐标表解析，不绑定任何数据集的通道布局。
    """
    _, coords, mask = resolve_channels(channel_names, montage)
    return coords, mask


def _as_montage(montage: str | Montage | None) -> Montage:
    if isinstance(montage, Montage):
        return montage
    return load_montage(montage or REFERENCE_MONTAGE)


def coverage_report(montage: str | Montage) -> dict[str, Any]:
    """统计一个 montage 自身通道的解析覆盖率，用于验收与报告。"""
    spec = _as_montage(montage)
    details = [classify_channel(raw, spec.coordinates, spec.raw_map) for raw in spec.channels]
    unresolved = [item.raw for item in details if not item.resolved]
    kinds: dict[str, int] = {}
    for item in details:
        kinds[item.kind] = kinds.get(item.kind, 0) + 1
    bipolar_without_link = [
        item.raw for item in details if item.kind == "bipolar" and item.link is None
    ]
    radii = spec.coords[spec.mask].norm(dim=-1) if spec.mask.any() else None
    radii_all = spec.coords.norm(dim=-1)
    low_information = [
        (channel, round(float(radius), 4))
        for channel, radius, ok in zip(spec.channels, radii_all, spec.mask.tolist())
        if ok and float(radius) < LOW_INFORMATION_RADIUS
    ]
    return {
        "montage": spec.name,
        "resolve": spec.resolve,
        "bipolar": spec.bipolar,
        "channel_count": spec.channel_count,
        "resolved": spec.channel_count - len(unresolved),
        "unresolved": unresolved,
        "kinds": kinds,
        "bipolar_without_link": bipolar_without_link,
        "aux": [item.get("raw") for item in spec.aux],
        "coord_radius_min": None if radii is None else float(radii.min()),
        "coord_radius_max": None if radii is None else float(radii.max()),
        # 跨半球双极导联（如 C4-A1、FT9-FT10）的中点会落在近头心，位置信息量很低。
        # 这里如实列出来，供下游决定是否降权，而不是悄悄塞进一个"看起来正常"的坐标。
        "low_information": low_information,
    }


# --------------------------------------------------------------------------
# 二维投影
# --------------------------------------------------------------------------
def project_to_disk(
    coords: Tensor,
    mask: Tensor | None = None,
    *,
    mode: str = "azimuthal",
) -> Tensor:
    """把三维坐标投到头部圆盘，范围约 ``[-1, 1]``。

    ``azimuthal``：以 +z 为极轴的等距方位投影，``ρ = 极角``、``φ = atan2(x, y)``
    （φ 从"前"方向起算、朝"右"为正），最后按有效通道的最大 ρ 缩放到单位圆。

    只保证**同一 montage 内部**的一致性：若某个数据集自带的坐标系朝向不同
    （如 ds004784 的 x=前后），投影结果只是该 montage 自己的一种稳定二维编码。
    """
    if mode != "azimuthal":
        raise MontageError(f"暂不支持的投影模式 {mode!r}")
    if mask is None:
        mask = torch.ones(coords.size(0), dtype=torch.bool)
    valid = coords[mask]
    if valid.numel() == 0:
        return torch.zeros(coords.size(0), 2, dtype=coords.dtype)

    radius = valid.norm(dim=-1).clamp_min(1e-6)
    polar = torch.acos((valid[:, 2] / radius).clamp(-1.0, 1.0))
    azimuth = torch.atan2(valid[:, 0], valid[:, 1])
    projected = torch.stack((polar * torch.sin(azimuth), polar * torch.cos(azimuth)), dim=-1)
    scale = polar.max().clamp_min(1e-6)
    projected = projected / scale

    out = torch.zeros(coords.size(0), 2, dtype=coords.dtype)
    out[mask] = projected
    return out


def describe_montage(name: str) -> str:
    """一句人类可读的摘要，用于日志与报告。"""
    spec = load_montage(name)
    report = coverage_report(spec)
    kind_text = ", ".join(f"{kind}={count}" for kind, count in sorted(report["kinds"].items()))
    return (
        f"{spec.name}: {spec.channel_count} 通道（{kind_text}），"
        f"解决 {report['resolved']}/{spec.channel_count}，"
        f"bipolar={spec.bipolar}，resolve={spec.resolve}"
    )
