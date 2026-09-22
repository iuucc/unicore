"""路径单一真相源（手册 T0.1）。

本模块是**全项目唯一允许出现外部数据池绝对路径的文件**（手册 §2.4 规则 3、§5.4）。
其它任何源码、配置、脚本都不得写死 ``D:\\codexwork\\eeg\\...``，必须经由本模块。

三个根目录（手册 §2.1）：

===========  ============================  ==========  ==========
代号         绝对路径                      角色        允许操作
===========  ============================  ==========  ==========
CODE         ``D:\\codexwork\\unicore``    代码与产物  读写
DATA         ``<CODE>\\data``             数据与登记  读写（``raw/`` 只读）
POOL         ``D:\\codexwork\\eeg\\data``  外部数据池  **只读**
===========  ============================  ==========  ==========

关于 POOL 路径的两种含义，本模块显式区分、避免歧义：

``POOL[name]``
    **数据集在 POOL 下的根目录**，等于 ``EXTERNAL_POOL / name``。
    手册 §3.1 表格声明的"相对路径"全部以它为基准拼接。
``POOL_DECLARED[name]``
    ``data/local_sources.json`` 里登记的路径。对多数数据集等于根目录；
    对 ``faced`` 登记的是**实际使用的受试者子目录** ``faced/nm000112``，
    因此**不能**用登记路径去拼 ``faced/feature_cache`` 之类的同级目录。
    当登记路径落在默认根目录之外时，视为真·覆盖，``POOL[name]`` 采用登记值，
    并在 ``PoolEntry.overridden`` 上标记。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "RAW_ROOT",
    "RUNS_ROOT",
    "CONFIG_ROOT",
    "EXTERNAL_POOL",
    "POOL_REGISTRY",
    "POOL_DATASET_NAMES",
    "POOL",
    "POOL_DECLARED",
    "POOL_ROLES",
    "PoolEntry",
    "pool_entry",
    "pool_path",
    "pool_roles",
    "pool_declared",
    "pool_exists",
    "is_in_pool",
    "assert_readonly_pool",
    "pool_open",
    "PATH_SYMBOLS",
    "resolve_path",
    "ensure_project_dirs",
    "describe_pool",
]

# --------------------------------------------------------------------------
# CODE / DATA 根目录
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\codexwork\unicore")
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
RUNS_ROOT = PROJECT_ROOT / "runs"
CONFIG_ROOT = PROJECT_ROOT / "configs"

#: 全项目唯一一处外部数据池绝对路径（手册 T0.1 步骤 4）。
EXTERNAL_POOL = Path(r"D:\codexwork\eeg\data")

#: POOL 数据集登记表，由 ``data/local_sources.json`` 提供用途（role）声明。
POOL_REGISTRY = DATA_ROOT / "local_sources.json"

#: 手册 §3.1 与 T0.3 覆盖率要求涉及的 12 个数据集，顺序与手册一致。
POOL_DATASET_NAMES: tuple[str, ...] = (
    "physiomotion",
    "bci2a",
    "openbmi",
    "physionet_mi",
    "chbmit",
    "sleep_edfx",
    "cap_sleep",
    "ds002094",
    "faced",
    "cho_gigadb",
    "bci2b",
    "tuar",
)


# --------------------------------------------------------------------------
# POOL 解析
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolEntry:
    """单个 POOL 数据集的解析结果。

    ``root``      数据集在 POOL 下的根目录；§3.1 的相对路径以此为基准。
    ``declared``  登记表里的原始路径（未做存在性校验的规范化结果）。
    ``roles``     登记表声明的用途；tuar 这类不可用数据集为 ``("declared_unavailable",)``。
    ``overridden````declared`` 是否落在默认根目录之外（真·覆盖）。
    """

    name: str
    root: Path
    declared: Path
    roles: tuple[str, ...]
    overridden: bool

    @property
    def exists(self) -> bool:
        return self.root.is_dir()

    def path(self, *parts: str) -> Path:
        """在数据集根目录下拼接子路径。"""
        return self.root.joinpath(*parts) if parts else self.root


def _load_registry(path: Path) -> Mapping[str, Any]:
    """读取登记表；缺失或损坏时返回空表（不阻断 import）。"""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_declared(name: str, record: Any) -> tuple[Path, tuple[str, ...]]:
    """从登记记录解析出 (路径, 用途)。登记路径允许绝对或相对两种写法。"""
    default_root = EXTERNAL_POOL / name
    if isinstance(record, Mapping):
        raw_path = record.get("path")
        raw_roles = record.get("role") or record.get("roles") or ()
    elif isinstance(record, str):
        raw_path, raw_roles = record, ()
    else:
        raw_path, raw_roles = None, ()
    roles = tuple(str(role) for role in raw_roles) if isinstance(raw_roles, Iterable) else ()
    if not raw_path:
        return default_root, roles
    declared = Path(str(raw_path))
    if not declared.is_absolute():
        declared = default_root / declared
    return declared, roles


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _build_pool() -> tuple[
    dict[str, Path],
    dict[str, Path],
    dict[str, tuple[str, ...]],
    dict[str, PoolEntry],
]:
    registry = _load_registry(POOL_REGISTRY)
    roots: dict[str, Path] = {}
    declared_map: dict[str, Path] = {}
    roles_map: dict[str, tuple[str, ...]] = {}
    entries: dict[str, PoolEntry] = {}
    for name in POOL_DATASET_NAMES:
        default_root = EXTERNAL_POOL / name
        declared, roles = _resolve_declared(name, registry.get(name))
        # 只有登记路径落在默认根目录之外时才算覆盖，避免 faced/nm000112 这类
        # subject 子目录把整个数据集根目录顶掉。
        overridden = declared != default_root and not _is_under(declared, default_root)
        root = declared if overridden else default_root
        roots[name] = root
        declared_map[name] = declared
        roles_map[name] = roles
        entries[name] = PoolEntry(
            name=name,
            root=root,
            declared=declared,
            roles=roles,
            overridden=overridden,
        )
    return roots, declared_map, roles_map, entries


POOL: dict[str, Path]
POOL_DECLARED: dict[str, Path]
POOL_ROLES: dict[str, tuple[str, ...]]
POOL_ENTRIES: dict[str, PoolEntry]
POOL, POOL_DECLARED, POOL_ROLES, POOL_ENTRIES = _build_pool()


def pool_entry(name: str) -> PoolEntry:
    """取得数据集的完整解析结果。名字未登记时抛 ``KeyError``，不静默兜底。"""
    try:
        return POOL_ENTRIES[name]
    except KeyError as error:  # pragma: no cover - 防御性分支
        known = ", ".join(POOL_DATASET_NAMES)
        raise KeyError(f"unknown POOL dataset {name!r}; known: {known}") from error


def pool_path(name: str, *parts: str) -> Path:
    """数据集根目录下的子路径，例如 ``pool_path("bci2a", "A01T.gdf")``。"""
    return pool_entry(name).path(*parts)


def pool_roles(name: str) -> tuple[str, ...]:
    return pool_entry(name).roles


def pool_declared(name: str) -> Path:
    return pool_entry(name).declared


def pool_exists(name: str) -> bool:
    return pool_entry(name).exists


def describe_pool() -> list[dict[str, Any]]:
    """供探测脚本与报告使用的可序列化清单。"""
    return [
        {
            "name": entry.name,
            "root": str(entry.root),
            "declared": str(entry.declared),
            "roles": list(entry.roles),
            "overridden": entry.overridden,
            "exists": entry.exists,
        }
        for entry in POOL_ENTRIES.values()
    ]


# --------------------------------------------------------------------------
# POOL 只读保护（手册 T0.1 步骤 3）
# --------------------------------------------------------------------------


def is_in_pool(path: str | Path) -> bool:
    """判断路径是否落在外部数据池内（含池根自身）。"""
    try:
        candidate = Path(path).resolve()
    except OSError:  # pragma: no cover - 仅在极端路径下触发
        return False
    return _is_under(candidate, EXTERNAL_POOL.resolve())


def assert_readonly_pool(path: str | Path, operation: str = "write") -> None:
    """写操作前的守卫：目标落在 POOL 内则直接抛错（手册 §2.4 规则 1）。"""
    if is_in_pool(path):
        raise PermissionError(
            f"refusing to {operation} inside the read-only external pool: {path}"
        )


def pool_open(path: str | Path, mode: str = "r", **kwargs: Any):
    """打开 POOL 文件的安全包装；任何写模式都被拒绝。"""
    target = Path(path)
    assert_readonly_pool(target, operation=f"open(mode={mode!r})")
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        raise PermissionError(f"refusing to open pool file for writing: {target}")
    return target.open(mode, **kwargs)


# --------------------------------------------------------------------------
# 配置里的符号引用（configs/*.yaml）
# --------------------------------------------------------------------------


#: ``@`` 引用可用的符号名。POOL 相关符号只在运行时解析，配置里不出现字面路径。
PATH_SYMBOLS: dict[str, Path] = {
    "paths.PROJECT_ROOT": PROJECT_ROOT,
    "paths.DATA_ROOT": DATA_ROOT,
    "paths.RAW_ROOT": RAW_ROOT,
    "paths.RUNS_ROOT": RUNS_ROOT,
    "paths.CONFIG_ROOT": CONFIG_ROOT,
    "paths.EXTERNAL_POOL": EXTERNAL_POOL,
    "paths.POOL_REGISTRY": POOL_REGISTRY,
}


def resolve_path(value: str | Path) -> Path:
    """把配置值解析成绝对路径。

    约定：

    - 以 ``@`` 开头 → 符号引用。``@`` 之后到第一个 ``/`` 为符号名，余下部分按 POSIX
      风格拼接到该符号指向的路径后，例如 ``"@paths.RUNS_ROOT/stage_b/seed42"``。
    - 其它相对路径 → 相对 ``PROJECT_ROOT`` 解释，而不是相对当前工作目录。
      这样配置在任何 cwd 下都指向同一处。
    - 绝对路径 → 原样返回。**不要把 POOL 的绝对路径写进配置**（手册 §2.4 规则 3），
      需要池内文件时用 ``"@paths.EXTERNAL_POOL/..."`` 或 ``pool_path()``。
    """
    if isinstance(value, Path):
        return value
    text = str(value)
    if text.startswith("@"):
        head, _, tail = text[1:].partition("/")
        try:
            base = PATH_SYMBOLS[head]
        except KeyError as error:
            known = ", ".join(sorted(PATH_SYMBOLS))
            raise KeyError(f"unknown path symbol {head!r}; known: {known}") from error
        segments = [segment for segment in tail.replace("\\", "/").split("/") if segment]
        return base.joinpath(*segments) if segments else base
    candidate = Path(text)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


# --------------------------------------------------------------------------
# 目录骨架
# --------------------------------------------------------------------------


def ensure_project_dirs() -> dict[str, Path]:
    """建立 CODE 侧的目录骨架（不触碰 POOL）。"""
    paths = {
        "configs": CONFIG_ROOT,
        "montages": CONFIG_ROOT / "montages",
        "runs": RUNS_ROOT,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
