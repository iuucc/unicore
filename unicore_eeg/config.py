"""实验配置加载器（手册 §5.4 配置化 + T0.4 配置基线）。

三条约定
--------

1. **继承**：``configs/*.yaml`` 可以用 ``extends`` 声明父配置，值为字符串或列表。
   列表按顺序从左到右深合并，**后者覆盖前者**，最后合并本文件自己的键。
   这样 ``configs/multichannel_base.yaml`` 只需写差异，不必复制整份 ``base.yaml``。

2. **路径符号**：任何以 ``@paths.`` 开头的字符串都会被 ``unicore_eeg.paths.resolve_path``
   解析成 ``Path``。这是手册 §2.4 规则 3 的落地方式——配置文件里永远不出现
   外部数据池的绝对路径。

3. **命令行覆盖**：脚本用 ``argparse`` 的 ``default=argparse.SUPPRESS`` 声明来自配置的选项，
   未在命令行显式给出的项不会出现在命名空间里；``explicit_overrides`` 据此挑出
   真正需要覆盖配置的项。命令行优先级最高，配置文件次之，代码内默认值最低。

采用这个顺序是为了同时满足两件事：手册 §5.4「所有实验参数进 YAML」，以及
「T0.5 的历史复现命令保持原样可跑」。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml

from . import paths


__all__ = [
    "EXTENDS_KEY",
    "PATH_PREFIX",
    "load_config",
    "load_raw_yaml",
    "deep_merge",
    "get",
    "set_dotted",
    "apply_overrides",
    "explicit_overrides",
    "to_jsonable",
    "save_config",
]


EXTENDS_KEY = "extends"
PATH_PREFIX = "@paths."
DEFAULT_CONFIG_ROOT = paths.CONFIG_ROOT


def load_raw_yaml(path: str | Path) -> dict[str, Any]:
    """读单个 YAML 文件，不解析 extends。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"config root must be a mapping, got {type(payload).__name__}: {path}")
    return payload


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并两个映射；``override`` 中标量覆盖 ``base``，映射逐键递归。"""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_extends(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    raise TypeError(f"{EXTENDS_KEY} must be a string or a list of strings, got {type(value).__name__}")


def _resolve_symbols(value: Any) -> Any:
    """把 ``@paths.*`` 字符串解析为 Path；递归处理映射与列表。"""
    if isinstance(value, str):
        return paths.resolve_path(value) if value.startswith(PATH_PREFIX) else value
    if isinstance(value, Mapping):
        return {key: _resolve_symbols(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_symbols(item) for item in value]
    return value


def load_config(
    path: str | Path,
    config_root: str | Path | None = None,
    *,
    _stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """加载配置并解析 ``extends`` 链与 ``@paths.*`` 符号。

    相对路径的 ``extends`` 以 ``config_root``（默认 ``configs/``）为基准；
    也接受同目录内的相对写法。
    """
    target = Path(path)
    if not target.is_absolute():
        target = (Path(config_root) if config_root else DEFAULT_CONFIG_ROOT) / target
    target = target.resolve()
    if target in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, target))
        raise ValueError(f"circular config extends chain: {chain}")
    if not target.exists():
        raise FileNotFoundError(target)

    raw = load_raw_yaml(target)
    parents = _resolve_extends(raw.pop(EXTENDS_KEY, None))

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = target.parent / parent_path
        merged = deep_merge(merged, load_config(parent_path, config_root, _stack=(*_stack, target)))

    merged = deep_merge(merged, raw)
    merged["_config_path"] = str(target)
    return _resolve_symbols(merged)


def get(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """按点分路径取值，例如 ``get(config, "dataloader.batch_size")``。"""
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def set_dotted(config: MutableMapping[str, Any], dotted: str, value: Any) -> None:
    """按点分路径写入，中间层不存在时自动建字典。"""
    parts = dotted.split(".")
    current: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        node = current.get(part)
        if not isinstance(node, MutableMapping):
            node = {}
            current[part] = node
        current = node
    current[parts[-1]] = value


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """把 ``{"dataloader.batch_size": 64}`` 之类的覆盖应用到配置**副本**上。

    必须深拷贝：浅拷贝只会复制顶层字典，嵌套字典仍是同一对象，
    ``set_dotted`` 会就地改动它们，从而污染原配置（多个实验共用同一份 base 配置时
    会静默串味）。这是实测发现过的真实缺陷，已被 ConfigTests 覆盖。
    """
    result: dict[str, Any] = copy.deepcopy(dict(config))
    for dotted, value in overrides.items():
        if value is None:
            continue
        set_dotted(result, dotted, value)
    return result


def explicit_overrides(namespace: argparse.Namespace) -> dict[str, Any]:
    """挑出命令行真正显式给出的参数。

    前提：脚本对"可由配置提供"的选项使用 ``default=argparse.SUPPRESS``。
    未给出的选项不会出现在命名空间里，因此这里返回的就是需要覆盖配置的项。
    """
    return {key: value for key, value in vars(namespace).items() if value is not None}


def to_jsonable(value: Any) -> Any:
    """把配置转成可写进 ``run_manifest.json`` 的形式（Path → 字符串）。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """把生效配置的副本落盘（``run_manifest.json`` 要求的"配置文件副本"）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(to_jsonable(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target
