"""run 目录登记（手册 §5.3「命名与登记」）。

§5.3 要求：**每个 run 目录必须含 ``run_manifest.json``**，记录
``git`` 状态（若有）、随机种子、配置文件副本、数据集实际记录数/受试者数、
采样率、通道列表、代码版本哈希。

设计要点
--------

* **代码版本哈希**不依赖 git：对所有源码文件的内容做 sha256 汇总。这样即便某个
  快照被导出成不含 ``.git`` 的目录，仍能判断"跑的到底是哪版代码"。
* **git 状态**单独记录 head / dirty / porcelain，便于回溯到具体 commit。
* 写 manifest 的时机是 run 目录创建之后、训练开始之前——**即使训练中途崩溃，
  登记信息也已落盘**。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import paths


__all__ = [
    "CODE_DIRS",
    "CODE_SUFFIXES",
    "code_version_hash",
    "git_state",
    "write_run_manifest",
]


#: 参与代码版本哈希的目录（相对 PROJECT_ROOT）。
CODE_DIRS = ("unicore_eeg", "scripts", "tests")

#: 参与哈希的后缀。
CODE_SUFFIXES = (".py",)


def _iter_code_files() -> list[Path]:
    files: list[Path] = []
    for name in CODE_DIRS:
        root = paths.PROJECT_ROOT / name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix in CODE_SUFFIXES and path.is_file():
                files.append(path)
    return sorted(files)


def code_version_hash(suffixes: Iterable[str] = CODE_SUFFIXES) -> str:
    """对 ``unicore_eeg`` / ``scripts`` / ``tests`` 下所有源码内容做 sha256 汇总。

    逐个文件先哈希、再把 ``相对路径\\0文件哈希`` 按路径排序后串起来做一次总哈希，
    因此对文件改名、内容改动、增删文件都敏感，且与文件系统遍历顺序无关。
    """
    digest = hashlib.sha256()
    entries: list[tuple[str, str]] = []
    allowed = tuple(suffixes)
    for path in _iter_code_files():
        if path.suffix not in allowed:
            continue
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append((str(path.relative_to(paths.PROJECT_ROOT)).replace("\\", "/"), file_digest))
    for relative, file_digest in entries:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def git_state(cwd: Path | None = None) -> dict[str, Any]:
    """git 状态；仓库不存在时返回 ``available=False`` 而不是抛错。"""
    workdir = cwd or paths.PROJECT_ROOT

    def run(*args: str) -> tuple[int, str]:
        try:
            result = subprocess.run(
                ["git", *args], cwd=workdir, capture_output=True, text=True, timeout=20
            )
        except (OSError, subprocess.SubprocessError) as error:
            return 1, f"{type(error).__name__}: {error}"
        return result.returncode, result.stdout.strip()

    code, head = run("rev-parse", "HEAD")
    if code != 0:
        return {"available": False, "detail": head}
    _, porcelain = run("status", "--porcelain")
    _, branch = run("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "available": True,
        "head": head,
        "branch": branch,
        "dirty": bool(porcelain),
        "porcelain": porcelain,
    }


def write_run_manifest(
    out_dir: str | Path,
    *,
    command: Iterable[str] | None = None,
    settings: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    filename: str = "run_manifest.json",
) -> Path:
    """把登记信息写进 ``out_dir``，返回 manifest 路径。

    ``config`` 同时以 ``config.yaml`` 形式落盘（§5.3 明确要求"配置文件副本"），
    既是给人看的，也是给后续脚本复跑用的。**不会写入外部数据池**。
    """
    target_dir = Path(out_dir)
    paths.assert_readonly_pool(target_dir, operation="write run manifest into")
    target_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project_root": str(paths.PROJECT_ROOT),
        "git": git_state(),
        "code_version_hash": code_version_hash(),
        "code_hash_scope": list(CODE_DIRS),
    }
    if command is not None:
        payload["command"] = [str(item) for item in command]
    if settings is not None:
        payload["settings"] = _jsonable(settings)
    if config is not None:
        from .config import save_config, to_jsonable

        saved = save_config(config, target_dir / "config.yaml")
        payload["config_file"] = str(saved)
        payload["config"] = to_jsonable(config)
    if dataset is not None:
        payload["dataset"] = _jsonable(dataset)
    if extra is not None:
        payload["extra"] = _jsonable(extra)

    manifest_path = target_dir / filename
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest_path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
