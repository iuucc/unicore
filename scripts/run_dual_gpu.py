"""双卡并行调度（手册 T0.7 步骤 5，对应 §0.3.6 策略 A）。

把"任务列表 + 配置文件"轮流分派到两个 `CUDA_VISIBLE_DEVICES` 进程，每个进程一张卡：

* **不使用 DDP / 跨卡梯度同步**——本机 NCCL 不可用，且两块卡在不同 PCIe 根端口（§0.3.6）。
* 每个进程 `--num-workers` 默认为 8，且强制 `len(gpus) * num_workers <= 16`（§0.3.7 的内存约束）。
* 采样每张卡的 `nvidia-smi` 利用率与显存，按任务生命周期汇总。
* 任务结束或异常时用进程树终止（Windows `taskkill /F /T`）**回收 DataLoader 子 worker**，
  避免 §0.3.7 坑 2 那种"一次泄漏 16 进程 ≈ 16.5 GB"的情况。

用法示例：

```bash
# 三种子并行（每个种子一个进程，两张卡轮流）
python scripts/run_dual_gpu.py --config configs/stage_b.yaml --seeds 42 1234 20260922

# 驱动任意入口（例如用首个实验脚本做一次双卡连通性验证）
python scripts/run_dual_gpu.py --script scripts/first_experiment.py \
  --extra "--epochs 1 --train-samples 256 --eval-samples 128 --batch-size 32" \
  --seeds 42 1234 --out-prefix runs/_env/dual_gpu_smoke
```
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unicore_eeg import paths


MAX_TOTAL_WORKERS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round-robin two independent runs over two GPUs (T0.7).")
    parser.add_argument("--config", type=Path, help="传给每个任务的配置文件")
    parser.add_argument(
        "--script",
        type=Path,
        help="任务入口脚本；默认由 --config 推导（configs/stage_b.yaml → scripts/train_stage_b.py）",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1234, 20260922], help="每个种子一个任务")
    parser.add_argument("--gpus", nargs="+", default=["0", "1"], help="CUDA_VISIBLE_DEVICES 取值列表")
    parser.add_argument("--num-workers", type=int, default=8, help="每进程的 DataLoader worker 数")
    parser.add_argument("--out-prefix", type=Path, default=None, help="各任务输出目录前缀")
    parser.add_argument("--python", default=sys.executable, help="解释器路径")
    parser.add_argument("--extra", default="", help="追加到每个任务命令行的额外参数（字符串）")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="nvidia-smi 采样间隔")
    parser.add_argument("--dry-run", action="store_true", help="只打印调度计划，不真正执行")
    parser.add_argument("--report", type=Path, default=None, help="调度报告输出路径")
    return parser.parse_args()


def script_for_config(config: Path) -> Path:
    """configs/stage_b.yaml → scripts/train_stage_b.py（手册附录 A 的命名约定）。"""
    stem = config.stem
    return paths.PROJECT_ROOT / "scripts" / f"train_{stem}.py"


@dataclass
class Task:
    name: str
    seed: int
    gpu: str
    out_dir: Path
    command: list[str] = field(default_factory=list)
    started_utc: str | None = None
    finished_utc: str | None = None
    returncode: int | None = None
    duration_s: float | None = None
    message: str = ""


class GpuSampler(threading.Thread):
    """后台采样每张卡的利用率与显存；nvidia-smi 不可用时静默停用。"""

    def __init__(self, gpus: list[str], interval: float) -> None:
        super().__init__(daemon=True)
        self.gpus = list(gpus)
        self.interval = max(interval, 0.5)
        self.samples: list[dict[str, Any]] = []
        # 注意：不要把这个事件命名为 _stop —— threading.Thread 内部就有 _stop 方法，
        # 覆盖它会让 Thread.join() 抛 TypeError: 'Event' object is not callable。
        self._halt = threading.Event()
        self.available = shutil.which("nvidia-smi") is not None

    def run(self) -> None:
        if not self.available:
            return
        while not self._halt.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                now = time.perf_counter()
                for line in result.stdout.strip().splitlines():
                    parts = [item.strip() for item in line.split(",")]
                    if len(parts) != 3:
                        continue
                    self.samples.append(
                        {
                            "t": now,
                            "gpu": parts[0],
                            "util": float(parts[1]),
                            "memory_mib": float(parts[2]),
                        }
                    )
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=self.interval + 2.0)

    def summarize(self, gpu: str, start: float, end: float) -> dict[str, Any]:
        window = [s for s in self.samples if s["gpu"] == gpu and start <= s["t"] <= end]
        if not window:
            return {"samples": 0}
        utils = [s["util"] for s in window]
        memories = [s["memory_mib"] for s in window]
        return {
            "samples": len(window),
            "util_mean_percent": round(sum(utils) / len(utils), 1),
            "util_max_percent": round(max(utils), 1),
            "memory_max_mib": round(max(memories), 1),
        }


def children_of(pid: int) -> list[int]:
    try:
        import psutil
    except ImportError:
        return []
    try:
        process = psutil.Process(pid)
    except psutil.Error:
        return []
    kids = []
    for child in process.children(recursive=True):
        try:
            kids.append(child.pid)
        except psutil.Error:
            continue
    return kids


def terminate_tree(process: subprocess.Popen) -> None:
    """终止进程树，回收 DataLoader 子 worker（§0.3.7 坑 2）。"""
    if process.poll() is not None:
        return
    kids = children_of(process.pid)
    if os.name == "nt":
        # taskkill /T 会连同子进程一起结束
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            text=True,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    # 兜底：仍存活的子进程逐个清掉
    for pid in kids:
        try:
            import psutil
        except ImportError:
            break
        try:
            psutil.Process(pid).kill()
        except psutil.Error:
            continue


def build_tasks(args: argparse.Namespace) -> list[Task]:
    if args.config is None and args.script is None:
        raise SystemExit("需要 --config 或 --script 之一")
    if args.config is not None and args.script is None:
        script = script_for_config(args.config)
    else:
        script = Path(args.script)  # type: ignore[arg-type]
    if not script.exists():
        raise SystemExit(f"入口脚本不存在：{script}")

    prefix = args.out_prefix if args.out_prefix is not None else paths.RUNS_ROOT / script.stem.replace("train_", "")
    tasks = []
    for index, seed in enumerate(args.seeds):
        name = f"{script.stem}_seed{seed}"
        command = [
            args.python,
            "-u",  # 关闭缓冲：长训练必须能实时看到进度（实测被块缓冲坑过）
            str(script),
        ]
        if args.config is not None:
            command += ["--config", str(args.config)]
        command += ["--seed", str(seed), "--num-workers", str(args.num_workers)]
        out_dir = Path(prefix) / f"seed{seed}" if len(args.seeds) > 1 else Path(prefix)
        command += ["--out", str(out_dir)]
        if args.extra:
            command += args.extra.split()
        tasks.append(
            Task(
                name=name,
                seed=seed,
                gpu=args.gpus[index % len(args.gpus)],
                out_dir=out_dir,
                command=command,
            )
        )
    return tasks


def run_task(task: Task, python: str, report_lines: list[str]) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = task.gpu
    task.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = task.out_dir / "stdout.log"
    task.started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.Popen(
            task.command,
            cwd=paths.PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            task.returncode = process.wait()
        except BaseException:
            terminate_tree(process)
            task.returncode = -1
            task.message = "terminated by scheduler"
            raise
    task.duration_s = round(time.perf_counter() - started, 1)
    task.finished_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if task.returncode != 0:
        task.message = f"exit code {task.returncode}"
    report_lines.append(
        f"  {task.name:<28s} gpu={task.gpu} rc={task.returncode} {task.duration_s:>8.1f}s  log={log_path}"
    )


def gpu_worker(
    gpu: str,
    queue: "deque[Task]",
    python: str,
    report_lines: list[str],
    windows: dict[str, tuple[float, float]],
) -> None:
    """单张卡上的串行执行器；不同卡之间由各自的线程并行推进。"""
    while queue:
        task = queue.popleft()
        started = time.perf_counter()
        run_task(task, python, report_lines)
        windows[task.name] = (started, time.perf_counter())


def main() -> None:
    args = parse_args()
    total_workers = len(args.gpus) * args.num_workers
    if total_workers > MAX_TOTAL_WORKERS:
        raise SystemExit(
            f"总 worker 数 {total_workers} > {MAX_TOTAL_WORKERS}：每 worker 常驻约 1.4 GB，"
            "会耗尽本机 31.8 GB 内存（手册 §0.3.7）。请减少 --num-workers 或 --gpus。"
        )

    tasks = build_tasks(args)
    report_path = args.report or paths.RUNS_ROOT / "_env" / "dual_gpu_schedule.json"

    print(f"gpus={args.gpus} num_workers={args.num_workers} total_workers={total_workers}")
    for task in tasks:
        print(f"  {task.name:<28s} -> gpu {task.gpu}   out={task.out_dir}")
        print(f"    {' '.join(task.command)}")
    if args.dry_run:
        return

    # 每张卡一个队列、一个线程：卡内串行，卡间并行。
    queues: dict[str, deque[Task]] = {gpu: deque() for gpu in args.gpus}
    for index, task in enumerate(tasks):
        queues[args.gpus[index % len(args.gpus)]].append(task)

    report_lines: list[str] = []
    windows: dict[str, tuple[float, float]] = {}
    sampler = GpuSampler(args.gpus, args.poll_seconds)
    sampler.start()
    started_wall = time.perf_counter()

    threads = [
        threading.Thread(
            target=gpu_worker,
            args=(gpu, queue, args.python, report_lines, windows),
            name=f"gpu{gpu}",
        )
        for gpu, queue in queues.items()
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sampler.stop()

    wall_clock_s = round(time.perf_counter() - started_wall, 1)
    serial_estimate_s = round(
        sum(item for item in (task.duration_s for task in tasks) if item is not None), 1
    )

    per_task = []
    for task in tasks:
        window = windows.get(task.name)
        gpu_stats = sampler.summarize(task.gpu, *window) if window else {"samples": 0}
        per_task.append(
            {
                "name": task.name,
                "seed": task.seed,
                "gpu": task.gpu,
                "command": task.command,
                "out_dir": str(task.out_dir),
                "started_utc": task.started_utc,
                "finished_utc": task.finished_utc,
                "duration_s": task.duration_s,
                "returncode": task.returncode,
                "message": task.message,
                "gpu_utilization": gpu_stats,
            }
        )

    payload = {
        "schema_version": 1,
        "scheduler": "scripts/run_dual_gpu.py",
        "measured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpus": args.gpus,
        "num_workers_per_process": args.num_workers,
        "total_workers": total_workers,
        "gpu_sampler_available": sampler.available,
        "config": str(args.config) if args.config else None,
        "wall_clock_s": wall_clock_s,
        "sum_of_task_durations_s": serial_estimate_s,
        "tasks": per_task,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n任务结果：")
    for line in report_lines:
        print(line)
    print(f"\n墙钟 {wall_clock_s:.1f}s（各任务耗时之和 {serial_estimate_s:.1f}s）")
    print(f"报告：{report_path}")
    failures = [item for item in per_task if item["returncode"] != 0]
    if failures:
        raise SystemExit(f"{len(failures)} task(s) failed")


if __name__ == "__main__":
    main()
