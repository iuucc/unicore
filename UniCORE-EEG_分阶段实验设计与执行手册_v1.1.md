# UniCORE-EEG 分阶段实验设计与执行手册 v1.1

> 编制日期：2026-09-22
> 编制依据：`UniCORE-EEG_统一脑电伪迹去除模型设计_v2.md`（v2.1）、`UniCORE-EEG_进度评估与行动建议_20260922.md`
> 文档性质：**可自动执行的实施规格书**。本文档的读者是执行 AI，不是人。所有路径、函数名、验收条件均为硬性要求。
> 与前序文档的关系：设计文档 v2.1 定义"要做成什么样"（WHAT），本手册定义"按什么顺序、改哪个文件的哪一行、用什么命令验证"（HOW）。
> **v1.1 变更**：环境章节（0.3）重写为实测版——硬件确认为 **2 × RTX 5080**；补充实测性能数据与四条能力约束；新增 T0.7（环境加固与性能基线）与 §5.5（运行配置）。

---

## 0. 给执行 AI 的元指令

**开工前必须完整读一遍本文档第 1–5 章。** 第 6 章之后是任务清单，可按 Gate 顺序推进。

### 0.1 执行纪律

1. **一次只做一个任务。** 每个任务都有独立的验收条件（`验收:` 行）。验收不通过，不得进入下一个任务，也不得修改验收条件来"通过"。
2. **不得跳步。** 阶段 Gate 是硬门禁。Gate 未通过就动手下一阶段，会产出无法归因的实验结果。
3. **不得删除既有产物。** `runs/` 下已有目录（尤其是 `runs/full_experiment_v2_fixed/`）是当前唯一的历史基线，只能新增目录，不得覆盖。
4. **不得修改设计文档 v2.1。** 它是冻结的规格。本手册与它冲突时，以设计文档为准，并在本文档末尾的"变更记录"中登记冲突。
5. **每完成一个任务，追写** `runs/<stage>/NOTES.md`（新建目录时同步新建），记录：任务编号、执行的命令、关键输出、遇到的问题、下一步。
6. **数字一律实测。** 任何写进报告的指标必须是命令输出，不得估算、不得从旧 run 抄（除非明确标注来源）。
7. **遇到需要主观判断的地方，停下来写 `BLOCKED.md`。** 不要自己替用户决定技术路线。

### 0.2 三件事永远不能做

- 向 `D:\codexwork\eeg\` 写入任何文件（那是只读外部池 + 旧项目）。
- 把外部池数据复制进 `D:\codexwork\unicore\`。
- 修改 `runs/full_experiment_v2_fixed/` 下的任何文件。

### 0.3 运行环境（全部为 2026-09-22 实测值）

#### 0.3.1 conda 环境

用户已为本项目创建专用 conda 环境，**名称 `unicore-eeg`，位置 `D:\Anaconda_envs\envs\unicore-eeg`**。

| 项 | 值 |
|---|---|
| 环境名 | `unicore-eeg` |
| 环境路径 | `D:\Anaconda_envs\envs\unicore-eeg` |
| 解释器绝对路径 | `D:\Anaconda_envs\envs\unicore-eeg\python.exe` |
| Python | 3.11.16 |

**三条使用纪律**

1. **所有脚本调用一律写解释器绝对路径**，不要依赖 `conda activate` 的 shell 状态：
   ```bash
   PY="D:/Anaconda_envs/envs/unicore-eeg/python.exe"     # Bash
   & "D:\Anaconda_envs\envs\unicore-eeg\python.exe"      # PowerShell
   ```
   理由：本机 Bash 工具有 PATH 损坏问题，`conda activate` 在非交互 shell 中不可靠。
2. **不要用同机的其它环境。** 本机还存在 `eeg`、`LenAdapt-Ensemble`、`convnext-cifar`、`vitd`、`knn-accel`、`hello-agent` 等环境（其中 `D:\Anaconda_envs\envs\eeg` 是上一代 MetacorrSeg 项目的环境）。**混用会引入不可复现的依赖差异。**
3. 装包只往 `unicore-eeg` 装，命令形如：
   ```bash
   "D:/Anaconda_envs/envs/unicore-eeg/python.exe" -m pip install <pkg>
   ```
   装完必须同步更新 `requirements.txt`、`pyproject.toml`、`environment.yml` 三处。

**Bash 工具的 PATH 修复**（每条命令前必须加）：

```bash
export PATH="/usr/bin:/bin:/c/Windows/System32:/c/Windows:/c/Users/admin/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"
```

若 PowerShell 工具出现"exit code 0 但无输出"，**改用 Bash**；确需 PowerShell 时把结果写进文件再读。

---

#### 0.3.2 硬件实测

| 项 | 实测值 | 备注 |
|---|---|---|
| GPU | **2 × NVIDIA GeForce RTX 5080** | `nvidia-smi` 确认两块，非单卡 |
| 架构 / 算力 | Blackwell，compute capability **12.0（sm_120）** | 需 CUDA ≥ 12.8 |
| 单卡显存 | **15.92 GiB**（16303 MiB） | 合计约 **31.8 GiB** |
| SM 数 | 84 / 卡 | |
| L2 缓存 | 64 MiB / 卡 | |
| PCIe 总线 | `01:00.0` 与 `05:00.0` | **两块卡在不同根端口**，跨卡通信需绕行 CPU |
| 驱动 | 595.97 | |
| CPU | **32 逻辑核** | |
| 系统内存 | **31.8 GiB 总** | **可用常低于 20 GiB** —— 见 0.3.7，这是本机最紧的资源 |
| 磁盘 | C: 151 GB 可用；D: 194 GB 可用 | 项目与数据均在 D: |

---

#### 0.3.3 软件栈实测

| 组件 | 版本 | 组件 | 版本 |
|---|---|---|---|
| torch | **2.11.0+cu128** | numpy | 2.4.6 |
| cuda.build | **12.8** | scipy | 1.17.1 |
| cudnn | 9.19.0 | pandas | 2.3.3 |
| mne | 1.13.2 | scikit-learn | 1.9.1 |
| wfdb | 4.3.1 | h5py | 3.16.0 |
| openneuro | 2026.7.1 | tabulate | 0.10.0 |
| matplotlib | 3.11.1 | tqdm | 4.70.0 |
| **PyYAML** | **缺失** | **psutil** | **缺失** |
| **Triton** | **缺失** | | |

---

#### 0.3.4 四条决定技术路线的能力约束（必须先读）

| 能力 | 实测 | 对技术路线的硬性影响 |
|---|---|---|
| bf16 支持 | **True** | 训练用 `torch.autocast("cuda", dtype=torch.bfloat16)`。**不需要 `GradScaler`**（bf16 无需损失缩放）。现有代码 `GradScaler(enabled=False)` 正确，保留 |
| **NCCL** | **不可用（False）** | **不能使用 DDP**。Windows 上只有 gloo，跨卡梯度同步会成为瓶颈。→ 见 0.3.6 |
| **Triton** | **缺失** | **不要用 `torch.compile`**。`scripts/train.py --compile` 在本机不可靠，**禁止在正式实验中启用** |
| TF32 / matmul high | 可用 | `torch.set_float32_matmul_precision("high")` + `torch.backends.cudnn.benchmark = True`（现有脚本已设，保留）。固定输入形状下有效 |
| flash SDP | enabled | 保留默认 |

---

#### 0.3.5 必须补装的依赖（第一个动作）

`PyYAML` 缺失会**直接阻断本手册的配置化要求**（手册多处要求 `configs/*.yaml`）。`psutil` 用于内存监控（本机内存紧张，必须监控）。

```bash
"D:/Anaconda_envs/envs/unicore-eeg/python.exe" -m pip install pyyaml psutil
"D:/Anaconda_envs/envs/unicore-eeg/python.exe" -c "import yaml, psutil; print(yaml.__version__, psutil.__version__)"
```

装完后把 `pyyaml`、`psutil` 补进 `requirements.txt` / `pyproject.toml` / `environment.yml`。

---

#### 0.3.6 双 RTX 5080 的正确用法

**策略 A —— 首选：双卡各跑一个独立实验（官方推荐路线）**

两块卡分别执行两个互不通信的进程，各自绑定一张卡：

```bash
# 进程 1
CUDA_VISIBLE_DEVICES=0 "D:/Anaconda_envs/envs/unicore-eeg/python.exe" scripts/train_stage_b.py \
  --config configs/stage_b.yaml --seed 42 --out runs/stage_b_seed42
# 进程 2（并行，另一张卡）
CUDA_VISIBLE_DEVICES=1 "D:/Anaconda_envs/envs/unicore-eeg/python.exe" scripts/train_stage_b.py \
  --config configs/stage_b.yaml --seed 1234 --out runs/stage_b_seed1234
```

**为什么这是首选**（三条理由都来自实测）：
1. 本机 **NCCL 不可用**，跨卡梯度同步只能走 gloo，代价高。
2. 两块卡在**不同 PCIe 根端口**，跨卡通信要经 CPU，进一步放大同步开销。
3. 本研究流程天然存在**可并行的独立单元**：T3.7 的 3 个随机种子、P3 的多组消融、P3 的多数据集交叉验证。这些本就无需通信。

→ **行动要求**：T3.7 的三种子训练**必须**用双卡并行，每个种子一个进程。这能把三种子的墙钟时间减半。

**策略 B —— 次选：单实验 DataParallel**

只有在"单个实验必须更快"时才用。**使用前必须先修一个既有缺陷：**

> `scripts/train.py:108` 与 `scripts/first_experiment.py:115` 都是 `loss.backward()`。在 `torch.nn.DataParallel` 下，损失会被跨卡聚合成**长度 = 卡数的向量**，此时 `backward()` 直接报
> `RuntimeError: grad can be implicitly created only for scalar outputs`。
> **修法**：`loss.backward()` → `loss.sum().backward()`（或 `loss.mean()`）。修复前 `train.py --data-parallel` 实际不可用。

**禁止事项**

- **禁止**把一个模型拆到两张卡上做梯度同步（无 NCCL + 异根端口）。
- **禁止**两进程共用同一张卡（显存不足，见 0.3.8）。
- 双卡并行时，**两进程的系统内存与 DataLoader worker 总数必须联合预算**，见 0.3.7。

---

#### 0.3.7 DataLoader 与 CPU：本机最大的性能浪费

**现状问题**：`scripts/train.py:25` 与 `scripts/first_experiment.py:43` 的 `--num-workers` **默认值为 0**，即数据加载完全在主进程串行执行。合成数据集每个样本都要做 FFT、`torch.linalg.lstsq`、多通道混音，是 CPU 密集型操作。

**实测对照**（C=34、batch=128、`use_public_sources=True`）：

| `num_workers` | 数据吞吐（samples/s） | 相对 0 worker |
|---:|---:|---:|
| **0（当前默认）** | **310.3** | 1.0× |
| 4 | 2399.1 | 7.7× |
| 8 | 2274.9 | 7.3× |
| 12 | 4412.0 | 14.2× |

**对照 GPU 侧实测吞吐**（同配置 C=34、batch=128）：**402.5 samples/s**。

**结论——这是本机最重要的一条优化**：

- `num_workers=0` 时，数据加载 310 samples/s **低于** GPU 的 402 samples/s，**GPU 有约 23% 的时间在等数据**。
  单通道情形更糟（GPU 477 samples/s vs 加载 310 samples/s，**GPU 约 35% 时间闲置**）。
- 把 `num_workers` 提到 8 以上，加载能力（>2200 samples/s）远超 GPU 能力（~400 samples/s），**瓶颈回到 GPU，这才是正常状态**。
- 预计端到端加速：单通道约 **1.5×**，34 通道约 **1.3×**。

**推荐配置**

```yaml
# configs/base.yaml
dataloader:
  batch_size: 128
  num_workers: 8            # 每进程
  persistent_workers: true  # Windows 下必须，否则每 epoch 重启 worker 极慢
  prefetch_factor: 4
  pin_memory: true
```

**系统内存是硬约束，必须据此限额**

实测每个 DataLoader worker 常驻约 **1.4 GB**（`use_public_sources=True` 时 mmap 的 npy + 副本）。本机系统内存 **31.8 GiB 总，可用常低于 20 GiB**。

→ **规则：双卡并行时，两进程 worker 总数 ≤ 16（即每个进程 8）。** 24 个 worker 会吃掉约 34 GB，直接耗尽内存。

**Windows 多进程的两个坑（已实际踩过，务必记住）**

1. **入口必须有 `if __name__ == "__main__":` 保护。** 缺了会在 `num_workers>0` 时报
   `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase`。
   现有 `scripts/*.py` 都有该保护，**新增脚本必须照做**。
2. **父进程被强杀后，子 worker 不会自动退出。** 实测一次失败运行泄漏了 **16 个 worker，合计约 16.5 GB 内存**，导致后续所有任务报
   `OpenBLAS error: Memory allocation still failed after 10 retries`。
   → 每次异常中断后，必须检查并清理残留 python 进程（见 0.3.9 的自检脚本）。

---

#### 0.3.8 显存预算与推荐 batch size（实测）

模型参数量：**15.37 M**（C=1）／**15.57 M**（C=34）。下表为 bf16 autocast 下 forward+backward+优化器一步的 PyTorch 分配器峰值（单卡）：

| C | batch | ms/step | samples/s | 分配器峰值 GiB | 结论 |
|---:|---:|---:|---:|---:|---|
| 1 | 64 | 225.6 | 283.6 | 3.94 | 可用 |
| 1 | 128 | 268.2 | **477.2** | 7.63 | **推荐** |
| 1 | 256 | — | — | — | **OOM** |
| 34 | 64 | 213.7 | 299.5 | 4.41 | 可用 |
| 34 | 128 | 318.0 | **402.5** | 8.56 | **推荐** |
| 34 | 256 | — | — | — | **OOM** |

**关键结论**

- **`batch_size = 128` 是本机的安全上限**；256 在 15.92 GiB 单卡上直接 OOM（中间激活 + MRSTFT 损失占用大）。
- batch 128 时分配器峰值 7.63–8.56 GiB，**留出约 7 GiB 余量**，容得下 128 通道（ds004784）实验。若 128 通道下 bs=128 OOM，退到 bs=64 或启用梯度累积（`grad_accum_steps=2`）以保持等效 batch。
- 注意 `nvidia-smi` 显示的占用量会**高于**分配器峰值（CUDA context + 碎片），空闲态可见 1.9–4.3 GiB。**判断 OOM 余量看分配器峰值，判断能否起进程看 nvidia-smi。**
- 参数量仅 15.4 M 却在小 batch 下 OOM，说明**本模型是激活内存受限而非参数受限**——因此提升 batch 应优先减少中间张量（例如 `multi_resolution_stft_loss` 的 3 个 n_fft 逐个计算），而不是减参数。

**充分利用单卡 15.92 GiB 的具体做法**

- 评估集从 6000 提到 **30000+**（评估不反传，显存占用远小于训练）。
- 训练样本从 65536 提到 **200000+**（合成数据可无限生成，瓶颈是时间不是显存）。
- 打开 `persistent_workers` + `prefetch_factor=4`，让显存和算力都不空转。
- 三种子实验用双卡并行（0.3.6 策略 A）而非单卡串行。

---

#### 0.3.9 环境自检（每个阶段开始前跑一次）

```bash
PY="D:/Anaconda_envs/envs/unicore-eeg/python.exe"
# 1) 依赖与版本
"$PY" -c "import torch,yaml,psutil,mne,wfdb;print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
# 2) 残留进程检查（应为空）
"D:/Anaconda_envs/envs/unicore-eeg/python.exe" -c "
import subprocess,sys
out=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],capture_output=True,text=True).stdout
print(out)"
# 3) 双卡可用性 + 显存
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
```

**若出现 `OpenBLAS error: Memory allocation still failed`** → 系统内存被残留进程占满。清理方式（PowerShell，精确匹配项目脚本名，**不要无差别杀 python 进程**）：

```powershell
$targets = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
  Where-Object { $_.CommandLine -like "*scripts*train*.py*" -or $_.CommandLine -like "*multiprocessing-fork*" }
$targets | Select-Object ProcessId,@{n='MB';e={[int]($_.WorkingSetSize/1MB)}},CommandLine | Format-List
# 核对无误后再执行：
# $targets | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

---


## 1. 三项决策的落地方式

用户已确认三项决策，全部为**最高优先级硬约束**，覆盖 `进度评估与行动建议_20260922.md` 第 7 章中我原先的保守建议。

| 决策 | 内容 | 手册中的落地位置 |
|---|---|---|
| **D1 多通道必须实现** | 不再走"先单通道验证、后切多通道"的迂回路线。**数据集是多通道的就直接按多通道方案一步到位**，不做单通道中间版本。 | 第 6 章（架构设计）+ 阶段 1 全部任务 + 阶段 2 起所有 loader 一律按原生多通道读取 |
| **D2 外部池只读引用** | 直接引用主机现有文件，不复制、不额外配置、不建软链。 | 第 3 章路径总表 + `unicore_eeg/paths.py`（T0.3） |
| **D3 ds004784 规范化** | 目录统一命名为 `ds004784`（当前为 `on004784`），并补一份含大小/哈希的 `download_manifest.json`。 | T0.2 |

### D1 的连带影响（必须在设计上想清楚，否则阶段 1 会返工）

**"一步到位"意味着放弃单通道作为主线的退路。** 由此产生三个必须现在就定的问题：

1. **通道数按数据集分别固定，不追求跨数据集统一。** 理由：设计文档 §2.2 明确要求"所有结果必须按通道设置分别报告，不得用'主体结构兼容'暗示能力等价"。强行把 2 导（Sleep-EDF）和 128 导（ds004784）统一到同一 C 上，只会引入插值误差并掩盖差异。
   → **规则：每个实验配置文件声明一个 `montage`，实验内 C 固定；跨数据集比较时在报告中显式列出各自的 C 与通道列表。**

2. **模型必须支持任意 C，不能有依赖固定 C 的层。** 现有 `model.py` 的卷积层天然支持任意 C，但 `ArtifactObservationExtractor` 目前把多通道压成单通道（详见 6.3），必须改造。新增的空间模块一律用 DeepSets 式（per-channel MLP + 池化）结构，禁止 `nn.Linear(C, …)`。

3. **合成器的"多通道"必须是真混音，不能是复制。** 现有 `synthetic.py:125-131` 的 `_expand_channels` 只是对同一个源做缩放 + 时移后堆叠，通道间没有真正的空间混合信息。**空间投影模块在这种数据上什么也学不到。** 必须替换为混音矩阵方案（T1.4）。

---

## 2. 目录与路径总表（职责边界）

这一章是本文档最重要的一章。**执行 AI 在任何时候都必须清楚当前操作落在哪个目录、该目录允许什么操作。**

### 2.1 三个根目录

| 代号 | 绝对路径 | 角色 | 允许的操作 |
|---|---|---|---|
| **CODE** | `D:\codexwork\unicore` | 唯一的代码仓库与产物根 | 读写 |
| **DATA** | `D:\codexwork\unicore\data` | 项目内已下载数据 + 数据登记文件 | 读写（`raw/` 下的大文件只读） |
| **POOL** | `D:\codexwork\eeg\data` | 外部数据池，约 190 GB | **只读** |

### 2.2 CODE 目录内部职责

| 路径 | 职责 | 内容约束 |
|---|---|---|
| `D:\codexwork\unicore\unicore_eeg\` | 库代码（可被 import） | 新增模块见第 6 章与各阶段任务 |
| `D:\codexwork\unicore\scripts\` | 可执行入口脚本 | 所有脚本必须支持 `--config`，禁止硬编码路径 |
| `D:\codexwork\unicore\tests\` | 单元测试 | 每个新模块必须有对应测试 |
| `D:\codexwork\unicore\configs\` | **新建**。实验配置（YAML） | 一个实验一个文件，禁止复用同一文件跑不同实验 |
| `D:\codexwork\unicore\runs\` | 实验产物 | 一个实验一个子目录，只增不改 |
| `D:\codexwork\unicore\data\raw\` | 项目内已下载的原始数据 | 见 2.3 |
| `D:\codexwork\unicore\UniCORE-EEG_*.md` | 规格与报告 | 设计文档 v2.1 冻结 |
| `D:\codexwork\unicore\1.5` | 遗留的空文件（0 字节） | T0.6 可删除 |

### 2.3 DATA 目录内容与可用性

| 路径 | 体积 | 可用性 | 说明 |
|---|---:|---|---|
| `data\raw\ds004784\` | 11 GB | **可用，完整** | 体模数据集，含物理真值。**T0.2 前名为 `on004784`** |
| `data\raw\eegdenoisenet\` | 93 MB | **可用，完整** | 3 个 npy：`EEG_all_epochs_512hz.npy`、`EOG_all_epochs.npy`、`EMG_all_epochs_512hz.npy` |
| `data\raw\mitdb\` | 9.4 MB | **可用，完整** | 仅 5 条记录（100/101/102/103/105），ECG 源池偏小 |
| `data\raw\physiomotion\` | 129 MB | **禁止用于标注类任务** | 只有 sub-1 原始 BIDS；`derivatives\preprocessed_BIDS\` 内仅一个 README，**无 EDF、无标注**。PhysioMotion 一律走 POOL（T2.6） |
| `data\DATASETS.md` | — | 登记表 | 数据集许可与用途说明 |
| `data\download_manifest.json` | — | 校验记录 | 当前只覆盖 EEGdenoiseNet + MIT-BIH；T0.2 后扩到 ds004784 |
| `data\local_sources.json` | — | POOL 路径声明 | T0.3 后由 `unicore_eeg/paths.py` 读取，不再散落各处 |

### 2.4 绝对禁止

1. 禁止创建、修改、删除 `D:\codexwork\eeg\` 下的任何文件（含 `__pycache__`）。需要 import 旧项目代码时，**复制粘贴到 CODE 下并改写**，不要 `sys.path.append` 到 eeg。
2. 禁止把 POOL 下的 `.npy/.npz/.edf/.mat/.gdf/.bdf/.set/.fdt` 复制进 CODE。
3. 禁止在 CODE 下的源码或配置里出现写死的 `D:\codexwork\eeg\...`。POOL 路径只能出现在 `unicore_eeg/paths.py` 一处。

---

## 3. 外部池数据集清单（POOL）

根路径 `D:\codexwork\eeg\data`。**下表路径均为相对 POOL 根**，`unicore_eeg/paths.py` 负责拼接。

### 3.1 主用数据集

| 数据集 | 相对路径 / 文件模式 | 采样率 | 通道 | 用途（对应设计文档章节） | 完整性 |
|---|---|---|---|---|---|
| **physiomotion** | `physiomotion\derivatives\preprocessed_BIDS\sub-{1..30}\eeg\sub-{id}_task-artifact_run-{01..06}_eeg.edf`<br>`physiomotion\derivatives\Manual_Annotations\sub{id}_run*.csv`（180 份） | 待读 | 34 双极 | §9.1 Cross-dataset；真实眼动/肌电/运动多标签路由外测与弱监督适配 | 完整（30 受试者） |
| **bci2a** | `bci2a\A0{1..9}{T,E}.gdf` + 同名 `.mat`（18 对） | 250 | 22 | §9.2 下游任务（运动想象）；§9.1 Clean-input | 完整 |
| **openbmi** | `openbmi\cache\subj{01..54}_session{1,2}_mi.npz`（108 份）+ `openbmi\*.mat`（216 份） | 待读 | 62 | 下游任务（大规模 MI）、Clean-input | 完整 |
| **physionet_mi** | `physionet_mi\four_class_cache\subj{001..109}_run{04,06,08,10,12,14}.npz` + `physionet_mi\*.edf`（654） | 160 | 64 | 4 类 MI 下游任务 | 4 类子集 |
| **chbmit** | `chbmit\chb{01..}\chb{XX}_{NN}.edf`（39 个） | 256 | 23（变） | §9.3 生理负对照（棘波/发作形态） | **子集**（全集约 686 EDF） |
| **sleep_edfx** | `sleep_edfx\cache20\SC4XX_night{1,2}.npz`（415 份）+ `sleep_edfx\*.edf`（306） | 100 | 2 | §9.3 负对照（K 复合波/纺锤波）；睡眠分期下游 | **子集** |
| **cap_sleep** | `cap_sleep\*.edf` + 同名 `.txt`（各 56）+ `cap_sleep\feature_cache\*.npz`（97） | 待读 | 待读 | §9.3 负对照；睡眠下游 | **子集** |
| **ds002094** | `ds002094\sourcedata\**`（43 组 `.vhdr/.vmrk`，107 `.tsv`，120 `.mat`） | 待读 | 待读 | §9.1 Unseen-artifact：TMS 强瞬态压力测试 | 基本完整 |
| **faced** | `faced\feature_cache\sub-XXX.npz`（74）+ `faced\*.bdf`（55）+ `.tsv`（168） | 250 | 32 | §9.2 情感识别下游；Clean-input | **子集**（全集 123 受试者） |
| **cho_gigadb** | `cho_gigadb\meta_cache\s{01..52}.npz`（208）+ `cho_gigadb\*.mat`（52） | 待读 | 待读 | Clean-input、MI 下游 | 52 受试者 |
| **bci2b** | `bci2b\B{01..09}{01..05}{T,E}.gdf`（45）+ `.mat`（45） | 250 | 3 双极 | MI 下游；小通道压力测试 | 完整 |
| **tuar** | — | — | — | **不可用，空目录** | 空 |

### 3.2 使用 POOL 的三条纪律

1. **"待读"字段必须实测填表。** 每个数据集在使用前，先写一个 `--dry-run` 的探测脚本，把 EDF/vhdr/npz 头里的实际采样率、通道名、时长写进 `runs/<stage>/pool_probe.json`。**不得凭记忆或凭本表格填**。
2. **`.npz` 缓存的 schema 未知，必须先 dump。** POOL 下的 `openbmi\cache`、`physionet_mi\four_class_cache`、`sleep_edfx\cache20`、`cho_gigadb\meta_cache`、`faced\feature_cache`、`chbmit\*.npz` 都是旧项目（MetacorrSeg）生成的缓存。**第一次使用前必须 `np.load(...).files` 打印键名与形状**，不得假设字段名。
3. **子集不是全集。** CHB-MIT、Sleep-EDF、CAP Sleep、FACED 只下载了子集，PhysioNet-MI 只有 4 类子集。任何报告凡涉及这些数据集，必须写明实际使用的记录数与受试者数，**不得表述为"在 CHB-MIT 上验证"**，应写"在 CHB-MIT 的 N 个记录子集上验证"。

---

## 4. 可复用资产（POOL 旁的旧项目代码，只读参考）

**重大发现：`D:\codexwork\eeg\src\metacorrseg\` 下有该 POOL 对应的完整 loader 与下游任务实现。** 这套代码是上一代项目的资产，质量可用于本项目，但**不得直接 import**（避免把旧项目的依赖和约定带进来）。

**使用方式：只读打开 → 理解 schema 与算法 → 在 CODE 下重新实现所需的子集 → 加测试。**

| 旧项目文件（只读参考） | 可复用的内容 | 对应本手册任务 |
|---|---|---|
| `src\metacorrseg\data\constants.py` | 通道名常量：`BCI2A_EEG_CHANNELS`（22）、`BCI2B_EEG_CHANNELS`（3）、`BCI2B_EOG_CHANNELS`（3）、`PHYSIOMOTION_BIPOLAR_CHANNELS`（34）、`BIPOLAR_CHANNELS`（18）、`RAW_ALIASES`（原始 EDF 通道名 → 10-20 标准名，19 导）；标签约定 `CLEAN_LABELS/ARTIFACT_LABELS/IGNORE_LABELS`、`IGNORE_INDEX=-100` | T1.1、T2.5、T2.7 |
| `src\metacorrseg\data\cross_domain.py::preprocess_continuous` | **预处理标准动作**：CAR → `resample_poly` 重采样 → 4 阶 Butterworth 带通 → 稳健 MAD 裁剪。这是 POOL 各数据集能对齐的基础 | T2.1 |
| `src\metacorrseg\data\cross_domain.py::CachedSegmentationDataset` | 分段数据集：`_augment_signal_domain` 含通道增益/高斯噪声/通道偏置/基线漂移/通道丢弃/时间掩码/时间平移七种增强；`_augment_paste_artifact` 伪迹粘贴；`_apply_boundary_ignore`、`_apply_positive_label_dilation`、`_loss_weights` | T2.1、T2.4 |
| `src\metacorrseg\data\cross_domain.py::CachedMetaTaskDataset::_overlap_exclusion` | **防重叠泄漏算法**：显式排除与已选窗口时间重叠的候选窗口。这正是设计文档 §4.2 与 P1"防止相邻重叠窗口跨集合"所要求的 | T2.3 |
| `src\metacorrseg\data\manifest.py` | `add_recording_split`（记录级 train/val/test 划分）、`add_recording_kfold_split`（记录级 K 折，docstring 明确"windows from the same EDF cannot leak across splits"）。这是 P1"先划分后切窗"的现成实现 | T2.2 |
| `src\metacorrseg\data\bci2a.py` | GDF 读取、`_bci_channel_map` 通道对齐、从 `.mat` 取 `ArtifactSelection`/`classlabel`、事件码解析（768 试次起、769-772 四类 cue、1023 伪迹段）、7.5 s 试次切分、缓存为 npz | T2.7 下游 |
| `src\metacorrseg\data\{bci2b,cap_sleep,chbmit,cho,faced,openbmi,physiomotion,physionet_mi_meta,sleep_edfx}.py` | 各数据集专用 loader，含缓存 schema 与划分字段 | T2.5–T2.7 |
| `src\metacorrseg\downstream\{bci2a,mi,fbcsp,artifact_stress}.py` | **下游任务已实现**：MI 分类、FBCSP 特征、伪迹压力测试 | T4.6 |
| `src\metacorrseg\training\metrics.py` | 指标实现可参考 | T4.2 |
| `src\metacorrseg\data\edf.py::edf_duration_seconds` | 轻量读取 EDF 时长 | T2.5 |

### 4.1 两套预处理约定的冲突（必须解决，否则下游结论不可比）

| 项 | 旧项目（MetacorrSeg） | 本项目（UniCORE-EEG） |
|---|---|---|
| 目标采样率 | 250 Hz | **500 Hz**（设计文档 §3.1） |
| 段长 | 4.0 s | **2.0 s**（T=1000） |
| 带通 | 0.5–45 Hz | 设计文档未指定，建议同为 0.5–45 Hz |
| CAR | 开 | 待定 |
| 通道 | 34 双极 | **各数据集原生** |

**规则**：UniCORE 的**去伪迹主实验**一律 500 Hz / 2 s。**下游任务**使用 500 Hz 去伪迹后的信号，再按任务惯例重新分段（例如 MI 用 4 s 试次），并在报告中显式声明"去伪迹在 500 Hz/2 s 下进行，下游在 X Hz/Y s 下进行"。**不得**把旧项目 250 Hz 下的既有指标与本项目 500 Hz 下的指标直接对比。所有旧项目的数值只能作为参考量级，不能进入排名表。

---

## 5. 全局技术约定

### 5.1 信号与窗口

| 项 | 约定 |
|---|---|
| 目标采样率 | 500 Hz |
| 窗口长度 | 1000 点（2.0 s） |
| 带通 | 0.5–45 Hz（4 阶 Butterworth，`filtfilt` 零相位） |
| 重采样 | `scipy.signal.resample_poly`（整数比） |
| 归一化 | 模型内部 `robust_normalize`（median/MAD，`model.py:455`），**数据集层不再额外归一化** |
| 通道选择 | 从文件头读取实际通道名，按 `montage` 配置选取；缺导必须报错，不得静默补零 |

### 5.2 划分

- **三级划分：train / val / test。**
- **先划分后切窗。** 划分单位：能拿到受试者 ID 的按受试者；拿不到的按记录（recording）。
- 禁止跨集合共享：相邻重叠窗口、同一清洁片段的不同污染版本、同一伪迹源的不同裁剪、同一受试者的不同连续片段、从同一 ERP 试次派生的多视图。
- 必须实现一个**泄漏检查器**（T2.3），对每个产出的划分打印"重叠窗口对数量"，**必须为 0**。

### 5.3 命名与登记

- 数据集目录名用官方 ID：`ds004784`、`ds002094`、`ds006386`（PhysioMotion）。
- 每个 run 目录必须含 `run_manifest.json`：`git` 状态（若有）、随机种子、配置文件副本、数据集实际记录数/受试者数、采样率、通道列表、代码版本哈希。
- 随机种子固定为 `42 / 1234 / 20260922` 三个（对应 P2 的三种子要求）。

### 5.4 配置化

所有实验参数进 `configs/*.yaml`，脚本只接受 `--config`。**验收方式：`grep -rn "D:\\\\codexwork\\\\eeg" unicore_eeg scripts configs` 必须无输出。**（POOL 路径只允许出现在 `unicore_eeg/paths.py`，见 T0.3 的例外声明。）

### 5.5 运行配置（依据 0.3 节实测，所有实验必须遵守）

```yaml
runtime:
  precision: bf16            # autocast("cuda", dtype=torch.bfloat16)
  grad_scaler: false         # bf16 无需 GradScaler
  torch_compile: false       # 本机无 Triton，禁止启用
  cudnn_benchmark: true
  matmul_precision: high     # set_float32_matmul_precision("high")，即 TF32
  device: cuda               # 双卡并行时由 CUDA_VISIBLE_DEVICES 指定

dataloader:
  batch_size: 128            # 安全上限；256 必 OOM（0.3.8）
  num_workers: 8             # 禁止 0；双卡并行时总 worker ≤ 16（0.3.7）
  persistent_workers: true
  prefetch_factor: 4
  pin_memory: true
  grad_accum_steps: 1        # 需更大等效 batch 时改为 2/4，而非加大 batch_size
```

**四条硬性约束**

1. `torch_compile` 必须为 `false`（本机 Triton 缺失）。
2. `num_workers` 不得为 0；双卡并行时两进程的 `num_workers` **之和不得超过 16**。
3. 多卡只用 `CUDA_VISIBLE_DEVICES` 做进程隔离；**禁止 DDP / 跨卡梯度同步**（本机无 NCCL）。
4. 需要更大等效 batch 时用 `grad_accum_steps`，**不要**把 `batch_size` 提到 256。


---

## 6. 多通道架构设计（D1 的核心）

### 6.1 现状盘点（已逐行核验）

| 组件 | 多通道支持情况 | 结论 |
|---|---|---|
| `SharedStem`（`model.py:86`） | `ConvGNAct(in_channels, 32, 7)`，Conv1d 支持任意 C | **无需改** |
| `ContentEncoder`（`model.py:276`） | 全程 Conv1d | **无需改** |
| `ArtifactExpert`（`model.py:296`） | `Conv1d(in_channels, dim, 1)` | **无需改** |
| `CoarseDecoder`（`model.py:368`） | `clean_head` 与 `artifact_heads` 输出 `in_channels` | **无需改** |
| `ResidualRefiner`（`model.py:396`） | `Conv1d(in_channels*4, 48, 7)` | **无需改** |
| `IdentityGate`（`model.py:421`） | `mean(dim=(1,2))` | **无需改** |
| `UniCORELoss`（`losses.py:61`） | `mad.unsqueeze(1)` 广播正确；STFT 用 `flatten(0,1)` | **无需改** |
| **`ArtifactObservationExtractor`**（`model.py:124`） | **三处把多通道压成单通道** | **必须改**（T1.2） |
| **空间投影** | **完全不存在** | **必须新增**（T1.3） |
| **合成器多通道** | **只是缩放+时移复制，无真混音** | **必须改**（T1.4） |

### 6.2 三处压维的具体位置（T1.2 的改造目标）

1. `model.py:131` `_harmonic_basis`：`spectrum = torch.fft.rfft(y, dim=-1).abs().mean(dim=1)` → 频率估计对全通道平均；`design` 拟合 `y.mean(dim=1)`；最后 `expand_as(y)` → **六类通道拿到完全相同的谐波基**。
2. `model.py:152` `mono = y.mean(dim=1)` → 自相关基于单通道平均信号；`cardiac_lags` 全通道共用一个滞后 → **丢失了各导联 QRS 到达时间与幅度的空间差异**。
3. `model.py:144` `fitted.unsqueeze(1).expand_as(y)` → 谐波拟合结果广播到所有通道。

### 6.3 改造后的观测抽取契约

`ArtifactObservationExtractor.forward(y)` 输入 `y: (B, C, T)`，返回 **6 个形状均为 `(B, C, T)` 的观测**，顺序与 `ARTIFACT_NAMES` 一致：

| 序号 | 专家 | 观测 | 多通道实现要求 |
|---|---|---|---|
| 0 | harmonic | 窄带正弦基拟合 | 频率估计可共享（谱平均找峰，8–80 Hz），但**幅相系数逐通道独立 lstsq**，输出逐通道拟合曲线 |
| 1 | ocular_drift | 慢趋势 | 逐通道移动平均（现状已是逐通道，保留） |
| 2 | myogenic | 高频分量 | 逐通道（现状已是逐通道，保留） |
| 3 | cardiac | 准周期脉冲 | **逐通道自相关寻峰**得到 `(B, C)` 滞后；`periodic` 逐通道构造 |
| 4 | motion_transient | 一阶/二阶差分组合 | 逐通道（现状已是逐通道，保留） |
| 5 | unknown | 已知解释后的残差 | 逐通道 |

**验收：** 单测断言 `len(obs) == 6`，且对 `C=1` 与 `C=4` 两种情况均能前向；断言 `C=4` 时 `obs[0][:, 0] != obs[0][:, 1]`（谐波基不再逐通道相同）。

### 6.4 新增空间投影模块（`SpatialProjectionHead`）

设计依据：设计文档 §3.4.4「多通道模式利用不同电极上的空间投影一致性」、§3.4.2「空间前额优势」、§3.7「源时程 + 空间投影因子化」。

**放置在 `model.py`，接在 `observations` 之后、`experts` 之前。**

**输入**
- `y_norm: (B, C, T)`
- `montage_coords: (C, 2) or (C, 3)`（10-20 标准坐标，来自 T1.1；缺失时置零并置 `has_coords=False`）

**结构（必须满足"任意 C 可前向"）**
1. 通道协方差：`R = y_norm @ y_norm.transpose(-1,-2) / T`，形状 `(B, C, C)`。
2. 取 `R` 的前 `k=4` 个特征向量时间投影：对 `y_norm` 做 SVD 取前 4 个右奇异向量，得到 `(B, 4, T)` 的**空间主成分时序**；再与 `y_norm` 做相关得到逐通道载荷 `(B, C, 4)`。
3. 通道 RMS 拓扑：`rms_c = y_norm.square().mean(-1).sqrt()` → `(B, C)`；与坐标拼接成 `(B, C, 3或4)`。
4. DeepSets 编码：`per_channel_mlp: 3→32→32`（对每个通道独立作用），再 `attention_pool` 得到全局空间上下文 `(B, 32)`。
5. 输出**逐通道空间权重** `w_s: (B, C, 1)`：用 `per_channel_mlp` 的输出 + 全局上下文 → `1` 个 logit → sigmoid。
6. 该权重作为 **cardiac 与 ocular_drift 两个专家观测的调制项**：`obs_k ← obs_k * (1 + α·w_s)`，`α` 可学习。其余四个专家不使用空间权重。

**为什么用 DeepSets 而不是线性层**：`nn.Linear(C, ·)` 会把 C 写死，违反 D1 的"每个实验内 C 固定但跨实验可变"要求。

**验收：** `C ∈ {1, 2, 3, 34, 64, 128}` 六种情况均可前向 + 反向，无形状错误；参数量与 C 无关（打印比对）。

### 6.5 合成器真混音（`SpatialMixer`）

**替换 `synthetic.py:125-131` 的 `_expand_channels`。**

```
对每个样本：
  1. 生成 K 个独立伪迹源 s_k(t)，k ∈ {harmonic, ocular, myogenic, cardiac, motion, unknown}
  2. 生成混音矩阵 A ∈ R^{C×K}，服从该伪迹的空间先验：
       harmonic        : 稀疏随机，2-4 个通道非零（模拟参考/局部干扰）
       ocular_drift    : 前额优势 → 按坐标 Fp1/Fp2/F7/F8 位置加权，其余衰减
       myogenic        : 局域（随机选 1 个中心通道，按距离高斯衰减）
       cardiac         : 弥散 + 全通道近同相（低秩，rank 1-2）
       motion_transient: 全通道，幅度逐通道随机
       unknown         : 随机稀疏
  3. 传播延迟 τ_ck ~ U(-8 ms, +8 ms)（保留现有 0.008 s 设定）
  4. y_c(t) = Σ_k A[c,k] · s_k(t − τ_ck) + clean_c(t)
  5. 记录 A 为真值，进样本字典的 "mixing_matrix" 字段，供空间指标评价
```

**验收：** 对 `(B, C=8, T=1000)` 样本，断言 `A` 的形状与稀疏度符合各自先验；断言不同通道的伪迹分量**互不相同**（`artifacts[:, k, 0] != artifacts[:, k, 1]`）。

### 6.6 多通道下的分量监督口径

设计文档 §3.7 允许比较"逐通道分量输出"与"源时程 + 空间投影因子化输出"，但**不得未经消融直接替换**。

**第一版保持逐通道分量输出**（即 `artifacts` 形状 `(6, C, T)`，现状不变）。因子化输出作为 **P3 阶段的消融项**，不是首发版结构。

---

## 7. 阶段划分总览

```
阶段0 地基加固 ──┐
                 ├─→ Gate0 ──→ 阶段1 多通道 ──→ Gate1
阶段1 多通道 ────┘                                │
                                                  ▼
                         阶段2 数据基础设施 ──→ Gate2
                                                  │
                                                  ▼
                         阶段3 训练协议 ────→ Gate3
                                                  │
                                                  ▼
                         阶段4 核心评价 ────→ Gate4
                                                  │
                        ┌─────────────────────────┼──────────────────┐
                        ▼                         ▼                  ▼
                  阶段5 tACS专项          阶段6 真实适配/自监督   阶段7 流式
                        └─────────────────────────┴──────────────────┘
                                                  │
                                                  ▼
                                      Gate5-7 ──→ 完成定义验收（第 9 章）
```

**依赖**：阶段 1–7 严格串行（后一阶段依赖前一阶段的产物）。阶段 5、6、7 之间无依赖，可并行，但都必须在阶段 4 完成后开始。

**最终目标（设计文档 §15 的 10 条完成条件）与阶段对应**：

| §15 条件 | 由哪些阶段达成 |
|---|---|
| 1. 六专家均实现并通过独立单元测试 | 阶段 0–1（补齐空间投影与多通道单测） |
| 2. 五类已知伪迹在严格分组划分下完成单伪迹/复合/未见组合测试 | 阶段 2 + 4 |
| 3. Unknown 完成开放集测试且不主导清洁与已知类 | 阶段 3 + 4 |
| 4. 自适应 Top-r / clean-bypass / 专家专属头 / 掩码监督 / 分阶段训练完成 | 阶段 3（前 4 项已实现，clean-bypass 需重训校准） |
| 5. 至少一个真实外部数据集验证合成到真实迁移，心电与运动不能只靠合成 | 阶段 2（ds004784 + PhysioMotion）+ 阶段 6 |
| 6. 外部基线、内部消融、≥3 随机种子 | 阶段 4 |
| 7. 清洁输入、生理保持、≥1 下游任务无不可接受退化 | 阶段 2 + 4 |
| 8. 含 tACS 主张则须完成专项保真协议 | 阶段 5 |
| 9. 含实时性主张则须完成流式协议与延迟测量 | 阶段 7 |
| 10. 分量解释性表述受监督阶梯与稳定性证据约束 | 阶段 4 |

---

## 8. 阶段详细任务

### 阶段 0：地基加固

**目标**：把目录、路径、登记、环境、基线都固定下来。此阶段不碰模型。

---

#### T0.1　建立路径单一真相源

**产出**：`unicore_eeg/paths.py`、`configs/_paths.yaml`

**步骤**
1. 新建 `unicore_eeg/paths.py`：
   ```python
   PROJECT_ROOT = Path(r"D:\codexwork\unicore")
   DATA_ROOT    = PROJECT_ROOT / "data"
   RAW_ROOT     = DATA_ROOT / "raw"
   RUNS_ROOT    = PROJECT_ROOT / "runs"
   CONFIG_ROOT  = PROJECT_ROOT / "configs"
   EXTERNAL_POOL = Path(r"D:\codexwork\eeg\data")
   ```
2. 读取 `data/local_sources.json`，把 POOL 各数据集相对路径合成绝对路径，暴露为 `POOL["bci2a"]` 等，并提供 `pool_path(name, *parts)` 函数。
3. 提供 `assert_readonly_pool(path)`：任何传给 POOL 的路径若被用于写操作则抛异常。
4. **这是全项目唯一允许出现 `D:\codexwork\eeg` 的文件。**

**验收**
```bash
grep -rn "codexwork.eeg" unicore_eeg scripts tests configs | grep -v "unicore_eeg/paths.py"
```
必须无输出。

---

#### T0.2　ds004784 目录规范化 + 校验清单（决策 D3）

**步骤**
1. 将 `data/raw/on004784` 重命名为 `data/raw/ds004784`。**C 盘与 D 盘为同一物理盘符时用 `Move-Item`，不要跨盘复制 11 GB。**
2. 更新全部引用：`unicore_eeg/on004784.py` → 重命名为 `unicore_eeg/ds004784.py`；类名 `On004784*` → `Ds004784*`；常量 `ON004784_*` → `DS004784_*`；`scripts/sanity_on004784.py` → `scripts/sanity_ds004784.py`；`data/on004784_sanity_report.md` → `data/ds004784_sanity_report.md`；`tests/test_core.py` 中的导入与用例名。
3. 生成 `data/raw/ds004784/download_manifest.json`：遍历 S3 对象列表（复用 `scripts/download_on004784.py::list_objects`，改为写 `ds004784/` 前缀），逐文件记录 `file`/`bytes`/`sha256`/`status`，格式对齐现有 `data/download_manifest.json`。对 ~11 GB 计算 SHA-256 需要时间，**后台运行**。
4. 把 `ds004784` 条目并入 `data/download_manifest.json` 的 `sources` 与 `files`。
5. 在 `data/DATASETS.md` 增加 ds004784 段：DOI `10.18112/openneuro.ds004784.v1.0.4`、许可 CC0、论文 Downey & Ferris, Sensors 2023, 23, 8214、内容（10 脑源 + 2 眼动 + 4 颈肌 + 4 面肌 + 1 trigger，512 Hz，5 min）、用途（体模真值监督 / 清洁负对照 / Unknown 压力测试）、限制（GT 为 300 s 而 BIDS 记录 311–357 s，需 trigger 对齐）。

**验收**
- `python scripts/sanity_ds004784.py` 成功且输出到 `data/ds004784_sanity_report.md`，窗口数与旧报告一致（971）。
- `python -m pytest tests/` 全绿。
- `data/download_manifest.json` 中 `files.ds004784` 非空且每项含 `sha256`。
- `grep -rn -i "on004784" unicore_eeg scripts tests configs data/DATASETS.md` 无输出。

---

#### T0.3　POOL 可达性与 schema 探测（决策 D2）

**产出**：`scripts/probe_pool.py`、`runs/_pool_probe/pool_probe.json`

**步骤**
1. 写探测脚本，对 POOL 下每个数据集：
   - 列出实际文件数与总字节；
   - 对 EDF/GDF/BDF/SET 用 `mne.io.read_raw_*`（`preload=False`）读 `sfreq`、`ch_names`、`n_times`，只读第一个文件；
   - 对 `.npz` 用 `np.load(...).files` 打印键名与各键 shape/dtype（**不要读全部数据**，用 `mmap_mode="r"` 或只读元信息）；
   - 对 `.mat` 用 `scipy.io.loadmat(..., simplify_cells=True)` 打印顶层键。
2. 结果写入 `runs/_pool_probe/pool_probe.json`。
3. 人工核对：把实测的采样率/通道数**回填本手册第 3.1 章所有"待读"单元格**，并在 `data/DATASETS.md` 补一张"实测规格表"。

**验收**
- `pool_probe.json` 覆盖第 3.1 章全部 12 个数据集（tuar 记录为 `empty`）。
- 无任何写入 POOL 的操作（用 T0.1 的 `assert_readonly_pool` 强制）。
- 本手册第 3.1 章无残留"待读"。

---

#### T0.4　建目录骨架与配置基线

**步骤**
1. 新建 `configs/`，写入 `configs/base.yaml`：采样率 500、窗口 1000、带通 0.5–45、CAR 开关、种子列表、划分比例（train/val/test = 0.6/0.2/0.2）。
2. 新建 `configs/multichannel_base.yaml`，在 `base.yaml` 基础上增加 `montage` 段（见 T1.1）。
3. 新建 `runs/.gitkeep`。
4. 更新 `README.md`：环境改为指向 `D:\Anaconda_envs\envs\unicore-eeg`；新增"本手册"链接；新增"POOL 只读"说明。

**验收**：`python -c "import yaml,sys; yaml.safe_load(open(r'configs/base.yaml'))"` 成功；README 中旧路径描述已更正。

---

#### T0.5　复现历史基线（不改代码）

**理由**：必须有一份"改动前"的可比数值，否则后面无法证明多通道改造是否引入退化。

**步骤**
1. 用现有代码在**不动任何源文件**的前提下重跑一次并另存：
   ```bash
   D:\Anaconda_envs\envs\unicore-eeg\python.exe scripts/first_experiment.py \
     --epochs 12 --train-samples 65536 --eval-samples 6000 --batch-size 64 \
     --use-public-sources --seed 42 --out runs/baseline_singlechannel_seed42
   ```
2. 记录 `runs/baseline_singlechannel_seed42/report.md` 的 7 项路由指标。
3. 与 `runs/full_experiment_v2_fixed/report.md` 比对，**差异应在随机种子可解释范围内**；若差异大，说明环境或数据已变，必须先查清再继续。

**验收**：新 run 的 `macro_auroc` 与历史值 0.8988 的绝对差 < 0.05；差异原因写进 `runs/baseline_singlechannel_seed42/NOTES.md`。

---

#### T0.6　清理遗留

**步骤**：删除 `D:\codexwork\unicore\1.5`（0 字节空文件）。确认无其他遗留物。

**验收**：`ls -la` 无 `1.5`。

---

#### T0.7　环境加固与性能基线（利用双 5080 的前置任务）

**目标**：把 0.3 节实测出的能力与约束固化进代码，并建立可复现的硬件基线。**此任务不做，后面的多阶段训练会持续浪费一半的机器。**

**步骤**

1. **补装缺失依赖**
   ```bash
   "D:/Anaconda_envs/envs/unicore-eeg/python.exe" -m pip install pyyaml psutil
   ```
   同步写进 `requirements.txt`、`pyproject.toml`、`environment.yml`。

2. **把硬件基准脚本固化入仓**：新建 `scripts/bench_hardware.py`，合并两类测量并输出 `runs/_env/hardware_baseline.json`：
   - **GPU 吞吐**：`C ∈ {1, 8, 34, 64, 128}` × `batch ∈ {32, 64, 128}`，测 `ms/step`、`samples/s`、分配器峰值；OOM 的组合记录为 `OOM` 而非中断。
   - **DataLoader 吞吐**：`num_workers ∈ {0, 4, 8, 12}`，测 `samples/s` 与测量后剩余物理内存。
   - **必须带 `if __name__ == "__main__":` 守卫**（0.3.7 坑 1），且**测量长度足够**：每组合至少 2000 个样本，否则数值不可用于决策（手册 0.3.7/0.3.8 的数值来自短测，只可用于定方向，不可用于定参数）。

3. **修 `num_workers` 默认值**：`scripts/train.py:25`、`scripts/first_experiment.py:43` 的默认值由 `0` 改为 `8`；两处 `DataLoader` 补 `persistent_workers=True`、`prefetch_factor=4`（仅当 `num_workers>0` 时传，否则 PyTorch 报错）。

4. **修 DataParallel 缺陷**：`scripts/train.py:108`、`scripts/first_experiment.py:115` 的 `loss.backward()` 改为 `loss.sum().backward()`，否则 `--data-parallel` 路径必然报 `grad can be implicitly created only for scalar outputs`。改动后必须单卡回归确认指标不变。

5. **双卡并行调度**：新建 `scripts/run_dual_gpu.py`（或 `scripts/run_dual_gpu.ps1`），输入"任务列表 + 配置文件"，自动把任务轮流分派到 `CUDA_VISIBLE_DEVICES=0/1` 两个进程，并：
   - 每进程 `num_workers=8`，总 worker ≤ 16（0.3.7 的 RAM 约束）；
   - 进程结束或异常时**主动回收子 worker**（0.3.7 坑 2）；
   - 打印每卡的实际利用率与耗时。

6. **配置化**：把 0.3.7 的 `dataloader` 段写进 `configs/base.yaml`。

7. **在 `README.md` 增加"本机环境"小节**，指向 0.3 节的实测数据。

**验收**
- `pip show pyyaml psutil` 成功；三处依赖文件均含这两项。
- `runs/_env/hardware_baseline.json` 存在，含 GPU 与 DataLoader 两组数据，且每组测量样本数 ≥ 2000。
- `scripts/run_dual_gpu.py` 能同时跑两个短实验，`nvidia-smi` 显示两卡均有占用，总 worker 数 ≤ 16。
- 单卡回归：`num_workers=8` 下的 `macro_auroc` 与 T0.5 基线（`num_workers=0`）差 < 0.01，且**墙钟时间明显下降**（记录于 `runs/_env/NOTES.md`）。
- 异常中断后无残留 python 进程（用 0.3.9 的自检命令确认）。

---

### Gate 0（硬门禁）

全部满足方可进入阶段 1：

- [ ] `grep -rn "codexwork.eeg" unicore_eeg scripts tests configs | grep -v paths.py` 无输出
- [ ] `python -m pytest tests/` 全绿
- [ ] `data/download_manifest.json` 含 ds004784 且带 sha256
- [ ] `runs/_pool_probe/pool_probe.json` 覆盖 12 个数据集，手册无"待读"
- [ ] `runs/baseline_singlechannel_seed42/report.md` 存在且指标与历史基线一致
- [ ] 无 `on004784` 残留命名
- [ ] **T0.7 完成**：`pyyaml`/`psutil` 已装并写入三处依赖文件；`runs/_env/hardware_baseline.json` 存在；`num_workers` 默认值已改为 8；DataParallel 的 `loss.backward()` 已修为 `loss.sum().backward()`；双卡调度脚本可同时跑两个实验

---

### 阶段 1：多通道能力落地（决策 D1）

**目标**：模型具备原生多通道能力，合成数据具备真空间混合，单通道作为退化特例仍可跑通。

---

#### T1.1　montage 与坐标表

**产出**：`unicore_eeg/montage.py`、`configs/montages/*.yaml`

**步骤**
1. 实现 10-20 标准坐标表（`Fp1 Fp2 F7 F8 F3 F4 Fz C3 C4 Cz T3 T4 T5 T6 P3 P4 Pz O1 O2` 及扩展导联 `FC1 FC2 FC3 FC4 C1 C2 C5 C6 CP1 CP2 CP3 CP4 P1 P2 POz`），单位归一化为头部半径 1。
2. 提供 `resolve_montage(channel_names) -> (coords, mask)`：把实际通道名映射到坐标；无法映射的通道 `mask=False` 且坐标置零。
3. 复用 POOL 旧项目 `constants.py` 的 `RAW_ALIASES` 思路（19 导别名 → 标准名），**在 CODE 下重新实现，不要 import**。
4. 为各数据集写 montage 配置：
   - `bci2a.yaml`：22 导（Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz）
   - `bci2b.yaml`：3 双极（C3 Cz C4）
   - `physiomotion.yaml`：34 双极（照抄旧项目 `PHYSIOMOTION_BIPOLAR_CHANNELS` 的 34 个名称，坐标用两端电极中点近似）
   - `ds004784.yaml`：128 导；提供一个 `ds004784_19ch.yaml` 子集（标准 19 导），用于与其它数据集比对
   - 其余数据集：T0.3 探测后补
5. 双极导联的坐标用两端电极坐标的中点，并在配置中标注 `bipolar: true`。

**验收**
- 单测：对每个 montage 配置，`resolve_montage` 返回的通道数与配置一致，`mask` 全 True。
- 单测：给一个不存在的通道名，`mask` 对应位为 False 且不抛异常（走 T5.1 的缺导报错策略时另论）。

---

#### T1.2　观测抽取器多通道化

**目标**：消除 6.2 节的三处压维。

**步骤**（改 `unicore_eeg/model.py` 的 `ArtifactObservationExtractor`）
1. `_harmonic_basis`：
   - 频率估计：`spectrum = rfft(y).abs().mean(dim=1)`（共享找峰，保留）→ `peak_frequency: (B,)`
   - 拟合：对每个通道独立 lstsq。实现上把 `design` 广播到 `(B, C, T, 6)` 与 `y.transpose` 一起解，或退化为对 `C` 的循环（C ≤ 128 时可接受，但必须测速）。
   - 返回 `(B, C, T)`，**不再 `expand_as`**。
2. 自相关与 cardiac：
   - `autocorrelation` 逐通道：`irfft(rfft(y).abs().square())` → `(B, C, T)`
   - `cardiac_lags: (B, C)`，逐通道 argmax
   - `periodic` 逐通道构造 → `(B, C, T)`
3. `unexplained = y - slow - high` 保持逐通道。
4. **保持返回 6 元素的顺序不变**（harmonic, slow/ocular, high/myogenic, pulse/cardiac, motion, unexplained）。
5. 文档字符串写明：输入 `(B, C, T)`，输出 `list[Tensor]` 长度 6，每个 `(B, C, T)`。

**验收**
- 单测：`C=1` 与 `C=4` 均前向成功，6 个输出 shape 均为 `(B, C, T)`。
- 单测：`C=4` 时 `not torch.allclose(obs[0][:,0], obs[0][:,1])`。
- 单测：`C=4` 时 `cardiac_lags[:,0]` 与 `cardiac_lags[:,1]` 不完全相同（构造一个左通道 0.8 s 周期、右通道 1.2 s 周期的合成输入来断言）。
- 单通道回归：T0.5 的基线在 T1.2 完成后重跑（`runs/after_obs_multichannel_seed42`），`macro_auroc` 与基线差 < 0.03，否则说明改造引入了行为变化，需排查。

---

#### T1.3　新增空间投影模块

**产出**：`model.py::SpatialProjectionHead`

**步骤**：按 6.4 节实现。要点重申：
- 禁止任何形状依赖 C 的 `nn.Linear`；
- `has_coords=False` 时坐标通道置零，模块仍可前向；
- 空间权重只调制 cardiac 与 ocular_drift 两个观测；
- 新配置项：`use_spatial: bool = True`、`spatial_pca_rank: int = 4`、`spatial_modulation_alpha_scale: float = 1.0`。

**接入 `UniCOREEG.forward`**：在 `observations = self.observations(y_norm)` 之后、专家调用之前，计算 `spatial_weights`，对索引 1（ocular）与 3（cardiac）的观测做 `obs * (1 + α·w_s)`。

**验收**
- 单测：`C ∈ {1, 2, 3, 34, 64, 128}` 前向 + 反向均成功，无 NaN。
- 单测：`count_parameters` 在 `C=2` 与 `C=128` 下**完全相同**（证明参数量与 C 无关）。
- 单测：`use_spatial=False` 时输出与未加该模块时一致（等价性回归）。

---

#### T1.4　合成器真混音改造

**产出**：`unicore_eeg/spatial_mixer.py`（新建）、`synthetic.py` 改造

**步骤**
1. 实现 `SpatialMixer`（6.5 节），提供 `build_mixing_matrix(kind, C, coords, generator) -> Tensor[C]`。
2. `SyntheticEEGDataset` 增加参数 `channels: int`（已有）与 `montage_coords: Tensor | None`，并：
   - 用 `SpatialMixer` 替换 `_expand_channels`；
   - 样本字典新增 `"mixing_matrix": (6, C)`；
   - `"artifacts"` 形状从 `(6, C, T)` **保持不变**；
   - 新增 `"coords": (C, 2)`。
3. `_clean` 的多通道生成：清洁源同样要经空间混合（不同通道的 alpha 节律应有相位/幅度差异），不能是同一源复制。
4. 保留 `_scale_to_snr` 的 SNR 逻辑，但作用在**混音后的多通道伪迹**上。

**验收**
- 单测：`C=8` 时 `mixing_matrix` 中 ocular 行在 Fp1/Fp2 类通道的权重显著高于枕区（构造坐标后断言）。
- 单测：`C=8` 时 `artifacts[:, 1, 0]` 与 `artifacts[:, 1, 1]` 不相等。
- 单测：cardiac 的 `mixing_matrix` 行近似低秩（各通道符号一致，`torch.corrcoef` 均值 > 0.8）。
- 单通道：`C=1` 时退化为现有行为，T0.5 基线在该改造后仍可复现（`runs/after_mixer_seed42`，差 < 0.03）。

---

#### T1.5　多通道冒烟与单通道回归

**步骤**
1. 新建 `scripts/smoke_multichannel.py`：对 `C ∈ {1, 3, 8, 34, 64, 128}` 各构造一个 batch，跑前向 + 损失 + 反向一步，打印参数量、显存峰值、单步耗时。
2. 新建 `scripts/regress_multichannel.py`：跑 T0.5 与 T1.2/T1.4 的对照，输出 `runs/_regress/regress_multichannel.md`，列出"改造前 / 改造后"的单通道指标差。
3. 用 `configs/multichannel_base.yaml` 起一个 `C=8` 的小规模合成训练（`--train-samples 2048 --epochs 2`），确认能跑完不崩。

**验收**
- `smoke_multichannel.py` 六种 C 全部成功。
- `regress_multichannel.md` 中单通道指标差 < 0.03（允许统计噪声）。
- `runs/smoke_multichannel_c8/` 训练完成，检查点 finite。

---

### Gate 1

- [ ] 观测抽取器返回 6 个 `(B,C,T)` 张量，多通道单测全绿
- [ ] `SpatialProjectionHead` 参数量与 C 无关
- [ ] 合成器混音矩阵具备各伪迹的空间先验，多通道分量互不相同
- [ ] 单通道回归：`macro_auroc` 相对 T0.5 基线差 < 0.03
- [ ] `C ∈ {1,3,8,34,64,128}` 冒烟全部通过
- [ ] `pytest tests/` 全绿
- [ ] `runs/smoke_multichannel_c8/` 完成训练

**若 Gate 1 失败**：不得进入阶段 2。多通道是用户硬要求，必须在此处解决。

---

### 阶段 2：数据基础设施（设计文档 P1）

**目标**：建立统一、无泄漏、可复现的数据管线，并把真实数据集接进来。

---

#### T2.1　预处理与样本记录统一（P1: 统一样本记录和字段有效性掩码）

**产出**：`unicore_eeg/preprocess.py`、`unicore_eeg/records.py`

**步骤**
1. 在 `preprocess.py` 实现（参考第 4 章旧项目 `preprocess_continuous`，重写）：
   ```
   preprocess_continuous(x, src_sfreq, tgt_sfreq=500, bandpass=(0.5,45.0),
                         car=True, robust_clip=8.0) -> np.ndarray
   ```
   顺序固定：CAR → `resample_poly` → `filtfilt` 带通 → robust MAD 裁剪。
2. 在 `records.py` 定义统一样本记录 dataclass，字段严格照设计文档 §4.1：
   `noisy_eeg, clean_eeg, artifact_sources, artifact_labels, label_mask, severity, dataset_id, subject_id, recording_id, session_id, channel_info, sampling_rate, metadata, event_annotations`
3. 提供 `to_batch(record) -> dict`，输出键名与 `losses.py` 期望完全一致（`noisy/clean/artifacts/labels/label_mask/component_mask/severity/is_clean/disabled_experts/metadata`）。**加单测：用 `to_batch` 的产物直接过一遍 `UniCORELoss`，不报错。**

**验收**：`to_batch` 产物可直接喂 `UniCORELoss`；带通前后频谱单测通过（45 Hz 以上衰减 > 40 dB）。

---

#### T2.2　受试者/记录级分组器（P1: 先划分后切窗）

**产出**：`unicore_eeg/splits.py`

**步骤**
1. 实现两个函数，参考第 4 章旧项目 `manifest.add_recording_split` / `add_recording_kfold_split`（重写）：
   - `assign_group_split(records, ratios=(0.6,0.2,0.2), seed, group_key="subject_id")`
   - `assign_group_kfold(records, fold, folds, seed, group_key="subject_id")`
2. **关键约束**：划分函数在**切窗之前**调用；函数签名接受"记录表"而非"窗口表"，从类型上防止误用。
3. 当 `subject_id` 缺失时（如 EEGdenoiseNet），回退到 `recording_id`，并在返回的划分表上标记 `split_level="recording"`。
4. 输出 `runs/<stage>/splits.json`，记录每个 group 落在哪个集合。

**验收**
- 单测：构造 10 个受试者 × 5 记录的表，划分后**同一受试者的所有记录必在同一集合**。
- 单测：三种子下划分比例与设定值偏差 < 2%。
- 单测：`split_level` 在缺 `subject_id` 时正确标记为 `recording`。

---

#### T2.3　泄漏检查器（P1: 防止清洁片段/伪迹源/派生视图跨集合）

**产出**：`scripts/check_leakage.py`、`unicore_eeg/leakage.py`

**步骤**
1. 实现四类检查，全部输出计数，**目标值均为 0**：
   - **窗口重叠**：跨集合的窗口在时间上重叠（参考旧项目 `_overlap_exclusion` 的区间比较逻辑）。报告必须为 0。
   - **清洁片段复用**：同一清洁源片段被用于不同集合的污染样本。检查 `source_segment_id` 字段。
   - **伪迹源复用**：同上，检查 `artifact_source_id`。
   - **受试者越界**：同一 `subject_id` 出现在两个集合。
2. 检查器接入所有训练脚本的启动流程，**检查不通过直接 `raise`**，不给出"仅警告"的选项。
3. 输出 `runs/<stage>/leakage_report.json`。

**验收**
- 单测：故意构造一个重叠窗口的划分，检查器必须报出非零并 raise。
- 单测：`synthetic.py` 与各真实 loader 的现有划分在检查器下**全部为 0**。
- 若发现现有合成划分存在"清洁片段跨集合复用"，必须修掉（当前 `synthetic.py:59-60` 用索引 80/20 边界切分源池，理论上无重叠，但需检查器确认）。

---

#### T2.4　合成器增强（P1: 频率漂移、传播延迟、非线性饱和、局部事件掩码）

**步骤**（改 `synthetic.py`，全部受开关控制，默认关闭以便回归）
1. **频率漂移**：谐波源于现有 `drift` 参数上增强，加入分段线性/正弦调频。
2. **传播延迟**：已在 `SpatialMixer` 中实现（T1.4 的 `τ_ck`），此处扩展为**频率相关**延迟（不同频带不同群延迟），模拟容积传导。
3. **非线性饱和**：对 `y` 施加 `clip` 或 `tanh` 软削顶，幅度阈值随机；记录被削顶样本比例。
4. **局部事件掩码**：模拟电极接触不良，随机选 1–2 通道、随机时段置零或大幅衰减，**并同步把该通道该段的 `component_mask` 置 0**（不能当作伪迹真值）。
5. 每项增强都写入样本字典的可追溯字段（如 `"augmentation_flags"`）。

**验收**
- 单测：每项增强开启后，样本统计量按预期变化（如削顶后 `max(|y|)` 受限）。
- 单测：局部事件掩码后对应位置 `component_mask == 0`。
- 单测：全部关闭时，输出与增强前逐元素相等。

---

#### T2.5　ds004784 真值接入（关键任务）

**产出**：`unicore_eeg/ds004784.py` 扩展、`scripts/build_ds004784_pairs.py`

**为什么关键**：这是全项目唯一带**物理真值分量**的数据（10 脑源 + 2 眼动 + 4 颈肌 + 4 面肌 + 1 trigger，512 Hz，5 min）。它直接支撑设计文档 §5.2「仅对具有可信分量标签的类别监督」、§15 第 5 条前半句、以及 §7 的分量解释力诊断。

**步骤**

**Step 1 — trigger 对齐（必做，且必须验证）**
- GT：`data\raw\ds004784\stimuli\GTdata_croppedToRisingEdge.mat`，形状 `(153600, 21)` @512 Hz = 300 s。
- BIDS：各任务 182784 / 162304 / 160256 / 163840 / 168448 / 159232 点 @512 Hz = 357/317/313/320/329/311 s。
- **要求**：用 GT 第 21 路（trigger）与 BIDS `sub-001_task-*_events.tsv` 的 `x65471` 事件标定偏移。**必须写成可复跑脚本，并把对齐残差（最大偏差、样本数）写进报告。**
- **禁止**假设 GT 从 0 开始对齐。先在 `Brain` 条件上验证：GT 的 10 路脑源投影后应与 BIDS 记录的脑区通道高度相关（报告相关系数）。
- 若对齐失败（相关 < 0.5），写 `BLOCKED.md` 并停止，不要硬凑。

**Step 2 — 构造监督对**
- 从 BIDS 取原生 128 导（或 `ds004784_19ch` 子集，由配置决定）。
- 由 GT 构造：
  - `clean_eeg` = GT[0:10]（脑源）经混音后投影到所指导联
  - `artifact_sources[ocular_drift]` = GT[10:12] 投影
  - `artifact_sources[myogenic]` = GT[12:20] 合并投影（颈肌 + 面肌），**同时保留 `family_subtype` 区分 neck/facial 以便分层统计**
  - `artifact_labels` = 按任务条件映射（Brain→全 0；Eyes→ocular=1；Facial/Neck→myogenic=1；Walking→motion=1；All→ocular+myogenic+motion=1）
  - `label_mask` = 全 1
  - `component_mask` = 有真值的三类为 1，harmonic/cardiac/unknown 为 0（**没有真值就不能监督**）
- 混音矩阵：从 `derivatives\Scripts\Compare_Ground_Truth\` 下的脚本与 `derivatives\Data\Imported\*.set` 推断，或直接在 GT 与 BIDS 记录之间做最小二乘估计（GT 前 20 路 → 128 导），把估计出的矩阵存盘并报告拟合残差。

**Step 3 — 三种用法**
1. **分量监督训练**：`component_mask` 只开三类，参与 `L_artifact`。
2. **清洁负对照**：`Brain` 条件视为 clean 输入，评价 bypass 率、修改率、频带保持（§9.1 Clean-input）。
3. **Unknown 压力测试**：`All` 条件（三类同时存在）作为复合伪迹样本；额外把 `harmonic` 正弦注入 GT 脑源，构造"已知 + 注入未知机制"样本。

**Step 4 — 划分**
只有一个受试者（`sub-001`）。**必须按 `recording_id`（即 6 个任务记录）划分，不得随机切窗划分。** 推荐：Brain/Walking 做 val，Eyes/Facial 做 test，Neck/All 做 train；或做留一任务交叉验证（6 折）。**划分方案必须写进配置并预注册，看结果前定死。**

**验收**
- `scripts/build_ds004784_pairs.py` 可复跑，产出 `runs/ds004784_pairs/`：对齐报告、混音矩阵、拟合残差、样本数统计。
- 对齐残差报告存在，且 `Brain` 条件下 GT 脑源投影与 BIDS 记录相关 > 0.5。
- 泄漏检查器（T2.3）在该划分上输出 0。
- 样本字典能通过 `to_batch` 并喂进 `UniCORELoss`。
- `component_mask` 统计写入报告：三类有真值，三类无真值。

---

#### T2.6　PhysioMotion 只读接入（走 POOL）

**步骤**
1. 改写 `unicore_eeg/physiomotion.py`：路径来自 `paths.pool_path("physiomotion", ...)`，**删除硬编码的 `D:/codexwork/eeg/data/physiomotion`**。
2. **统一接口**：`PhysioMotionWindowDataset.__getitem__` 的返回必须与 `Ds004784WindowDataset` 对齐——加上 `"coords"`、`"metadata"`、`"mixing_matrix"`（此处为 `None`），并**不再在数据集层做归一化**（交给模型内部 `robust_normalize`），与 T2.1 的约定一致。
3. 确认标注路径：`derivatives\Manual_Annotations\sub{id}_run*.csv`，字段 `channel/start_time/stop_time/label`。
4. `map_annotation` 扩展：把 `PHYSIOMOTION` 的标注名完整枚举（从 180 份 CSV 的 `label` 列去重得到），**打印完整标签列表并写进报告**，确认无遗漏映射。当前映射只覆盖 blink/eyem/tongue/swallow/chew/eyebrow/headm，需核实是否完整。
5. 用 `--subjects 1..30` 全量建索引，报告总窗口数、各 family 样本数。

**验收**
- `grep -rn "codexwork.eeg" unicore_eeg/physiomotion.py` 无输出。
- 标签去重列表写入 `runs/physiomotion_labels/labels.md`，未映射标签数为 0。
- 泄漏检查器在划分上输出 0。

---

#### T2.7　外部池下游与负对照 loader（P1 剩余项）

**步骤**（按依赖顺序，每个数据集一个任务）
1. **bci2a**（优先，完整且下游最成熟）：参考旧项目 `data/bci2a.py`，在 CODE 下重写 `unicore_eeg/pool/bci2a.py`。要点：GDF 读取、22 导映射、`.mat` 取 `ArtifactSelection`/`classlabel`、事件码（768/769-772/1023）、7.5 s 试次。用途：MI 下游 + Clean-input。
2. **chbmit**：`unicore_eeg/pool/chbmit.py`，用途为**生理负对照**——棘波/发作形态**不得作为待删除目标**。报告必须写明"使用 N 个记录子集"。
3. **sleep_edfx / cap_sleep**：`unicore_eeg/pool/sleep.py`，用途为睡眠事件负对照（K 复合波、纺锤波）。**先核验事件标签质量**，标注不可靠的记录排除并记录原因。
4. **openbmi / physionet_mi / bci2b / cho_gigadb**：`unicore_eeg/pool/mi.py` 统一 MI 下游 loader。
5. **ds002094**：`unicore_eeg/pool/ds002094.py`，用途为 TMS 强瞬态压力测试 + Unknown 开放集。注意 TMS 脉冲与神经响应重叠，**不能当作简单加性真值**。
6. **faced**：`unicore_eeg/pool/faced.py`，用途为情感下游 + Clean-input。**预处理版本与原始 BDF 必须分开报告。**
7. 每个 loader 都统一走 `preprocess_continuous` + `assign_group_split` + `check_leakage` 三步，并用 `--dry-run` 产出 `runs/_pool_probe/<name>_index.json`。

**验收**
- 每个 loader：`--dry-run` 成功，输出实际记录数/受试者数/采样率/通道数/总窗口数。
- 每个 loader：泄漏检查 0。
- 每个 loader：在报告里写明是"全集"还是"N 个记录子集"。
- `tuar`：明确记录为不可用，不写 loader。

---

### Gate 2

- [ ] `preprocess_continuous` 单测通过（含 45 Hz 抗混叠断言）
- [ ] 分组器单测通过（同受试者不跨集合）
- [ ] 泄漏检查器在**所有**数据集划分上输出 0，且接入训练脚本启动流程
- [ ] 合成器四项增强（漂移/延迟/饱和/掩码）单测通过，全关时与增强前逐元素相等
- [ ] `ds004784` 对齐报告存在且相关 > 0.5；`component_mask` 统计正确（3 真 3 无）
- [ ] PhysioMotion 无硬编码路径，标签映射完整
- [ ] 至少 `bci2a`、`chbmit`、`sleep_edfx` 三个 loader 的 `--dry-run` 成功
- [ ] `data/DATASETS.md` 已补"实测规格表"，子集情况全部注明
- [ ] `pytest tests/` 全绿

---

### 阶段 3：训练协议（设计文档 P2）

**目标**：按 §6 五阶段协议重训，补齐保持损失、综合选模、多种子。

---

#### T3.1　频带与事件保持损失接口（P2: 增加频带与事件保持损失的可配置接口）

**产出**：`unicore_eeg/losses.py` 新增 `PreservationLoss`

**步骤**
1. 实现三类保持项，全部**可选、可配置权重、默认关闭**：
   - **频带功率保持**：Delta(1-4) / Theta(4-8) / Alpha(8-13) / Beta(13-30) / Gamma(30-45) 的功率比。用 Welch 或 `rfft` 带内积分。
   - **ERP/事件形态保持**：输入事件标注 `event_annotations`（onset, 类型），在事件窗口内比较幅值与潜伏期。**仅在标注可靠时启用。**
   - **刺激锁相保持**：需要刺激频率，用于 tACS（阶段 5）。
2. 未定义保真目标的类别**不得**施加保持损失（设计文档 §5.4："不能把所有固定频段能量都当作应保留真值"）。
3. 权重加入 `UniCORELossWeights`，并在 `configs` 中暴露。

**验收**
- 单测：权重为 0 时损失与不加该模块完全一致。
- 单测：构造一个只有 Alpha 的输入，Alpha 保持项在 Alpha 被削时上升、其余项不变。
- 单测：无 `event_annotations` 时 ERP 项返回 0 且不报错。

---

#### T3.2　阶段 A：令牌与专家预训练（§6 阶段 A）

**产出**：`scripts/train_stage_a.py`、`configs/stage_a.yaml`

**步骤**
1. 数据组成：清洁 + 五类单伪迹 + 留一类开放集异常 + 有分量标签的合成样本 + **ds004784 真值样本**（三类分量有真值）。
2. 留一类 episode：屏蔽对应标签（`label_mask[family]=0`）并禁用对应已知专家（`disabled_experts[family]=True`）。**`synthetic.py` 已有此逻辑，ds004784 需复用。**
3. Unknown 训练批次必须**同时包含**已知类别、清洁信号和真正的留出异常（防止 Unknown 变成默认残差桶）。
4. 评价：只看路由识别与分量重建，不看最终论文指标。
5. 产出 checkpoint 到 `runs/stage_a_seed{42,1234,20260922}/`。

**验收**：三个阶段子 checkpoint 均存在且 finite；留一 episode 的 `disabled_experts` 统计正确；Unknown 批次构成统计写入报告（三类样本比例）。

---

#### T3.3　阶段 B：统一粗分解（§6 阶段 B）

**步骤**
1. 从阶段 A checkpoint 出发，加入二至五种已知伪迹组合 + "已知 + 未知"组合。
2. 批次**同时包含**清洁、单伪迹、复合伪迹、开放集异常。
3. 训练内容流、六专家、自适应路由、粗分解器。
4. `r_max` 初始 4，用**含三类及以上复合伪迹**的验证集调整（这是设计文档 §3.5 的要求）。

**验收**：`r_max` 调优曲线写入报告，含"固定 Top-2 / 固定 Top-4 / 自适应"三者对比。

---

#### T3.4　阶段 C：冻结粗分解器训练残差（§6 阶段 C）

**步骤**：冻结令牌、专家、粗分解器，只训练 `ResidualRefiner` 学 `x - x̂_c`。冻结策略已有（`first_experiment.py:56-66`），沿用并抽成 `scripts/train_common.py` 供三阶段共用。

**验收**：训练日志证明粗分解器参数 `requires_grad=False` 且梯度为 None（打印断言）。

---

#### T3.5　阶段 D：低学习率联合微调（§6 阶段 D）

**步骤**
1. 以阶段 C 学习率的 0.1–0.2 倍解冻联合训练。
2. **模型选择改用 T3.6 的综合验证指标**，不以单一 MSE 选模。

**验收**：`best.pt` 的选择依据可追溯到综合指标，`NOTES.md` 中列出每轮的组合得分。

---

#### T3.6　综合验证选模（P2: 避免单一 MSE）

**产出**：`unicore_eeg/selection.py`

**步骤**
1. 定义预注册的组合得分（看结果前定死，写进 `configs/selection.yaml`）：
   ```
   score = w1·(-val_RRMSE) + w2·val_band_preservation + w3·val_macro_F1
         + w4·val_clean_modified_rate(-) + w5·val_unknown_false_alarm(-)
   ```
2. 每个权重必须有书面理由，且**权重不随结果调整**（若调整，必须记录调整时间与原因，并说明调整前的结果）。
3. 应用到阶段 B/C/D 的 checkpoint 选择。

**验收**：`selection.yaml` 的预注册时间戳早于首个训练结果时间戳；选模过程可复现。

---

#### T3.7　≥3 随机种子完整训练（P2）

**步骤**
1. 用 `42 / 1234 / 20260922` 三个种子，从阶段 A 跑到阶段 D，各产出一套 checkpoint。
2. 报告每个种子的全部指标，并给出**均值 ± 标准差**与**逐种子明细**（设计文档 §9.2 要求"同时报告平均结果、受试者级分布、改善案例和劣化案例"）。

**验收**：`runs/final_seed{42,1234,20260922}/` 三套完整 checkpoint；`runs/seed_summary.md` 含 mean±std。

---

### Gate 3

- [ ] `PreservationLoss` 三子项单测通过，默认关闭时无影响
- [ ] 阶段 A–D 各阶段可独立复跑，冻结策略有断言
- [ ] `r_max` 调优曲线含固定 Top-2/Top-4/自适应对比
- [ ] `selection.yaml` 预注册时间戳早于训练结果
- [ ] 三个种子完整跑完，`seed_summary.md` 含 mean±std
- [ ] 训练数据构成中 ds004784 真值样本已占比（报告给出比例）
- [ ] `pytest tests/` 全绿

---

### 阶段 4：核心评价（设计文档 P3）

**目标**：把 §9 的评价协议完整实现。

---

#### T4.1　八类划分场景（P3: Seen / Unseen-* / Cross-dataset / Clean / Streaming / Negative-control）

**步骤**：实现 `unicore_eeg/eval/scenarios.py`，产出八类测试集：
1. `seen_artifact`：类型相同，受试者与记录独立。
2. `unseen_combination`：单类见过，特定组合只在测试出现。
3. `unseen_severity`：测试强度超出训练中心范围。
4. `unseen_artifact`：完整保留一种伪迹家族，只在开放集出现。
5. `cross_dataset`：完整外部数据集留出（用 ds004784 + PhysioMotion + bci2a）。
6. `clean_input`：无伪迹输入（ds004784 的 Brain 条件 + bci2a 静息段）。
7. `streaming`：连续记录滚动推理（阶段 7 产出）。
8. `negative_control`：不应删除的生理事件（chbmit 棘波、sleep_edfx 纺锤波/K 复合波）。

**验收**：八类测试集各自产出 `runs/eval/<scenario>/index.json`，含样本数、来源数据集、受试者数；泄漏检查 0。

---

#### T4.2　指标组（P3: 信号、时频、频带、路由、清洁输入指标）

**步骤**：实现 `unicore_eeg/eval/metrics.py`，覆盖 §9.2 全部五组：
- **信号恢复**：RRMSE、时域相关、频域相关、多分辨率时频误差、SNR 改善、伪迹残余比。
- **生理保持**：Delta–Gamma 频带功率、ERP 幅值与潜伏期、Alpha 保持、刺激锁相响应、事件形态保持率。
- **路由与清洁输入**：macro-F1、AUROC、校准误差（ECE）、严重程度相关、clean-bypass 比例、清洁修改率、身份门分布。
- **分量诊断**：跨种子稳定性、槽位干预、类别条件专属性、分量间相似度、真实数据支持性。
- **效率**：参数量、FLOPs、显存、batch=1 推理时间、P50/P95 延迟（阶段 7）。

**验收**：每个指标有单测（用构造的平值/完美值输入断言边界）。

---

#### T4.3　Pareto 分析（P3: 去除率与失真度帕累托分析）

**步骤**：实现 `unicore_eeg/eval/pareto.py`。对每个伪迹类别，扫描"去除强度"（可通过 `probability_threshold` 或 `r_max` 或后处理缩放），画出**去除率 vs 失真度**曲线，标出 Pareto 前沿，并与基线前沿对比。

**验收**：`runs/eval/pareto/<artifact>.png` 与 `.csv` 产出；前沿点单调性检查通过。

---

#### T4.4　槽位干预与跨种子匹配（P3）

**步骤**
1. **专家槽位干预**（§7.1）：槽位 = 专家主体 + 对应输出头 + 槽位专属令牌/FiLM 参数。**必须整体交换**（只换一部分会破坏接口配对，不能作为语义证据）。在保持目标类别索引不变的前提下交换完整槽位，按伪迹类别报告性能变化。负对照：随机重新初始化单个完整槽位。
2. **跨种子分量匹配**（§7.2）：先做专家匹配 + 符号与尺度对齐，再比较预测分量相似度。**必须同时报告匹配置换本身**；对固定类型专家，非恒等匹配提示语义不稳定。
3. 输出"伪迹类别 × 专家"激活矩阵、对角优势、类别条件困惑度。

**验收**：干预实验的正/负对照结果均产出；跨种子匹配报告含置换分布。

---

#### T4.5　外部基线（P3: 按任务可比性预注册）

**步骤**
1. **先预注册**再跑：按 §10.1 的六类候选，为每个子任务选择可运行且可公平比较的方法，写进 `configs/baselines.yaml` 并记时间戳。
2. 候选类别：① 传统多通道流程（ASR、ICA + 自动成分标注）；② 伪迹专项方法（tACS 回归、参考通道法）；③ 公开深度去噪模型；④ 参数量匹配的共享单编码器；⑤ 多个专用模型串联；⑥ 内部基线（无条件混训、人工条件、自主条件）。
3. **单通道时 ASR/ICA 标记为"不适用"**，不得把"无法运行"当作性能优势。
4. 外部模型只有在输入、预处理、训练数据、评估协议可对齐时才进入定量排名。

**验收**：`baselines.yaml` 时间戳早于基线结果；每个基线报告"可运行/不适用"及理由；参数量对齐表存在。

---

#### T4.6　下游任务与受试者级统计（P3）

**步骤**
1. **至少一个**下游任务。推荐 **bci2a 运动想象**（完整、且旧项目 `downstream/bci2a.py` + `fbcsp.py` 有可参考实现）。
2. 流程：`去伪迹（500 Hz/2 s）→ 按任务惯例重分段（MI 用 4 s 试次）→ 特征（FBCSP 或简单 CSP+分类器）→ 受试者内交叉验证`。
3. 报告：受试者级置信区间、配对检验（去伪迹前 vs 后）、效应量。
4. **必须报告劣化案例**，不能只报平均改善。

**验收**：`runs/eval/downstream_bci2a/` 含逐受试者结果、置信区间、配对检验 p 值与效应量；报告显式声明采样率与分段差异。

---

#### T4.7　生理负对照（P3）

**步骤**
1. 用 chbmit（棘波/发作形态）、sleep_edfx 与 cap_sleep（K 复合波、纺锤波）测量**保持率**。
2. **先核验事件标注质量**，不可靠的记录排除并记录原因（设计文档 §9.3："不能仅凭数据集名称推定具备所需事件标签"）。
3. 病理信号（棘波、发作）**禁止作为待删除目标**。

**验收**：保持率报告含"标注质量核验"章节，列出纳入/排除的记录数与理由；子集情况已声明。

---

### Gate 4

- [ ] 八类划分场景全部产出且有索引
- [ ] 五组指标全部实现且有单测
- [ ] Pareto 前沿产出
- [ ] 槽位干预正/负对照完成；跨种子匹配含置换分布
- [ ] 基线预注册时间戳早于结果；不适用项已标注
- [ ] ≥1 下游任务完成，含受试者级统计与劣化案例
- [ ] 生理负对照完成，含标注质量核验
- [ ] `pytest tests/` 全绿

---

### 阶段 5：tACS 专项（设计文档 P4）

**先决判断（执行前必读）**

POOL 与 DATA 中**没有任何真实 tACS 数据**。现有相关资源只有 `ds002094`（TMS-EEG），而 TMS 与 tACS 的伪迹机制不同（脉冲 vs 窄带周期）。因此 P4 的 6 项中，**只有一部分可用体模注入执行**：

| P4 条目 | 可执行性 | 执行方式 |
|---|---|---|
| 有/无刺激频率先验对照 | **可执行** | 用 ds004784 注入已知频率正弦，metadata 提供/不提供频率两路对比 |
| 谐波检测与同频响应恢复分开评价 | **可执行** | 注入信号有真值，可分离"检测到基频"与"同频响应失真" |
| 接入 sham、校准记录或体模中**至少两类**证据 | **部分可执行** | 体模（ds004784）✓；sham/校准记录 ✗（**需用户提供或采购**） |
| 校准参考通道的逐通道逐谐波幅相传递 | **不可执行** | 需真实 tACS 记录与参考通道 |
| 报告刺激锁相响应保持与失败案例 | **部分可执行** | 可在体模上做 |
| 不将 tACS 结果表述为已证明神经响应恢复 | **可执行（表述纪律）** | 强制 |

**执行要求**：先写 `BLOCKED.md` 列出缺失数据（真实 tACS 记录、sham、校准参考），向用户申请；**在体模注入可执行的部分照常推进**，但报告标题必须写"体模注入实验"，**不得写"tACS 有效性验证"**。

**步骤**
1. 实现 `unicore_eeg/eval/tacs.py`：向 ds004784 的 Brain 条件注入基频 f0 ∈ {10, 20, 40} Hz 及 1–3 次谐波，幅值按 SNR 阶梯。
2. 两路对照：metadata 提供 f0 vs 不提供 f0（**注意：T1.3 之后的 metadata 通道需先真正实现刺激频率字段，见 T5.1**）。
3. 两个专家对照：启用 vs 禁用谐波专家。
4. 两个处理对照：仅窄带抑制 vs 完整内容流约束。
5. 指标：谐波检测 AUROC（相对注入真值）；同频神经响应保真（注入的"神经"分量恢复误差）；去除率 vs 失真的 Pareto。
6. 报告刺激锁相响应保持率与失败案例。

**验收**：`runs/eval/tacs_phantom/` 含两路×两组×两类对照的全部结果；报告标题为"体模注入实验"；`BLOCKED.md` 已列出缺失证据。

---

### 阶段 6：真实适配与自监督（设计文档 P5）

**步骤**（每步都必须先满足前置假设）
1. **书面定义假设**（T6.1）：为候选自监督方法写清四点——① 两视图共享的底层目标；② 条件独立/盲点假设为何近似成立；③ 哪些相关伪迹可能被目标保留；④ 如何检测恒等映射和神经活动误删。**固定奇偶采样、固定盲点掩码、直接通道切分不得作为默认方案。**
2. **验证假设**（T6.2）：用相关性、频谱和伪迹注入实验检验假设。**假设不成立则停止，不得继续。**
3. **反退化测试**（T6.3）：检测恒等映射与生理活动误删（用 ds004784 的 Brain 条件做真值）。
4. **三路报告**（T6.4）：分别报告 ① 合成监督训练后直接测试真实数据；② 真实适配后的真实结果；③ 适配前后在合成留出集与清洁负对照上的变化。
5. **ERP 配对纪律**（T6.5）：ERP 试次配对**仅用于**诱发成分恢复或保持性验证，**不得**用于单试次解码、连接分析或通用去噪目标。

**验收**：`runs/eval/adaptation/` 含三路报告；假设验证文档存在且结论明确；恒等映射与误删检测指标已报告。

---

### 阶段 7：流式部署（设计文档 P6）

**步骤**
1. **重叠相加/交叉淡化**（T7.1）：实现重叠窗口 + 加权重建。
2. **运行稳健归一化**（T7.2）：比较三种方案（窗口内统计量 / 指数滑动或稳健运行统计量 / 训练集固定统计量 + 在线小幅校正），报告中给出跨窗尺度跳变对比。**不得默认独立窗口反归一化连续。**
3. **谐波相位/状态缓存与重置**（T7.3）：实现状态缓存与重置接口。
4. **窗口边界误差测量**（T7.4）。
5. **端到端延迟测量**（T7.5）：P50/P95，包含缓存、预处理、推理、反归一化、通信；report 显存与长时间连续运行（建议 ≥1 h）稳定性。
6. **流式基线对比**（T7.6）：同一硬件与缓存条件下比较。

**验收**：`runs/eval/streaming/` 含边界误差、P50/P95、显存曲线、≥1 h 稳定性记录；三种归一化方案的对比表。

---

### Gate 5–7

- [ ] 阶段 5：体模注入实验完成；`BLOCKED.md` 列出缺失证据；报告标题纪律合规
- [ ] 阶段 6：假设验证有明确结论；三路报告完整；ERP 配对仅用于规定用途
- [ ] 阶段 7：边界误差、P50/P95、显存、≥1 h 稳定性全部有实测记录

---

## 9. 设计文档 §14 清单 → 任务映射（50 项全覆盖）

执行 AI 用此表核对完整性。**每一项都必须有对应任务的验收记录。**

### P0：统一设计与当前代码（14 项）

| §14 条目 | 状态 | 对应任务 |
|---|---|---|
| 三专家原型…前向闭环 | 完成 | — （历史） |
| CUDA 环境、GPU smoke、训练/推理脚本 | 完成 | T0.5、T1.5 |
| 专家与令牌扩展为六 | 完成 | — （历史） |
| `[~]` 心电专家脉冲形态与自相关；**显式空间投影待多通道** | **补齐** | **T1.2、T1.3** |
| `[~]` 运动/瞬态差分与变化点；峰度和饱和待消融 | **补齐** | **T2.4（饱和）、T4.4（消融）** |
| 低容量 Unknown 及能量/调用/替代约束 | 完成 | — （保留，阶段 3 重训校准） |
| 路由返回 p_k / g̃_k / g_k / bypass | 完成 | — |
| clean-bypass（首轮调用率不达标） | **校准** | **T3.2–T3.5、T4.2** |
| 固定 Top-2 → 阈值激活 + 自适应 Top-r | 完成 | T3.3（r_max 调优） |
| 路由稀疏损失移到 p_k/g̃_k/激活数 | 完成 | — |
| 共享输出头 → 专家专属头 | 完成 | T4.4（利用其可干预性） |
| 谐波专家候选频率 + 可微正余弦基 | 完成 | T1.2（多通道化） |
| `[~]` 可选元数据；**元数据随机屏蔽待实验** | **补齐** | **T5.1** |
| 统一损失/身份门/输出字段命名 | 完成 | — |

### P1：数据基础设施（8 项）

| §14 条目 | 状态 | 对应任务 |
|---|---|---|
| 统一样本记录和字段有效性掩码 | 完成 | T2.1（多通道化） |
| **受试者/记录级分组器，先划分后切窗** | **待做** | **T2.2** |
| **防止清洁片段/伪迹源/派生视图跨集合** | **待做** | **T2.3** |
| 合成器时变包络与组合采样 | 完成 | T2.4（扩展） |
| **频率漂移、传播延迟、非线性饱和、局部事件掩码** | **待做** | **T2.4** |
| `[~]` 已接 EOG/EMG/ECG/真实运动标签；真实在线 tACS 未接 | **部分** | T2.5、T2.6、T2.7；tACS 见阶段 5 先决判断 |
| 留一伪迹家族与未见设备异常 Unknown 测试集 | 完成 | T4.1 |
| 数据集注册表与下载校验记录 | 完成 | T0.2（补 ds004784） |

### P2：训练协议（6 项）

| §14 条目 | 状态 | 对应任务 |
|---|---|---|
| 粗分解/残差/联合微调检查点与冻结策略 | 完成 | T3.3–T3.5 |
| masked BCE、masked severity、可信分量掩码 | 完成 | T2.5（ds004784 真值掩码） |
| **频带与事件保持损失可配置接口** | **待做** | **T3.1** |
| **综合验证选模，避免单一 MSE** | **待做** | **T3.6** |
| **≥3 随机种子完整训练** | **待做** | **T3.7** |
| 保存路由/分量/身份门诊断输出 | 完成 | T4.2 |

### P3：核心评价（7 项）

| §14 条目 | 状态 | 对应任务 |
|---|---|---|
| **八类划分测试** | **待做** | **T4.1** |
| **信号/时频/频带/路由/清洁输入指标** | **待做** | **T4.2** |
| **去除率与失真度 Pareto** | **待做** | **T4.3** |
| **专家槽位干预与跨种子分量匹配** | **待做** | **T4.4** |
| **按任务可比性预注册外部基线** | **待做** | **T4.5** |
| **≥1 下游任务及受试者级统计** | **待做** | **T4.6** |
| **事件级生理负对照** | **待做** | **T4.7** |

### P4：tACS 专项（6 项）

| §14 条目 | 可执行性 | 对应任务 |
|---|---|---|
| 有/无刺激频率先验对照 | 可执行 | T5.1–T5.2 |
| 谐波检测与同频响应恢复分开评价 | 可执行 | T5.5 |
| 接入 sham/校准/体模**至少两类** | **仅一类可执行** | T5.0（BLOCKED 申请） |
| 校准参考通道逐通道逐谐波幅相传递 | **不可执行** | T5.0（BLOCKED 申请） |
| 报告刺激锁相响应保持与失败案例 | 部分可执行 | T5.6 |
| 保真协议完成前不作神经响应恢复表述 | 可执行 | T5.7（表述纪律） |

### P5：真实适配与自监督（5 项）

| §14 条目 | 对应任务 |
|---|---|
| 书面定义共享目标与独立性假设 | T6.1 |
| 用相关性/频谱/注入实验验证假设 | T6.2 |
| 检测恒等映射与生理活动误删 | T6.3 |
| 分别报告适配前/后/负对照 | T6.4 |
| ERP 配对仅用于诱发成分恢复或保持性验证 | T6.5 |

### P6：流式部署（6 项）

| §14 条目 | 对应任务 |
|---|---|
| 重叠相加或交叉淡化 | T7.1 |
| 运行稳健归一化 | T7.2 |
| 谐波相位/状态缓存与状态重置 | T7.3 |
| 测量窗口边界误差 | T7.4 |
| 端到端 P50/P95 延迟、显存、持续运行稳定性 | T7.5 |
| 与流式基线同硬件同缓存比较 | T7.6 |

### 补充任务（本手册新增，非 §14 原有）

| 任务 | 内容 | 理由 |
|---|---|---|
| T0.1 | 路径单一真相源 | 决策 D2 的落地前提 |
| T0.2 | ds004784 规范化 | 决策 D3 |
| T0.3 | POOL 探测与 schema 实测 | 防止基于臆测写 loader |
| T0.4 | 目录骨架与配置基线 | 多阶段实验的配置管理前提 |
| T0.6 | 清理遗留 | 卫生 |
| **T0.7** | **环境加固与性能基线** | **充分利用双 5080 与 32 核 CPU 的前置任务；实测 `num_workers=0` 浪费 23%–35% GPU 时间，且 DataParallel 路径存在必现缺陷** |
| T1.1 | montage 与坐标表 | 决策 D1 的落地前提 |
| T1.4 | 合成器真混音 | 现有实现无空间信息，D1 下必须改 |
| T1.5 | 多通道冒烟与单通道回归 | 防 D1 改造引入静默退化 |
| T5.1 | 元数据真实实现 + 随机屏蔽 | §14 P0 遗留项，且 tACS 频率先验对照依赖它 |

---

## 10. 产物清单一览

| 路径 | 内容 | 产出任务 |
|---|---|---|
| `unicore_eeg/paths.py` | 路径单一真相源 | T0.1 |
| `data/raw/ds004784/download_manifest.json` | ds004784 校验清单 | T0.2 |
| `runs/_pool_probe/pool_probe.json` | POOL 实测规格 | T0.3 |
| `configs/base.yaml`、`configs/multichannel_base.yaml` | 配置基线 | T0.4 |
| `runs/baseline_singlechannel_seed42/` | 改造前基线 | T0.5 |
| `scripts/bench_hardware.py`、`runs/_env/hardware_baseline.json` | 硬件基线（GPU + DataLoader 吞吐） | T0.7 |
| `scripts/run_dual_gpu.py` | 双卡并行调度 | T0.7 |
| `unicore_eeg/montage.py`、`configs/montages/*.yaml` | 通道坐标 | T1.1 |
| `unicore_eeg/spatial_mixer.py` | 真混音 | T1.4 |
| `runs/_regress/regress_multichannel.md` | 单通道回归对照 | T1.5 |
| `unicore_eeg/preprocess.py`、`records.py` | 预处理与样本记录 | T2.1 |
| `unicore_eeg/splits.py` | 分组划分 | T2.2 |
| `unicore_eeg/leakage.py`、`scripts/check_leakage.py` | 泄漏检查 | T2.3 |
| `runs/ds004784_pairs/` | 对齐报告、混音矩阵、样本统计 | T2.5 |
| `unicore_eeg/pool/*.py` | 外部池 loader | T2.7 |
| `unicore_eeg/selection.py`、`configs/selection.yaml` | 综合选模 | T3.6 |
| `runs/final_seed{42,1234,20260922}/` | 三种子完整结果 | T3.7 |
| `unicore_eeg/eval/*.py` | 场景/指标/Pareto/tACS | T4.1–T4.3、T5 |
| `configs/baselines.yaml` | 基线预注册 | T4.5 |
| `runs/eval/` | 全部评价产物 | T4–T7 |

---

## 11. 风险与处置

| 风险 | 触发信号 | 处置 |
|---|---|---|
| 多通道改造引入单通道静默退化 | T1.2/T1.4 后单通道 `macro_auroc` 差 > 0.03 | 停止推进，二分定位到具体改动；必要时保留 `use_spatial=False` 与 `use_mixing=True` 的中间态 |
| ds004784 trigger 对齐失败 | 相关 < 0.5 | 写 `BLOCKED.md`；不得用"近似对齐"硬凑；先把 `Brain` 条件的对齐做成独立可验证子任务 |
| ds004784 单受试者导致无法做受试者级泛化 | 划分时发现只有 sub-001 | 划分改按 recording（6 任务），报告必须声明"单受试者体模，结论限于该体模" |
| POOL `.npz` schema 与旧代码不一致 | 探测时键名与旧 loader 期待不符 | 以实测 schema 为准，在 CODE 下重写解析；不要 `sys.path` 到旧项目 |
| 真实域失效原因混杂 | bypass 仍为 0，但输入尺度已统一 | 按 T2.6 + 阶段 4 分层归因：先统一接口，再逐层排除；报告必须分开列"尺度因素"与"域偏移因素" |
| ocular_drift 专家坍缩未解决 | 路由矩阵 `cardiac→ocular_drift` 仍 > 0.5 | 按 T4.4 做槽位干预定位；若确认特征不足，回 T2.4 补 cardiac 的空间/形态观测 |
| CHB-MIT/Sleep-EDF 等为子集 | 抽样核验发现记录数远少于全集 | 报告中一律写实际记录数；**不得**写"在 X 数据集上验证" |
| tACS 证据不足 | 阶段 5 发现只有体模一类证据 | 严格按先决判断执行；报告标题限定为"体模注入"；向用户申请真实 tACS 数据 |
| **GPU 被数据加载饿死** | `nvidia-smi` 显示利用率长期 < 60%，且 `num_workers=0` | 按 0.3.7 把 `num_workers` 提到 8；先跑 T0.7 的基线确认瓶颈在哪一侧 |
| **系统内存耗尽** | 报 `OpenBLAS error: Memory allocation still failed`，或可用内存 < 4 GiB | 双卡并行总 worker 降到 8（每进程 4）；按 0.3.9 清理残留 worker 进程；**不要**靠加内存条拖延 |
| **残留 worker 进程泄漏** | `nvidia-smi`/进程表出现无主的 `multiprocessing-fork` python 进程，各占约 1.4 GB | 任何脚本异常中断后，立即按 0.3.9 的 PowerShell 片段**精确匹配后**清理；这曾一次性泄漏 16 进程 / 16.5 GB |
| **`torch.compile` 卡死或报错** | 启用 `--compile` 后报 Triton 相关错误或长时间无输出 | 立即关闭。本机无 Triton，`torch.compile` 不在支持范围内；配置中 `torch_compile: false` 是硬约束 |
| **`--data-parallel` 报非标量梯度错** | `RuntimeError: grad can be implicitly created only for scalar outputs` | T0.7 的修法未落地：把 `loss.backward()` 改为 `loss.sum().backward()`。修完先单卡回归再启用多卡 |
| 计算预算超限 | 三种子 × 五阶段训练时间过长 | 先用小规模（`--train-samples 8192`）验证流水线，再决定全量；**不得**以减少种子数来"完成" T3.7 |

---

## 12. 附录

### 附录 A　命令模板

> **注意路径分隔符**：在 Bash 中一律用正斜杠写解释器路径（`D:/.../python.exe`）。反斜杠版本在 Git Bash 下可能被转义，导致找不到解释器。

```bash
# ---- 每条命令前 ----
export PATH="/usr/bin:/bin:/c/Windows/System32:/c/Windows:/c/Users/admin/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"
PY="D:/Anaconda_envs/envs/unicore-eeg/python.exe"
cd /d/codexwork/unicore

# ---- 环境自检（每阶段开始前）----
"$PY" -c "import torch,yaml,psutil,mne,wfdb;print(torch.__version__,torch.version.cuda,torch.cuda.device_count())"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv

# ---- 依赖（一次性）----
"$PY" -m pip install pyyaml psutil

# ---- 测试 ----
"$PY" -m pytest tests/ -v

# ---- 硬件基线（T0.7）----
"$PY" scripts/bench_hardware.py --out runs/_env/hardware_baseline.json

# ---- POOL 探测（只读）----
"$PY" scripts/probe_pool.py --out runs/_pool_probe/pool_probe.json

# ---- 泄漏检查（训练前必跑）----
"$PY" scripts/check_leakage.py --config configs/stage_b.yaml --require-zero

# ---- 单通道基线复现（num_workers=8，见 0.3.7）----
"$PY" scripts/first_experiment.py --epochs 12 --train-samples 65536 \
  --eval-samples 6000 --batch-size 128 --num-workers 8 --use-public-sources \
  --seed 42 --out runs/baseline_singlechannel_seed42

# ---- 多通道冒烟 ----
"$PY" scripts/smoke_multichannel.py --channels 1 3 8 34 64 128

# ---- 双卡并行跑三种子（0.3.6 策略 A；总 worker ≤ 16）----
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_stage_b.py --config configs/stage_b.yaml \
  --seed 42 --num-workers 8 --out runs/stage_b_seed42 &
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/train_stage_b.py --config configs/stage_b.yaml \
  --seed 1234 --num-workers 8 --out runs/stage_b_seed1234 &
wait
# 或用调度脚本（T0.7）：
# "$PY" scripts/run_dual_gpu.py --config configs/stage_b.yaml --seeds 42 1234 20260922
```

**允许使用 / 禁止使用**

| 用法 | 结论 |
|---|---|
| 双卡各跑一个独立实验 | ✅ 首选（0.3.6 策略 A） |
| `nn.DataParallel` 单实验加速 | ⚠️ 次选；**必须先修 `loss.backward()` → `loss.sum().backward()`** |
| 跨卡梯度同步 / DDP | ❌ 禁止（本机无 NCCL） |
| `torch.compile` | ❌ 禁止（本机无 Triton） |
| `num_workers=0` | ❌ 禁止用于正式训练（0.3.7 实测浪费 23%–35% GPU 时间） |
| 双卡并行且每进程 `num_workers=16` | ❌ 禁止（32 worker ≈ 45 GB > 31.8 GB 系统内存） |

### 附录 B　关键事实备忘（避免重复盘查）

**路径与项目**
- 主项目 = `D:\codexwork\unicore`；POOL = `D:\codexwork\eeg\data`（只读）。
- POOL 旁旧项目代码 = `D:\codexwork\eeg\src\metacorrseg\`，可只读参考、**禁止 import**。
- 旧项目约定：250 Hz / 4 s / 34 双极 —— 与本项目 500 Hz / 2 s **不同**，数值不可直接比较。

**环境（2026-09-22 实测）**
- conda 环境名 **`unicore-eeg`**，路径 `D:\Anaconda_envs\envs\unicore-eeg`；解释器 `D:\Anaconda_envs\envs\unicore-eeg\python.exe`。
- Python 3.11.16 / **PyTorch 2.11.0+cu128** / CUDA 12.8 / cuDNN 9.19.0 / 驱动 595.97。
- **GPU = 2 × RTX 5080**，各 15.92 GiB（合计 31.8 GiB），sm_120，84 SM，L2 64 MiB，两块在不同 PCIe 根端口。
- **CPU 32 核；系统内存 31.8 GiB 总（可用常低于 20 GiB）—— 内存比显存更紧。**
- 缺失：`PyYAML`、`psutil`、`Triton`（→ T0.7 补前两个，第三个不用）。
- **bf16 = 支持；NCCL = 不支持；Triton = 缺失。**

**性能实测（短测，用于定方向；定参数须按 T0.7 重测）**
- 模型参数量 15.37 M（C=1）／15.57 M（C=34）。
- 单卡 bf16：`bs=128` 是安全上限（C=1 时 7.63 GiB／477 samples/s；C=34 时 8.56 GiB／402 samples/s）；**`bs=256` 必 OOM**。
- DataLoader：`num_workers=0` → 310 samples/s（**成为瓶颈**）；`=4` → 2399；`=8` → 2275；`=12` → 4412。
- 每 worker 常驻约 1.4 GB → **双卡并行时总 worker ≤ 16**。
- Windows 多进程：入口必须 `if __name__ == "__main__":`；父进程被强杀后子 worker 不退出（实测泄漏 16 进程 ≈ 16.5 GB，会引发 `OpenBLAS memory allocation failed`）。

**模型与数据**
- 设计文档 §14 的 50 项清单经逐行核验**完全准确**，可直接当台账。
- ds004784 GT = `(153600, 21)` @512 Hz = 300 s；通道分组：脑 0-9、眼动 10-11、颈肌 12-15、面肌 16-19、trigger 20。
- 历史基线指标（`runs/full_experiment_v2_fixed`，单种子）：macro-AUROC 0.8988、artifact-presence 0.9937、clean-bypass 0.9820；**两个失败点**：ocular_drift 坍缩（cardiac→0.768）、PhysioMotion bypass = 0.0000。


### 附录 C　变更记录

执行 AI 每遇到与本手册冲突的情形，在此追加一行。

| 日期 | 任务 | 冲突/变更内容 | 处理方式 |
|---|---|---|---|
| 2026-09-22 | — | 手册初版（v1.0） | — |
| 2026-09-22 | 第 0.3 节 | v1.0 的环境章节过于简略，且已知 GPU 信息有误（实为**两块** RTX 5080，非一块） | 升为 v1.1：重写 0.3 节为实测环境章节（硬件/软件/能力约束/双卡策略/DataLoader/显存预算/自检）；新增 **T0.7 环境加固与性能基线**；新增 **§5.5 运行配置**；更新 Gate 0、附录 A/B、产物清单、风险表 |
