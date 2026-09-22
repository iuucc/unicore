# UniCORE-EEG

本仓库实现设计文档 v2.1 的完整六专家框架：Harmonic、Ocular/Drift、Myogenic、Cardiac、Motion/Transient 与 Unknown。模型包含自主多标签令牌、自适应阈值 Top-r 路由、clean-bypass、专家专属输出头、加性粗分解、单步残差细化和身份门。

## 规格文档

| 文档 | 性质 |
|---|---|
| `UniCORE-EEG_统一脑电伪迹去除模型设计_v2.md` | 设计规格 v2.1，**已冻结**，与其它文档冲突时以它为准 |
| `UniCORE-EEG_分阶段实验设计与执行手册_v1.1.md` | **执行依据**：阶段划分、任务级步骤、验收条件、Gate 门禁 |
| `UniCORE-EEG_进度评估与行动建议_20260922.md` | 2026-09-22 的资产盘点与优先级建议 |

执行进度与每步实测记录见 `runs/<stage>/NOTES.md`。

## 环境

专用 Conda 环境，**不要用同机的其它环境**（`eeg`、`LenAdapt-Ensemble`、`convnext-cifar`、`vitd`、`knn-accel`、`hello-agent` 等），混用会引入不可复现的依赖差异。

| 项 | 值 |
|---|---|
| 环境名 | `unicore-eeg` |
| 环境路径 | `D:\Anaconda_envs\envs\unicore-eeg` |
| 解释器绝对路径 | `D:\Anaconda_envs\envs\unicore-eeg\python.exe` |
| Python | 3.11.16 |
| PyTorch | 2.11.0+cu128（CUDA 12.8，cuDNN 9.19.0） |

**所有脚本调用一律写解释器绝对路径**，不要依赖 `conda activate` 的 shell 状态：

```bash
PY="D:/Anaconda_envs/envs/unicore-eeg/python.exe"     # Bash
& "D:\Anaconda_envs\envs\unicore-eeg\python.exe"      # PowerShell
```

安装本项目（可编辑安装）：

```powershell
& "D:\Anaconda_envs\envs\unicore-eeg\python.exe" -m pip install -e .
```

装包只往该环境装，且**必须同步更新 `requirements.txt`、`pyproject.toml`、`environment.yml` 三处**。

### 本机环境实测（2026-09-22）

- **GPU = 2 × RTX 5080**，各 15.92 GiB（合计 31.8 GiB），sm_120。
- **CPU 32 逻辑核；系统内存 31.8 GiB 总，可用常低于 20 GiB —— 内存比显存更紧。**
- **bf16 支持；NCCL 不可用；Triton 缺失。** 因此**禁止 DDP、禁止 `torch.compile`**，多卡只用 `CUDA_VISIBLE_DEVICES` 做进程隔离。
- 单卡 `batch_size=128` 是安全上限，`256` 必 OOM。
- `num_workers=0` 会让 GPU 有约 23%–35% 的时间等数据；默认已改为 8，双卡并行时两进程 worker 总数**不得超过 16**。

硬件基线的可复现测量见 `runs/_env/hardware_baseline.json`（生成脚本 `scripts/bench_hardware.py`），完整约束推导见手册 §0.3。

## 目录与路径

| 代号 | 绝对路径 | 角色 | 允许的操作 |
|---|---|---|---|
| CODE | `D:\codexwork\unicore` | 唯一代码与产物根 | 读写 |
| DATA | `D:\codexwork\unicore\data` | 项目内已下载数据 + 登记文件 | 读写（`raw/` 下大文件只读） |
| POOL | `D:\codexwork\eeg\data` | 外部数据池（约 190 GB） | **只读** |

**`unicore_eeg/paths.py` 是全项目唯一允许出现外部数据池绝对路径的文件。** 需要池内文件时用 `paths.pool_path("bci2a", "A01T.gdf")`，或在配置里写 `@paths.EXTERNAL_POOL/...` 符号引用。写操作前用 `paths.assert_readonly_pool(target)` 守卫。

配置基线在 `configs/`：

- `base.yaml`：采样率/窗口/带通/划分/种子，以及运行与 DataLoader 配置；
- `multichannel_base.yaml`：在 `base.yaml` 之上增加 `montage`、`spatial`、`mixer` 段；
- `_paths.yaml`：`paths.py` 的可核对镜像（有单测保证不产生第二个真相源）。

一个实验一个配置文件，禁止复用同一文件跑不同实验。

### 同一权重处理不同通道数

模型的时序主干、专家和输出头对每个电极共享参数；路由器通过坐标条件化的集合汇聚整合通道信息。模型参数量不依赖 `C`，同一 checkpoint 可直接处理不同 montage 的原生通道数。不同 C 的记录可在一个 batch 内补齐到最大通道数，`channel_mask` 会屏蔽补齐位置。

通道坐标进入模型前统一为 `(right, anterior, superior)` 轴顺序。新增验收包含同一模型实例连续处理 C=1/3/8/34/64/128、通道重排等变，以及混合 C 批次损失反向传播。

本次跨 C 架构重构改变了主干和输出层参数形状；此前 `runs/` 中训练的 checkpoint 属于旧架构，不能直接加载到新模型，需重新训练后生成可跨 C 使用的新 checkpoint。

## 外部数据池（POOL）：只读

外部数据池直接引用主机现有文件，**不复制、不建软链、不额外配置**。三条纪律：

1. 禁止创建、修改、删除 POOL 下的任何文件（含 `__pycache__`）；
2. 禁止把 POOL 下的 `.npy/.npz/.edf/.mat/.gdf/.bdf/.set/.fdt` 复制进本仓库；
3. POOL 旁的旧项目代码（`D:\codexwork\eeg\src\metacorrseg\`）**可只读参考，禁止 import**；需要时复制到本仓库改写。

池内各库的实测规格（采样率、通道数、文件数、体积）见 `data/DATASETS.md` 的"外部池实测规格表"，原始探测结果见 `runs/_pool_probe/pool_probe.json`。**这些数据不会因为"存在于硬盘"就直接混入训练**，必须先核对许可、标签含义，并按受试者划分。

## 数据下载

下载 EEGdenoiseNet、MIT-BIH ECG 和 PhysioMotion 的首个受试者：

```powershell
& "D:\Anaconda_envs\envs\unicore-eeg\python.exe" scripts/download_data.py --dataset all --motion-subjects sub-1
```

数据许可、用途与限制见 `data/DATASETS.md`，逐文件大小和 SHA-256 写入 `data/download_manifest.json`。PhysioMotion 全集约 36 GiB；可将 `sub-1` 替换为 `sub-1 sub-2 ... sub-30` 下载完整数据。

ds004784（头模体模数据，含物理真值分量）的校验清单：

```powershell
& "D:\Anaconda_envs\envs\unicore-eeg\python.exe" scripts/download_ds004784.py --verify
```

它逐个核对文件大小与 SHA-256 并写出 `data/raw/ds004784/download_manifest.json`，不发起下载。

## 快速检查

```powershell
& "D:\Anaconda_envs\envs\unicore-eeg\python.exe" -m pytest tests/ -q
& "D:\Anaconda_envs\envs\unicore-eeg\python.exe" scripts/smoke_test.py
```

## 首个实验

该实验训练完整六专家模型，并在清洁输入、五类已知单伪迹和 Unknown 未见机制上比较 Oracle、Learned、All Experts 三种路由。

```powershell
& "D:\Anaconda_envs\envs\unicore-eeg\python.exe" scripts/first_experiment.py --use-public-sources
```

结果位于 `runs/first_experiment/`：

- `report.md`：主要指标和伪迹类别 x 专家激活矩阵；
- `summary_metrics.csv`：按路由和条件汇总的信号指标；
- `sample_metrics.csv`：逐样本结果；
- `diagnostics.json`：F1、AUROC、Unknown 误激活和完整路由矩阵；
- `last.pt`：模型检查点。

改造前的单通道历史基线与复现命令见 `runs/baseline_singlechannel_seed42/NOTES.md`。

使用本机已有的完整 PhysioMotion 做真实标签路由外测：

```powershell
& "D:\Anaconda_envs\envs\unicore-eeg\python.exe" scripts/evaluate_physiomotion.py --checkpoint runs/baseline_singlechannel_seed42/last.pt
```

## 单独训练与推理

```powershell
& "D:\Anaconda_envs\envs\unicore-eeg\python.exe" scripts/train.py --epochs 10 --samples 8192 --use-public-sources
& "D:\Anaconda_envs\envs\unicore-eeg\python.exe" scripts/infer.py --checkpoint runs/unicore_synth/best.pt --input sample.npy --output cleaned.npy
```

公开源池按原始片段索引固定做 80/20 训练测试隔离。PhysioMotion 没有逐点清洁真值，不会被误当作配对重建训练数据；它用于真实运动伪迹外部验证。

当前 `runs/baseline_singlechannel_seed42/` 是**改造前基线**，不是最终性能结论。已知类 macro-AUROC 约 0.90，但真实域 clean-bypass 失效、cardiac 与 motion 仍存在专家坍缩；正式结论前需要多通道改造、独立验证集校准、多随机种子和外部基线。
