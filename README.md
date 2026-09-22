# UniCORE-EEG

本仓库实现设计文档 v2.1 的完整六专家框架：Harmonic、Ocular/Drift、Myogenic、Cardiac、Motion/Transient 与 Unknown。模型包含自主多标签令牌、自适应阈值 Top-r 路由、clean-bypass、专家专属输出头、加性粗分解、单步残差细化和身份门。

## 环境

已创建本地 Conda 环境 `unicore-eeg`，当前验证组合为 Python 3.11、PyTorch 2.11.0+cu128 和 RTX 5080。

```powershell
conda activate unicore-eeg
python -m pip install -e .
```

## 数据下载

下载 EEGdenoiseNet、MIT-BIH ECG 和 PhysioMotion 的首个受试者：

```powershell
python scripts/download_data.py --dataset all --motion-subjects sub-1
```

数据许可、用途与限制见 `data/DATASETS.md`，逐文件大小和 SHA-256 写入 `data/download_manifest.json`。PhysioMotion 全集约 36 GiB；可将 `sub-1` 替换为 `sub-1 sub-2 ... sub-30` 下载完整数据。

## 快速检查

```powershell
python scripts/smoke_test.py
```

## 首个实验

该实验训练完整六专家模型，并在清洁输入、五类已知单伪迹和 Unknown 未见机制上比较 Oracle、Learned、All Experts 三种路由。

```powershell
python scripts/first_experiment.py --use-public-sources
```

结果位于 `runs/first_experiment/`：

- `report.md`：主要指标和伪迹类别 x 专家激活矩阵；
- `summary_metrics.csv`：按路由和条件汇总的信号指标；
- `sample_metrics.csv`：逐样本结果；
- `diagnostics.json`：F1、AUROC、Unknown 误激活和完整路由矩阵；
- `last.pt`：模型检查点。

使用本机已有的完整 PhysioMotion 做真实标签路由外测：

```powershell
python scripts/evaluate_physiomotion.py --checkpoint runs/first_experiment/last.pt
```

本机其他可用数据路径及预定用途见 `data/local_sources.json`。这些数据不会因为“存在于硬盘”就直接混入训练；必须先核对许可、标签含义，并按受试者划分。

## 单独训练与推理

```powershell
python scripts/train.py --epochs 10 --samples 8192 --batch-size 24 --use-public-sources
python scripts/infer.py --checkpoint runs/unicore_synth/best.pt --input sample.npy --output cleaned.npy
```

公开源池按原始片段索引固定做 80/20 训练测试隔离。PhysioMotion 没有逐点清洁真值，不会被误当作配对重建训练数据；它用于后续真实运动伪迹外部验证。

当前 `runs/first_experiment/report.md` 是首轮基线，不是最终性能结论。首轮已知类 macro-AUROC 为约 0.79，但 clean-bypass、Unknown、Cardiac 和 Motion 仍明显不足；正式结论前需要继续训练、独立验证集校准、多随机种子和外部基线。
