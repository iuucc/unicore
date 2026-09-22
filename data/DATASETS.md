# UniCORE-EEG 数据注册表

## EEGdenoiseNet

- 官方来源：https://gin.g-node.org/NCClab/EEGdenoiseNet
- 许可：CC0 1.0
- 内容：4514 个清洁 EEG、3400 个 EOG、5598 个 EMG 片段。
- 本项目用途：清洁目标池、眼动伪迹源池、肌电伪迹源池。
- 限制：原始片段约 1 秒；构造 2 秒窗口时需要重采样或拼接，不能把派生窗口当作独立受试者样本。

## MIT-BIH Arrhythmia Database

- 官方来源：https://physionet.org/content/mitdb/1.0.0/
- 许可：ODC Attribution 1.0。
- 本项目用途：将 ECG 波形作为受控心电污染源注入清洁 EEG。
- 限制：这是 ECG 源，不是真实 EEG 中的心电投影；只能支持受控混合实验。

## PhysioMotion Artifact

- OpenNeuro：ds006386，版本 1.0.1。
- DOI：10.18112/openneuro.ds006386.v1.0.1。
- 许可：CC0 1.0。
- 内容：30 位受试者、180 段记录、点级运动伪迹标签。
- 本项目用途：运动/瞬态专家的真实数据外部验证和后续弱监督适配。
- 下载策略：默认先下载 `sub-1` 以验证解析与训练流程；完整数据约 36 GiB，可通过下载器指定全部受试者。

## ds004784 —— 头模（体模）数据集

- OpenNeuro：ds004784，版本 1.0.4。DOI：`10.18112/openneuro.ds004784.v1.0.4`。
- 论文：Downey, R.J.; Ferris, D.P. *iCanClean Removes Motion, Muscle, Eye, and Line-Noise
  Artifacts from Phantom EEG*. Sensors 2023, 23, 8214。DOI：`10.3390/s23198214`。
- 许可：CC0 1.0。
- 内容：导电头模记录。真值 `stimuli/GTdata_croppedToRisingEdge.mat` 为 `(153600, 21)`
  @512 Hz = 300 s，通道分组为脑源 0–9、眼动 10–11、颈肌 12–15、面肌 16–19、trigger 20；
  BIDS 侧六个任务 Brain / Eyes / Facial / Neck / Walking / All，各 311–357 s。
  每份记录的 264 个通道 = **128 个 EEG**（`A1`…`A128`，BioSemi ActiveTwo 布局）
  + 8 个 EMG（`EXG1`…`EXG8`）+ 128 个 MISC（`N-A1`…`N-A128` 噪声通道）。
  `sub-001_task-*_electrodes.tsv` 给出逐通道 x/y/z（单位 **mm**，未归一化）。
- 本项目用途：**体模真值监督**（唯一带物理真值分量的数据）、**清洁输入负对照**（Brain 条件）、
  **Unknown 压力测试**。
- 限制：
  - GT 为 300 s 而 BIDS 记录 311–357 s，**必须按 trigger 事件 `65471` 对齐**后才能做配对监督；
  - 只有单受试者（`participants.tsv`：`sub-001, group=phantom`），
    因此划分只能按 recording（6 个任务），报告必须声明"单受试者体模，结论限于该体模"。
- 校验：`data/raw/ds004784/download_manifest.json` 逐文件记录 `bytes` 与 `sha256`。
  2026-09-22 实测 348 个对象全部 `verified`，合计 11 621 460 277 字节（10.82 GiB）；
  `aria2_missing.txt` 为 0 字节，即无缺失对象。

## 数据边界

首次实验使用 EEGdenoiseNet 与 MIT-BIH 的真实源波形，加上可控谐波、运动及未知机制生成配对数据。PhysioMotion 没有逐点清洁真值，不能直接并入配对重建训练，其首个用途是独立真实运动伪迹验证。

## 本机已有扩展数据

路径注册在 `data/local_sources.json`，不复制大型原始文件。

| 数据集 | 可用用途 | 当前边界 |
|---|---|---|
| PhysioMotion | 眼动、肌电、头动及组合伪迹的真实多标签路由外测 | 无逐点清洁真值，不做重建排名 |
| BCI Competition IV 2a/2b、OpenBMI、PhysioNet MI、Cho GigaDB | 清洁输入保持与运动想象下游任务 | 需按受试者划分，不默认全部片段清洁 |
| CHB-MIT | 棘波/发作形态负对照 | 病理信号不是伪迹，禁止作为待删除目标 |
| CAP Sleep、Sleep-EDF Expanded | K 复合波、纺锤波及睡眠分期保持 | 事件标签质量需逐库核验 |
| ds002094 TMS-EEG | 强刺激瞬态与 Unknown 压力测试 | TMS 脉冲和神经响应重叠，不能当作简单加性真值 |
| FACED | 清洁保持与情感识别下游任务 | 预处理版本与原始 BDF 必须分开报告 |

`tuar` 目录当前为空，不能宣称已接入 TUAR。根目录下未登记来源的 `eeg*.edf` 暂不使用，需先确认来源、许可和标签定义。

## 外部池实测规格表（2026-09-22）

来源：`scripts/probe_pool.py` → `runs/_pool_probe/pool_probe.json`。
采样率与通道数由 `mne` 实测每个数据集的**前 3 条**原始记录（`raw_uniform` 标明整库是否一致）；
`npz` 用 `np.load(mmap_mode="r").files` dump schema；`mat` 用 `scipy.io.loadmat(simplify_cells=True)`。
本次探测已验证对池零写入（`pool_untouched_all = true`）。

| 数据集 | 文件数 | 体积 | 采样率 | 通道 | 整库一致 | 备注 |
|---|---:|---:|---|---|---|---|
| physiomotion | 1351 | 20.62 GiB | **1000 Hz** | 34（双极） | 是 | 低通 150 Hz；`Manual_Annotations` 180 份 CSV |
| bci2a | 36 | 574.7 MiB | 250 Hz | **25**（22 EEG + 3 EOG） | 是 | `.mat` 只含 `classlabel`(288) |
| openbmi | 324 | 63.50 GiB | 原始 **1000 Hz**／缓存 npz 250 Hz | 原始 **62**／缓存 npz **20** | 原始是 | 原始路径 `raw/session{n}/s{k}/sess*_EEG_{MI,Artifact}.mat`，顶层为嵌套 struct |
| physionet_mi | 2376 | 1.99 GiB | 160 Hz | EDF 64／缓存 npz 21 | 是 | 缓存导联 `FC5`…`CP6`；`tmin=0.5, tmax=3.5` |
| chbmit | 89 | 2.25 GiB | 256 Hz | 23 | 是 | 39 个 EDF（**子集**，全集约 686） |
| sleep_edfx | 725 | 10.05 GiB | 100 Hz | EDF 7 路信号（2 EEG + EOG + Resp + EMG + Temp + Event） | 是 | `*-Hypnogram.edf` 无信号通道（预期） |
| cap_sleep | 209 | 28.34 GiB | **256 或 512 Hz（不统一）** | **18 或 22（不统一）** | **否** | 缓存 npz 8 导 @128 Hz，`source_sample_rate=512` |
| ds002094 | 461 | 39.45 GiB | **5000 Hz** | 30 | 是 | BrainVision `.vhdr`；TMS 记录时长可达 3096 s |
| faced | 411 | 6.25 GiB | 250 Hz | 32 | 是 | 缓存 npz 键含 `channel_names`、`rating` |
| cho_gigadb | 260 | 10.05 GiB | 原始 **512 Hz**／缓存 npz 256 Hz | 原始 **64**（`senloc` `(64,3)`）／缓存 npz **21** | — | 缓存导联 `FC5`…`CP6` |
| bci2b | 92 | 487.2 MiB | 250 Hz | **6**（3 EEG `C3/Cz/C4` + 3 EOG `ch01..03`） | 是 | `.mat` 只含 `classlabel`(120) |
| tuar | 0 | 0 | — | — | — | **空目录，未接入** |

### 实测发现的三处与预期不符

1. **PhysioMotion 是 1000 Hz，不是 500 Hz。** 低通 150 Hz。当前 `unicore_eeg/physiomotion.py`
   用 `F.interpolate` 线性插值到 500 Hz，**不抗混叠**，必须改为 `scipy.signal.resample_poly`。
2. **CAP Sleep 整库不统一**：`brux1.edf` 512 Hz/18 导、`brux2.edf` 256 Hz/18 导、`ins8.edf` 512 Hz/22 导。
   涉及该库的结论必须写明具体记录，不能假设统一采样率。
3. **缓存与原始的通道数不同**：openbmi 20 vs 62、cho_gigadb 21 vs 64、physionet_mi 21 vs 64。
   凡使用 `*_cache` / `meta_cache` / `feature_cache` 的路径，必须先 dump schema 并声明实际导联数。
