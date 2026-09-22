# UniCORE-EEG 资产盘点、进度评估与下一步行动建议

> 评估日期：2026-09-22
> 评估对象：`D:\codexwork\unicore`（主项目）、`D:\codexwork\eeg\data`（外部数据池）
> 评估依据：仓库源码、`data/` 目录实测、`runs/` 实验产物、设计文档 v2.1 第 13–14 节清单

---

## 0. 结论速览

1. **主项目不在当前工作区。** 当前会话目录 `C:\Users\admin\WorkBuddy\2026-09-22-10-32-14` 为空目录，无 `data`、无代码、无文档。主项目为 `D:\codexwork\unicore`，`D:\codexwork\eeg` 是上一代（MetacorrSeg）项目兼外部数据池。

2. **已注册数据集基本完整，但"完整"只对已登记范围成立。** 项目内 `data/raw` 的 EEGdenoiseNet、MIT-BIH、on004784 逐文件校验通过；PhysioMotion 本仓副本**不完整**（缺全部标注与预处理数据）；外部池的 `tuar` **为空**，CHB-MIT/CAP Sleep/Sleep-EDF/FACED 为**子集**而非全集。

3. **约 190 GB 外部数据完全未接入代码。** `bci2a / bci2b / openbmi / physionet_mi / cho_gigadb / chbmit / cap_sleep / sleep_edfx / ds002094 / faced / physiomotion` 在全部 `.py` 文件中**零引用**，仅在 `data/local_sources.json` 里声明了用途。设计文档 P1–P3 中大量条目依赖它们，目前一条都跑不起来。

4. **框架主体（六专家前向 + 训练 + 首轮实验闭环）已完成且可运行**，合成实验指标良好（macro-AUROC 0.899、artifact-presence AUROC 0.994、clean-bypass 0.982）。设计文档 §14 清单与实际代码**逐项吻合**，文档可信，无需重新盘查。

5. **真实数据是最大短板。** PhysioMotion 外测 clean-bypass 为 **0.0000**、clean 窗口 ocular_drift 激活 **0.674**，域差距显著；专家坍缩仍在（synthetic 路由矩阵里 cardiac→ocular_drift 0.768、myogenic→ocular_drift 0.445）。

6. **新下载的 on004784（ds004784 体模数据集）是被低估的机会。** 它是本仓库目前**唯一带物理真值分量**的数据——10 路脑源 + 2 路眼动 + 8 路肌电真实记录，可直接提供设计文档最缺的"可信分量标签"和"清洁输入真值"。但当前只做了 sanity check，`On004784WindowDataset` **未接入任何训练/评估脚本**，GT 只取了 shape 元信息。

---

## 1. 目录与资产定位

| 路径 | 角色 | 状态 |
|---|---|---|
| `C:\Users\admin\WorkBuddy\2026-09-22-10-32-14` | 当前会话工作区 | **空**（0 个条目） |
| `D:\codexwork\unicore` | **主项目**（本报告对象） | 2 461 行 Python + 设计文档 + 实验产物 |
| `D:\codexwork\eeg` | 上一代 EEG 项目（MetacorrSeg 等）兼**外部数据池** | 数据可复用，代码与本项目无关 |

> 注：`D:\codexwork\eeg` 根目录下的 `checkpoints/`、`configs/`（含 `metacorrseg_v2.yaml`）、`runs` 类目录、`src/` 均属旧项目，**不要**与 UniCORE-EEG 混淆。UniCORE 的检查点一律在 `unicore/runs/<run_name>/*.pt`。

---

## 2. 数据资产盘点

### 2.1 项目内 `D:\codexwork\unicore\data\raw`

| 数据集 | 文件数 | 体积 | 完整性判定 | 依据 |
|---|---:|---:|---|---|
| `eegdenoisenet` | 3 | 93 MB | **完整** | 三份 npy 字节数与 `download_manifest.json` 完全一致（36 978 768 / 45 858 896 / 13 926 480） |
| `mitdb` | 15 | 9.4 MB | **完整（但仅 5 条记录）** | 100/101/102/103/105 × {`.hea`,`.dat`,`.atr`}，与 manifest 一致；MIT-BIH 全集 48 条，此处为子集 |
| `physiomotion` | 26 | 129 MB | **不完整** | 只有 `sub-1` 原始 BIDS；`derivatives/preprocessed_BIDS` 内**仅一个 README**，无 `sub-*/eeg/*.edf`；无 `derivatives/Manual_Annotations/` |
| `on004784`（ds004784） | 358 | 11 GB | **完整** | 6 个任务（Brain/Eyes/Facial/Neck/Walking/All）的 `.set/.fdt/.json/.tsv` 齐全；`stimuli/GTdata_croppedToRisingEdge.mat` (22 MB) 在位；`aria2_missing.txt` 为 **0 字节**（即无缺失对象） |

**关于 PhysioMotion 本仓副本**：`unicore_eeg/physiomotion.py:89` 需要
`derivatives/preprocessed_BIDS/sub-{id}/eeg/sub-{id}_task-artifact_run-{NN}_eeg.edf`，
`unicore_eeg/physiomotion.py:87` 需要 `derivatives/Manual_Annotations/sub{id}_run*.csv`。
本仓副本两者皆无，**无法在本仓路径下运行 `evaluate_physiomotion.py`**。脚本能跑通是因为其默认根路径硬编码指向外部池（`scripts/evaluate_physiomotion.py:19` → `D:/codexwork/eeg/data/physiomotion`）。

### 2.2 外部数据池 `D:\codexwork\eeg\data`

| 数据集 | 文件数 | 体积 | 完整性判定 | 代码接入 |
|---|---:|---:|---|---|
| `physiomotion` | 1 351 | 21 GB | **完整** | 仅路由外测（不含训练） |
| `bci2a` | 36 | 575 MB | 完整（9 受试者 × 2 session） | **无** |
| `bci2b` | 92 | 488 MB | 完整（9 × 5 session） | **无** |
| `openbmi` | 324 | 64 GB | 完整（54 受试者 × 2 session 已缓存） | **无** |
| `physionet_mi` | 2 376 | 2.0 GB | 4 类子集（runs 04/06/08/10/12/14） | **无** |
| `cho_gigadb` | 260 | 11 GB | 52 受试者，基本完整 | **无** |
| `chbmit` | 89 | 2.3 GB | **子集**（仅 39 个 EDF，全集约 686） | **无** |
| `cap_sleep` | 209 | 29 GB | **子集**（56 条记录） | **无** |
| `sleep_edfx` | 725 | 11 GB | **子集**（306 个 EDF） | **无** |
| `ds002094` | 461 | 40 GB | 基本完整（43 条记录 / vhdr+vmrk） | **无** |
| `faced` | 411 | 6.3 GB | **子集**（55 个 BDF，全集 123 受试者） | **无** |
| `tuar` | **0** | **0** | **空目录** | **无**（`data/DATASETS.md:44` 已如实声明） |

外部池合计约 **190 GB**，全部只在 `data/local_sources.json` 里以"角色声明"形式存在，**没有任何 Python 模块能读它们**。

### 2.3 完整性的诚实口径

可以说的：EEGdenoiseNet、MIT-BIH、on004784 三份已登记数据**逐文件校验通过**。
不能说的：PhysioMotion 在本仓已完整；外部池各库已全集下载（CHB-MIT、CAP Sleep、Sleep-EDF、FACED 明确是子集）；TUAR 已接入。
另外：`data/raw/on004784` **没有生成 `download_manifest.json`**（走的是 aria2 通道），完整性依据是**文件大小比对**而非 SHA-256，与 EEGdenoiseNet/MIT-BIH 的校验强度不一致。

---

## 3. 数据集与模型框架匹配度校验

### 3.1 框架的真实输入契约（从代码提取）

`unicore_eeg/model.py:21-38` 默认配置：`in_channels=1`、`sample_rate=500`、`window_size=1000`（2 s）、`artifact_count=6`、`metadata_dim=8`、`max_active_experts=4`。

`unicore_eeg/losses.py:61-127` 对 batch 的硬性要求：`noisy`、`clean`、`artifacts`（形状 `(6, C, T)`）、`labels`(6)、`label_mask`、`component_mask`、`severity`、`is_clean`、`disabled_experts`、`metadata`。

**推论：能直接训练的数据集必须同时提供"配对 clean + 六类分量 + 多标签"。** 满足这一条目的目前只有 `SyntheticEEGDataset`。

### 3.2 逐数据集匹配结论

| 数据集 | 采样率 | 与 500 Hz / 2 s 契约 | 可提供的监督 | 接入状态 |
|---|---|---|---|---|
| EEGdenoiseNet | 512 Hz，1 s/段 | 需拼接 + 重采样（`synthetic.py:56-69` 已实现） | clean、EOG、EMG 源 | **已接入**（训练主源） |
| MIT-BIH | 360 Hz | 需重采样（`synthetic.py:80-94` 已实现） | ECG 源 | **已接入** |
| PhysioMotion | 500 Hz EDF | 直接可用（`physiomotion.py:124-134`） | **仅点多标签**，无 clean 真值 | 仅外测 |
| on004784 | 512 Hz，128 导 | 需重采样；`channels=1` 时只用第 1 导 | **真实 clean + 眼动/颈肌/面肌分量真值** | **仅 sanity check** |
| 外部池 11 个库 | 100–1000 Hz 不等 | 未做任何适配 | clean 保持 / 下游 / 负对照 | **零接入** |

### 3.3 匹配层面的关键问题

**（1）单通道设定浪费了最关键的数据。** `model.py:23` 默认 `in_channels=1`，`on004784.py:194` 只取 `eeg_channel_indices[:channels]` → 128 导体模数据只用 1 导。而设计文档 §2.2、§3.4.4 明确指出心电/眼动的可分性主要来自**空间投影**，`§13` 也承认"显式空间投影仍待多通道实验"。当前所有已跑实验均为单通道，多通道路径**架构上支持但从未被验证过**。

**（2）on004784 的 GT 与 BIDS 记录存在时间对齐问题。** `stimuli/readme.txt` 声明 GT 为 5 分钟（153 600 点 @ 512 Hz = 300 s），而 BIDS 六个任务时长 311–357 s（`data/on004784_sanity_report.md`）。两者需要按同步脉冲（GT 第 21 路 trigger）对齐后才能做配对监督——这是一项**必须做但尚未做**的工程。

**（3）元数据通道是空壳。** `metadata_dim=8`，但合成数据只填 2 个槽位（`synthetic.py:280-281`：采样率、通道数归一化），on004784 填 3 个（`on004784.py:203-205`）。设计文档 §3.3 要求的"刺激频率/导联/设备信息"和"随机屏蔽以支持无元数据推理"**均未实现**，P4 的"有/无刺激频率先验对照"因此无从做起。

**（4）PhysioMotion 外测存在归一化不一致。** `physiomotion.py:137-142` 返回**未做稳健标准化**的原始 `eeg`，而 `on004784.py:199-201` 返回已标准化的信号。模型内部有 `robust_normalize`（`model.py:455-461`）兜底，故不致命，但两个真实数据集接口不一致，会干扰"域差距"的归因——外测表观差距里有多少来自真实域偏移、有多少来自输入尺度差异，目前分不清。

---

## 4. 模型框架实现进度

### 4.1 已完成（可运行、有代码证据）

| 模块 | 证据 |
|---|---|
| 六专家主干 | `model.py:296-325`，`ARTIFACT_NAMES` 六类；每类独立 kernel/dilation 配置与 FiLM 适配器 |
| 低容量 Unknown 专家 | `model.py:308` `hidden_scale=0.5`；`model.py:328-343` `UnknownDetector`；`losses.py:121-126` 能量 + 替代率约束 |
| 自主多标签令牌 | `model.py:167-218`，存在性/严重程度/优先级/条件令牌四路输出 |
| 伪迹观测抽取 | `model.py:124-164`：可微正弦基（谐波）、移动平均（眼动）、差分（肌电/运动）、自相关寻峰（心电） |
| 自适应 Top-r 路由 + clean-bypass | `model.py:221-273`，返回 `p_k / 归一化前分数 / active_mask / g_k / bypass`，支持 oracle·learned·all 三模式；`tests/test_core.py:28-41` 有单测 |
| 内容主干四级多尺度 CNN | `model.py:276-293`，48/96/160/256，dilation 1/2/4/8，短-中-长三卷积并行（`model.py:99-110`） |
| 跨流软门控 | `model.py:346-365` |
| 专家专属输出头 + 加性粗分解 | `model.py:378-393`（六头独立，可整体干预） |
| 单步残差细化（有界 tanh + ρ） | `model.py:396-418` |
| 清洁身份门 | `model.py:421-433` |
| 统一损失 | `losses.py:56-152`：Charbonnier + 相关 + 差分 + MRSTFT + coarse + 分量（带 mask）+ 总量一致 + masked BCE/severity + 归一化前稀疏 + 身份 + Unknown |
| 分阶段训练（对应 §6 阶段 B/C/D） | `first_experiment.py:56-75`：decomposition → residual → joint（joint 用 0.2× lr） |
| 首轮实验闭环 + 报告 | `first_experiment.py` 全流程，产物落在 `runs/full_experiment_v2_fixed/` |
| 数据注册与校验 | `data/download_manifest.json`（EEGdenoiseNet/MIT-BIH 含 SHA-256）、`data/DATASETS.md`、`data/local_sources.json` |
| 环境 | conda env `unicore-eeg` 存在于 `D:\Anaconda_envs\envs\unicore-eeg` |

### 4.2 部分完成

| 模块 | 差在哪 |
|---|---|
| 心电专家 | 有脉冲形态 + 自相关周期观测（`model.py:153-161`），**无空间投影、无参考通道** |
| 运动/瞬态专家 | 有一阶/二阶差分（`model.py:162`），**无峰度、无变化点、无饱和指示** |
| 元数据 | 通道打通，**内容为空壳、无随机屏蔽**（见 §3.3-3） |
| 分量稳定性与物理支持性 | 有 `artifact_components_norm` 输出，**无跨种子匹配、无槽位干预实验** |
| 真实数据 | PhysioMotion 路由外测已跑，**无弱监督/自监督适配** |
| 论文级评价 | 有单种子首轮三路路由对比，**无多种子、无外部基线、无下游任务** |

### 4.3 未完成（设计文档已列、代码确认缺失）

- **P1 数据基础设施**：受试者/记录级分组器（`synthetic.py:59-60` 用索引 80/20 切分，非受试者级；on004784/PhysioMotion **完全没有划分**）；频率漂移 / 传播延迟 / 非线性饱和 / 局部事件掩码增强。
- **P2 训练协议**：频带与事件保持损失接口（`losses.py` 中不存在）；综合验证选模（`train.py:140` 仅按 `val_loss` 选 `best.pt`）；≥3 随机种子完整训练。
- **P3 核心评价**：Seen/Unseen-combination/Unseen-severity/Unseen-artifact/Clean-input 五类划分、时频与频带指标、Pareto 分析、槽位干预、跨种子分量匹配、外部基线、下游任务、生理负对照——**整块空白**（`first_experiment.py:144-232` 只有 RRMSE/相关/SNR/artifact-RRMSE/修改量）。
- **P4 tACS 专项**：全部未做。
- **P5 真实适配与自监督**：全部未做。
- **P6 流式部署**：全部未做（无重叠相加、无运行归一化、无状态缓存、无延迟测量）。
- **阶段 A**（令牌与专家独立预训练）：未做成独立脚本，被折进联合训练。
- **阶段 E**（真实数据适配）：未实现。

### 4.4 实验证据现状（`runs/full_experiment_v2_fixed`，2026-09-21 19:04，单种子，12 epochs / 65 536 样本）

合成留出集：

| 指标 | 数值 |
|---|---:|
| macro-F1 @0.5 | 0.5685 |
| macro-AUROC | 0.8988 |
| artifact-presence AUROC | 0.9937 |
| unknown detection AUROC | 0.9997 |
| unknown 清洁误激活 | 0.0019 |
| clean 条件 bypass_rate | 0.9820 |
| clean 条件 RRMSE | 0.0021 |

各伪迹 SNR 改善（learned）：harmonic 13.58 dB、motion 8.22、unknown 7.12、myogenic 5.66、ocular 4.97、cardiac 4.76。

**两个未解决的失败点：**

1. **专家坍缩仍在。** learned 路由矩阵：`cardiac → ocular_drift 0.768`、`myogenic → ocular_drift 0.445`；留一条件下 `heldout_cardiac → ocular_drift 0.718`、`heldout_motion → ocular_drift 0.584`。ocular_drift 成了实际上的"默认桶"，直接威胁设计文档 §11"稀疏专家优于共享分支"这条假设的可证伪性。
2. **真实域彻底失效。** PhysioMotion 外测 `clean bypass_rate = 0.0000`，clean 窗口 `ocular_drift` 激活 0.674、myogenic 0.246；motion 的 F1 仅 0.0623。合成集上的 0.98 bypass 完全没能迁移。

现有结论的正确表述（与设计文档 §15 一致）：**当前是"可运行原型"，不是"可支撑论文主张的首发模型"。**

---

## 5. 文档规划梳理

项目内 Markdown 文档已全部通读，规划类只有四份（其余为实验报告）：

| 文档 | 性质 | 关键内容 |
|---|---|---|
| `UniCORE-EEG_统一脑电伪迹去除模型设计_v2.md` (v2.1, 708 行) | **路线图与验收标准** | §6 五阶段训练、§7 监督阶梯、§9 评价协议、§10 基线与消融、§11 可证伪预测、§13 实现差距表、§14 带 `[x]/[~]/[ ]` 的实施清单、§15 完成定义（10 条） |
| `README.md` | 使用说明 | 环境、下载、smoke、首轮实验命令与产物说明 |
| `data/DATASETS.md` | 数据注册表 | 许可、用途、边界；**明确写了 tuar 为空、未登记 edf 不用** |
| `runs/full_experiment_v2_fixed/问题修正与全量重跑总结.md` | 阶段性复盘 | 五个原始问题、五项修改、重跑对照表、**下一步建议** |

**文档与代码的一致性核验结论：设计文档 §14 清单准确，可以当作任务台账直接使用。** 我逐条对照了源码，`[x]` 的都真有实现，`[~]` 的确实差上述内容，`[ ]` 的确一条都没有。没有发现"文档说做了、代码没做"的情况——这一点值得肯定。

文档给出的下一步（`问题修正与全量重跑总结.md:97`）：*"PhysioMotion 弱监督适配、真实 clean/baseline 校准，以及对 cardiac/motion 的特征和标签监督继续增强。"* 这与我的独立判断一致，但它**没有意识到 on004784 的价值**——因为该数据集是复盘报告写完之后（10-05 至 10-23）才接入的。

---

## 6. 带优先级的行动建议

排序逻辑：先补齐"证据链最短缺且成本最低"的一环，再动需要长周期的训练/评价工程。**P0 三项互相独立，可并行。**

### P0-1　把 on004784 从"能读"推进到"能训练"（最高性价比）

**理由**：这是全仓唯一能提供**可信分量标签 + 真实 clean 真值**的数据，直接命中设计文档 §5.2「仅对具有可信分量标签的类别监督」和 §15 第 5 条「心电与运动/瞬态不能只靠合成结果支持」的前半句需求。当前价值几乎为零。

**具体动作**
1. 实现 GT↔BIDS 时间对齐：用 `stimuli/GTdata_croppedToRisingEdge.mat` 第 21 路 trigger 与 BIDS `events.tsv` 的 `x65471` 标定偏移，落成可复跑的脚本，并把对齐残差写进报告。
2. 由 GT 21 路源 + 混音矩阵构造**分量真值**：`ocular = GT[10:12]`、`neck_myogenic = GT[12:16]`、`facial_myogenic = GT[16:20]`、`brain = GT[0:10]` 投影到所选导联；据此填充 `artifacts (6,C,T)`、`component_mask`、`clean`，使样本记录满足 `losses.py` 的契约。
3. 保留 `Facial` 与 `Neck` 的 family 区分（`on004784.py:59-64` 已映射到同一 `myogenic`，要做分层统计就得留住这个区分）。
4. 用 `Brain` 条件做**清洁输入负对照**：这一条同时服务 P3 的 Clean-input 与真实数据保真评价。

**验收**：`artifacts` 与 GT 的重建误差可作为"分量真值一致性"基准报告出来；`Brain` 条件下 bypass 率与修改率达标。

### P0-2　查清并修掉 PhysioMotion 真实域失效

**理由**：真实数据 bypass = 0.0000 是当前最刺眼的失败，且结论"模型不能迁移"目前**证据不干净**——输入尺度不一致（§3.3-4）、单通道截取、clean 窗口本身可能含未标注伪迹，三个因素混在一起。

**具体动作**
1. 统一真实数据集接口：`physiomotion.py` 与 `on004784.py` 采用同一套稳健标准化与 metadata 填法。
2. 逐层归因：分别在（a）原始幅度、（b）稳健标准化后、（c）训练集固定统计量下重跑外测，分离"尺度问题"与"真实域偏移"。
3. 核实 `open_base/close_base` 是否真的可视为 clean（`physiomotion.py:36`）；若不可，则"clean 窗口被误激活"这一结论要改写。
4. 补上 `PhysioMotion` 本仓副本的 `derivatives`（本仓缺，脚本却默认指向外部绝对路径），让实验不依赖硬编码路径。

### P0-3　正面对付 ocular_drift 坍缩

**理由**：cardiac/myogenic/motion 三个专家被 ocular_drift 吸走（0.445–0.768），会同时毁掉 §11 的"稀疏专家优于共享分支"和 §7.3 的专属性诊断。这是**架构级**问题，比补指标更优先。

**具体动作**
1. 先做最小诊断：固定 `route_mode="all"` 看三个专家是否本来就不具备可分特征。若 `all` 模式下同样混淆 → 是特征/标签问题（心电只有自相关寻峰、运动只有差分，参考 `model.py:153-162`，建议补峰度、变化点、QRS 模板匹配）；若 `all` 正常、仅 learned 混淆 → 是路由监督问题。
2. 补充标注错误来源：MIT-BIH 是 ECG 源而非真实 EEG 中的心电投影（`data/DATASETS.md:16`），若注入方式使心电与低频眼动在合成样本里高度相关，路由无从区分——需检查 `synthetic.py:188-202` 的注入与 `_scale_to_snr` 的幅度分布。
3. 把「类别 × 专家激活矩阵的对角优势」加入常规报告（`first_experiment.py:228-231` 已产出矩阵，只差判定阈值）。

### P1　填平"有数据但无代码"的鸿沟

**理由**：约 190 GB 数据零接入，设计文档 P3 的下游任务（运动想象、睡眠分期、情感识别）与生理负对照（棘波、纺锤波、K 复合）**全部**依赖它们。这是论文主张的必要条件。

**具体动作（按依赖顺序）**
1. 先建统一 loader 骨架 + **受试者级分组器**（P1 清单第一条，同时也是 §4.2 防泄漏的硬要求）。当前 on004784/PhysioMotion 连基础 train/val/test 划分都没有。
2. 运动想象下游：`bci2a`（9 受试者，完整）→ 成本最低，建议作为第一个下游任务。
3. 生理负对照：`chbmit`（棘波，注意**子集**）与 `sleep_edfx` / `cap_sleep`（睡眠事件）——先核验事件标注质量，DATASETS.md 已提醒"事件标签质量需逐库核验"。
4. `ds002094` 用于 Unknown 压力测试（TMS 瞬态）；`faced` 做情感下游，但**预处理版与原始 BDF 必须分开报告**（DATASETS.md 的边界要求）。
5. `tuar` 为空：要么补齐，要么在后续文档中持续声明未接入。

### P2　补齐论文级证据

- 频带/事件保持损失接口（`losses.py` 完全缺 `L_preserve`）；
- 综合验证选模（`train.py:140` 只看 `val_loss`，违反 §6 阶段 D "不以单一 MSE 选模"）；
- ≥3 随机种子 + 跨种子分量匹配（§7.2）；
- 专家槽位干预实验（§7.1，`CoarseDecoder` 的六头已具备整体干预条件）；
- 外部基线按 §10.1 预注册；
- Pareto 分析（去除率 vs 失真度）。

### P3　tACS 专项 / 自监督适配 / 流式部署

§14 的 P4/P5/P6 三块**整体为零**。建议在 P0–P2 有结论前不要启动——尤其自监督，§4.5 明确要求先验证假设，而 §11 已把"恒等退化"列为否定证据；流式在单通道主线都未稳固时序缓存收益不明。

---

## 7. 需要你决策的三个问题

1. **多通道要不要现在做？** on004784 有 128 导，心电/眼动的可分性主要来自空间投影（§2.2、§3.4.4）。若目标是心脏与运动伪迹的论文级结论，单通道路线可能走不通；但改多通道会牵动 stem、观测抽取、粗分解头与全部已跑实验。**建议：先在单通道上完成 P0-1/P0-2 把真实域问题定位清楚，再决定是否切多通道。**

2. **PhysioMotion 用本仓副本还是外部池？** 本仓副本缺 `derivatives`，外部池（21 GB）完整但属于另一个项目的目录。**建议：把本仓副本补全（或明确改为只读引用外部路径），避免"脚本默认路径指向项目外"这种隐性依赖。**

3. **`data/raw/on004784` 的命名与校验口径。** 目录名是 `on004784`，实际 OpenNeuro ID 是 `ds004784`（下载器 `download_on004784.py:15` 用的是 `ds004784/`）。另外它没生成 `download_manifest.json`，目前只有"大小比对"结论。**建议：统一命名，并补一份含大小（及可选 SHA-256）的 manifest，与 EEGdenoiseNet/MIT-BIH 保持同等校验强度。**

---

## 附：本次评估的证据边界

- 未执行训练或推理（仅静态审查 + 文件系统核验），故 §4.4 的数值全部引自 `runs/` 既有产物，非本次复现。
- 未核验外部池各库的**标签语义**（如 CHB-MIT 发作标注、Sleep-EDF 分期标注），仅核验了文件数量与扩展名。
- 各外部库的"子集/完整"判定基于文件计数与常见发布规模的比对，未逐库下载官方清单核对；若需精确结论应逐库比对官方 manifest。
- `D:\codexwork\eeg` 下的旧项目代码（`src/`、`configs/metacorrseg_v2.yaml` 等）未纳入本评估——若该项目的结论会进入同一篇论文，需要单独审查其与 UniCORE-EEG 的口径一致性。
