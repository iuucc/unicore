# 阶段 1：多通道能力落地（决策 D1）执行记录

对应手册 §8「阶段 1」与 Gate 1。本文件记录每个任务执行的命令、实测输出、遇到的问题与下一步。

---

## 任务清单

| 任务 | 内容 | 状态 |
|---|---|---|
| T1.1 | montage 与坐标表 | ✅ 完成 |
| T1.2 | 观测抽取器多通道化 | ✅ 代码与单测完成；单通道回归运行中 |
| T1.3 | 新增 `SpatialProjectionHead` | ✅ 完成 |
| T1.4 | 合成器真混音改造 | ⏳ |
| T1.5 | 多通道冒烟与单通道回归 | ⏳ |

---

## T1.1　montage 与坐标表

**产出**：`unicore_eeg/montage.py`、`configs/montages/*.yaml`（13 个）、`scripts/build_montages.py`

**命令**

```bash
python scripts/build_montages.py --all        # 生成全部配置
python scripts/build_montages.py --report     # 逐 montage 解析覆盖率自查
```

**实测覆盖率（12 个数据集 montage 全部 100%）**

| montage | resolve | 通道 | 解析 | 双极 | 说明 |
|---|---|---:|---:|---:|---|
| `bci2a` | positional | 22 | 22 | 0 | EDF 通道名匿名，按 `raw_layout` + 锚点校验映射 |
| `bci2b` | names | 3 | 3 | 0 | 实测是标准位 `C3/Cz/C4`，无第二电极 |
| `cap_sleep` | names | 13 | 13 | 13 | 原记录 18 路，其余 5 路是 ROC/EMG/ECG/DX/SX |
| `chbmit` | names | 23 | 23 | 23 | `T8-P8-0/-1` 剥末位序号后落到同一中点 |
| `cho_gigadb` | names | 64 | 64 | 0 | 末两路 `FCz_ref`/`Oz_ref` 是参考通道 |
| `ds002094` | names | 30 | 30 | 0 | `IZ→Iz`、`TP9/TP10` 走别名 |
| `ds004784` | names | 128 | 128 | 0 | 自带坐标表（单位球，半径恰好 1） |
| `faced` | names | 32 | 32 | 0 | 含 `A1/A2` 乳突位 |
| `openbmi` | names | 62 | 62 | 0 | 6 个 10-5 名字由 `standard_1005` 补齐 |
| `physiomotion` | names | 34 | 34 | 34 | 与旧项目常量逐字一致 |
| `physionet_mi` | names | 64 | 64 | 0 | 缓存 21 路是前 21 个 |
| `sleep_edfx` | names | 2 | 2 | 2 | `EEG Fpz-Cz` / `EEG Pz-Oz` |
| `standard_reference` | — | 100 | 100 | 0 | 参考表本身也是合法 montage |

**验收对照（手册 T1.1）**

- 单测：每个 montage 的 `resolve_montage` 返回通道数与配置一致、`mask` 全 True → ✅
  （`test_every_montage_resolves_fully`，含 `channel_count` 与 `channels` 长度的一致性检查）
- 单测：不存在的通道名 → `mask=False`、坐标置零、不抛异常 → ✅（`test_unknown_channel_is_masked_without_raising`）

**四条与手册不同的处置（均已登记附录 C）**

1. **`ds004784_19ch.yaml` 放弃**。实测 128 电极是均匀铺满球冠的阵列（半径恒 56.0 mm、
   对 y 严格镜像对称），不是 10-20 头皮帽。严格 ICP（60 次迭代 × 8 种初始朝向）后，
   标准 19 导到最近电极的平均距离仍有 **0.1485 半径 ≈ 8.3 mm**，且随机旋转对照显示
   该匹配几乎无判别力。经用户确认放弃该文件。
2. **`bci2b` 不做中点**。手册写"3 双极"，但原始通道是无第二电极的标准位名。
3. **`cap_sleep` / `sleep_edfx` 拆出 aux 通道**。分类按"能否解析出坐标"判定，
   而不是"名字里有没有 `-`"——后者会把 `ROC-LOC`、`ECG1-ECG2` 误判成 EEG 双极
   （第一版实现踩过这个坑，已加 `_split_by_resolvability`）。
4. **新增 `frame_axes`**。空间先验需要知道"前/上"是哪个轴；参考表是 MNE RAS，
   ds004784 必须显式声明 `anterior=(1,0,0)`。

**实测产出的额外事实**

- 跨半球双极导联的中点会落近头心：`C4-A1` 半径 **0.2104**、`FT9-FT10` **0.5412**。
  保留手册规定的中点规则，但 `coverage_report` 的 `low_information` 字段把它们如实列出，
  不伪装成正常位置。
- `cho_gigadb` 自带的 `eeg.psenloc`（64×3 单位球）**被拒绝使用**：已知通道对应下做
  Kabsch 刚性对齐，最优平均弦长 0.90（完全随机 ≈ 1.27），说明其数组顺序与
  `CHO_EEG_CHANNELS` 不对应或不是头部坐标系。改用按名解析，零近似。

---

## T1.2　观测抽取器多通道化

**目标**：消除 §6.2 的三处压维（谐波基 `expand_as`、`mono = y.mean(dim=1)`、
`fitted.unsqueeze(1).expand_as(y)`）。

**改造结果**：`ArtifactObservationExtractor` 的输入输出契约改为
输入 `(B, C, T)` → 6 个 `(B, C, T)`，顺序不变。逐通道化的部分：

- 谐波基：频率仍共享（全通道平均谱找峰），**幅相系数逐通道独立求解**
- 自相关/心电：**逐通道自相关寻峰**得 `(B, C)` 滞后，`periodic` 逐通道构造
  （用 `gather` 一次算完，替掉原先对 batch 的 Python 循环）
- 其余四路本来就是逐通道，保持不变

**验收对照（手册 T1.2）**

| 验收项 | 结果 |
|---|---|
| C=1 与 C=4 前向成功、6 个输出 shape 均为 `(B,C,T)` | ✅ `test_output_shapes_are_per_channel` |
| C=4 时 `obs[0][:,0] != obs[0][:,1]` | ✅ `test_channels_are_no_longer_identical`（6 路逐项断言） |
| 左通道 0.8 s / 右通道 1.2 s 周期 → `cardiac_lags` 不同 | ✅ `test_cardiac_lags_are_per_channel`（并断言滞后值落在 400±15 / 600±20） |
| 单通道回归 `macro_auroc` 与基线差 < 0.03 | ✅ 见下表，差 **0.00194** |

**单通道回归结果**（`runs/after_obs_multichannel_seed42`，12 epoch 全量，墙钟 1152 s）

| 指标 | T0.5 基线(nw=0) | T0.7 同口径(nw=8) | T1.2 本次 | vs T0.7 | vs T0.5 |
|---|---:|---:|---:|---:|---:|
| macro_auroc | 0.8987 | 0.8969 | **0.8968** | 0.00017 | **0.00194** |
| macro_f1_at_0_5 | 0.5683 | 0.5626 | 0.5634 | 0.00083 | 0.00489 |
| macro_f1_at_route_threshold | 0.5167 | 0.5109 | 0.5114 | 0.00045 | 0.00529 |
| artifact_presence_auroc | 0.9937 | 0.9939 | 0.9939 | 0.00002 | 0.00015 |
| unknown_detection_auroc | 0.9997 | 0.9998 | 0.9998 | 0.00001 | 0.00010 |
| unknown_false_activation_clean | 0.0020 | 0.0024 | 0.0024 | 0.00002 | 0.00042 |
| unknown_false_activation_known | 0.0468 | 0.0531 | 0.0529 | 0.00028 | 0.00603 |

**与同口径对照（T0.7，唯一差别是抽取器）的最大差只有 0.00083** —— 说明 T1.2 对 C=1 基本是中性的，
这正是期望的结果（C=1 时"逐通道"与"单通道"本来就该等价）。相对 T0.5 基线的最大差 0.00603
来自 T0.7 的 num_workers 改动，不是 T1.2 引入的。

**过程中修掉的三个真缺陷**

1. **`torch.linalg.lstsq` 去掉 `.float()` 导致 bf16 崩溃**（第一次回归在第 0 个 batch 就挂）。
   根因不是 dtype 而是 **autocast 会把 `@` 降精度**：即使输入是 float32，autocast 也会把
   matmul 降到 bf16，随后 `torch.linalg.solve` 报
   `NotImplementedError: "lu_factor_cublas" not implemented for 'BFloat16'`。
   修法：抽取器**显式退出 autocast**（`model.no_autocast`），内部统一 float32、出入口转回输入 dtype。
   附带发现 CPU 的 `fft.rfft` 也不支持 bf16，所以"统一 float32"同时消除了跨设备差异。
2. **`run_dual_gpu.py` 的 GPU 采样线程覆盖了 `threading.Thread._stop` 方法**，
   `Thread.join()` 抛 `TypeError: 'Event' object is not callable`——任务失败时报告环节先崩，
   把真实的 bf16 错误盖住了。事件改名 `self._halt`。
3. **谐波拟合改为带 1e-6 岭项的正规方程**（解 6×6 而非 lstsq），并用
   `test_harmonic_fit_equals_direct_lstsq` 断言与原 lstsq 结果一致（容差 1e-3）。

**未预期的性能收益（待归因）**

同 stage、同 `num_workers=8`、同规模下，逐 epoch 墙钟由 T0.7 对照的
~178–199 s（decomposition 2–6）降到 **~94–101 s**。两次运行的唯一差异是 T1.2 的抽取器改造，
最可能的来源是替掉了 `torch.linalg.lstsq`。**该归因尚未做受控验证**，只作为观察记录；
指标对照才是本次运行的验收依据。

---

## T1.3　新增空间投影模块

**产出**：`model.py::SpatialProjectionHead`，接在观测抽取之后、专家调用之前。

**结构**（任意 C 可前向，参数量与 C 无关）

1. 空间主成分时序：`rank=4`；默认用**确定性幂迭代**（`spatial_pca_mode="power"`，4 次迭代），
   精确 SVD 路径保留为 `"svd"` 供复核。
2. 逐通道载荷 `(B, C, rank)`：通道时序与主成分的相关（按 RMS 归一）。
3. 逐通道特征 = 投影坐标(2) + RMS(1) + 有坐标标志(1) + 载荷(4) = **8 维，与 C 无关**。
4. DeepSets：逐通道 MLP `8→32→32` → 注意力池化得全局上下文 `(B,32)`。
5. 输出 `w_s: (B, C, 1)`，调制 `obs_k ← obs_k · (1 + α·w_s)`，仅 k ∈ {1 ocular, 3 cardiac}。
   α = `spatial_modulation_alpha_scale · tanh(α_raw)`，`α_raw` 初值 0 → **初始化时模块是恒等映射**。

**两处与 §6.4 字面的差异（已登记附录 C）**

1. §6.4 步骤 2 算出了逐通道载荷却没在后续步骤使用；这里把它作为 MLP 的输入特征之一
   （"空间投影一致性"的载体就是它）。
2. §6.4 说 SVD；`C=128` 时每 batch 要 64 次 128×1000 分解，代价过大，默认改用幂迭代，
   并用 `test_power_iteration_matches_exact_svd` 断言两条路径的子空间一致
   （主角度余弦 > 0.98）。

**验收对照（手册 T1.3 与 Gate 1）**

| 验收项 | 结果 |
|---|---|
| `C ∈ {1,2,3,34,64,128}` 前向 + 反向成功、无 NaN | ✅ `test_model_forward_and_backward_all_counts` |
| `count_parameters` 与 C 无关 | ⚠️ 按 **Gate 1** 的措辞判：`SpatialProjectionHead` 参数量恒为 **1,444** ✅。**整模型**参数量本就随 C 变化（`ArtifactTokenizer` 的 `stat_dim = 3C` 进 `fuse`，加上首尾 `Conv1d(in_channels, …)`），C=2→128 差 **771,120**，这是改造前就存在的事实；已用 `test_whole_model_parameter_growth_is_explained` 把它钉成显式记录 |
| `use_spatial=False` 与未加模块时输出一致 | ✅ `test_alpha_zero_is_identity`（α=0 时逐键一致；关闭时返回字典不多出空间相关键） |
| 只有 1/3 两路观测被调制 | ✅ `test_modulation_reaches_experts_for_the_right_observations`（抓取各专家实际收到的观测逐项比对） |

**过程中修掉的缺陷**

- **正交化没有逐列归一化** → 参考向量范数指数放大（4 列就从 337 涨到 3.4e22），
  归一化时除以 inf 得到 NaN。C=1/2/3 时列数少、恰好看不出问题，C=34 才暴露。
  已改为"投影完立刻归一化 + 第二遍精修"，并加 `test_orthonormalize_is_numerically_stable`。
- C < rank 时载荷维度不足会让 MLP 输入维度随 C 变化，已用零列补齐。

---

## T1.4　合成器真混音改造

**产出**：`unicore_eeg/spatial_mixer.py`（新建）、`synthetic.py` 改造

**结构变化**

- 每个伪迹先作为**独立源** `s_k(t)`（单通道 `(T,)`）生成，再统一交给 `SpatialMixer`：
  按各伪迹的空间先验生成混音矩阵 `A ∈ R^{C×K}`、加上逐通道传播延迟（`U(±8 ms)`），
  得到**逐族**的多通道伪迹；最后按目标 SNR 缩放并叠加。
- 清洁源同样经空间混合（`mix_clean`）：拿两个独立节律源各混一次再相加，
  既保留通道间幅度/相位差异（来自混合），又保留各通道的节律内容（来自独立源），
  不是"同一个源复制到所有通道"。
- SNR 逻辑（`_scale_to_snr`）作用在**混音后的多通道伪迹**上（手册 T1.4 步骤 4）。

**样本字典新增**

| 字段 | 形状 | 含义 |
|---|---|---|
| `mixing_matrix` | `(6, C)` | **实际施加**到每个伪迹下标上的混音权重；未激活的族为全零行 |
| `coords` | `(C, 3)` | 本样本使用的电极坐标 |

两处与手册字面的差异（已登记附录 C）：

1. `coords` 存 **`(C, 3)`** 而非手册写的 `(C, 2)`。二维投影依赖坐标系朝向
   （`project_to_disk` 假设 y=前），而 ds004784 的体模坐标系不是 RAS；把投影结果
   固化进数据集会把一个错误假设变成既成事实。模型自己按前两个分量取用。
2. `mixing_matrix` 记的是"**实际施加**"而非"本次生成的 A 的全部列"。原因见下面第 1 条缺陷。

**过程中修掉的缺陷（一个会影响阶段 4 的真问题）**

1. **held-out 重定向导致混音矩阵记录失真**。`first_experiment` 模式下 condition 7–11
   会把第 0–4 族分别留出并把它们的能量重定向到下标 5（unknown）。若照抄本次生成的
   `A` 的第 5 列，记下的会是 **unknown 源那一列**，而实际施加到下标 5 的是**另一个族**
   的权重——阶段 4 的空间指标会拿错误的 A 去比对。
   处置：**①** 一个伪迹下标只允许收到一个源族的能量（held-out 族与 unknown 源互斥），
   语义也更干净："下标 5 = 某一族未知伪迹"，而不是"两族之和"；
   **②** 记录改写为 `applied[target_family] = matrix[:, family]`，未激活的族留全零行。
   判据用 `test_mixing_matrix_records_what_was_applied`（某下标有能量 ⟺ 该行非零）
   与 `test_held_out_family_prior_is_carried_into_the_unknown_row`
   （condition 8 留出 ocular 时，下标 5 的行必须仍是前额优势；condition 10 留出 cardiac 时仍是全通道同号）。
2. 稀疏先验（harmonic / unknown）与局域先验（myogenic）下，未参与的通道权重接近 0，
   "任意两个通道互不相同"这个判据不成立——那是先验的正确表现，不是复制。
   判据改为取**权重最大的两个通道**比较。

**验收对照（手册 T1.4）**

| 验收项 | 结果 |
|---|---|
| `C=8` 时 `artifacts[:, 1, 0] != artifacts[:, 1, 1]` | ✅ `test_artifact_components_differ_across_channels`（6 路逐项，按权重挑通道） |
| ocular 行在 Fp1/Fp2 类通道权重显著高于枕区 | ✅ `test_ocular_mixing_favours_frontal`（要求 > 5×，且要求至少 5 个激活样本） |
| cardiac 行近似低秩（各通道符号一致、通道间相关 > 0.8） | ✅ `test_cardiac_component_is_coherent` |
| `C=1` 退化、`runs/after_mixer_seed42` 与基线差 < 0.03 | ⏳ 回归运行中 |

---

## T1.5　多通道冒烟与单通道回归

**产出**：`scripts/smoke_multichannel.py`、`scripts/regress_multichannel.py`、`runs/_regress/regress_multichannel.md`

（待执行）

---

## Gate 1 GPU 重验证（2026-09-23）

- 环境：`D:\Anaconda_envs\envs\unicore-eeg\python.exe`，Python 3.11.16，torch 2.11.0+cu128，RTX 5080，CUDA_VISIBLE_DEVICES=0，bf16，单卡，无 DDP/torch.compile。
- 新增公共源池明确 `train/val/test=60/20/20` 源索引分割与交集单测；训练入口使用独立 train/validation 数据集和 seed。
- 新增 `scripts/calibrate_routing.py`：validation 选择逐类阈值并输出 precision/recall/F1/AUROC/AUPRC/support/FPR/FNR/ECE/Brier；test 只用冻结阈值运行一次。
- C=1 GPU 训练：4096 train、4 epochs；C=8 GPU 训练：1024 train、4 epochs。详细结果见 `reports/gate1_shared_weights/revalidation_20260923/`。
- Gate 1：**FAIL**。C=1 test known macro-F1=0.2864、macro-AUROC=0.6479；C=8 test known macro-F1=0.2849、macro-AUROC=0.6013。阶段 2 不得开始。
- 失败定位：harmonic/ocular 路由概率校准与判别能力不足；C=8 训练规模仍低于正式性能实验要求。

## 2026-09-23 v3 修订

- T1.4：✅ 真混音代码与单测完成；C=1 回归不再标记“运行中”。
- T1.5：⚠️ 多通道 smoke 已完成；正式 Gate 1 重验证因路由诊断 FAIL 停止，不能标记完成。
- 参数量口径修订：共享权重指专家/空间头不按 C 复制；整模型仍包含按输入通道数变化的 stem、统计融合等参数，因此整模型参数量随 C 变化。报告中“固定参数量”仅适用于共享专家主体，不能用于整模型。
- v3 固定 64 样本过拟合：C=1 与 C=8 均未达到每类 F1/AUROC ≥0.98，按停机规则未运行更大实验，Gate 1 继续 FAIL。

## 2026-09-23 v4 评估器修订

- 修复路由评估：逐类 precision/recall/F1/AUROC/AUPRC/FPR/FNR/ECE/Brier、validation 阈值校准与 `best.pt` 选择均按 `label_mask` 排除 held-out 已知类样本；Unknown 单独按真实 Unknown 目标评估。
- 修复 64 样本诊断：C=1 attention 参数因单通道 softmax 恒为 1 而无梯度，不再作为失败项；梯度审计改为 5 个 known 概率输出行与真正的 `unknown_detector`。
- v4 固定 64 样本结果：C=1 known macro-F1@0.5=0.6000、known macro-AUROC=0.9653、同 64 样本校准 known macro-F1=0.8288；C=8 known macro-F1@0.5=0.9200、known macro-AUROC=0.9959、同 64 样本校准 known macro-F1=0.9600；两者六头梯度均有限非零。
- 结论修订：v3 的“64 样本过拟合失败可指向模型结构问题”不成立；但 v4 仍未达到 known macro-F1 ≥0.98，Gate 1 继续 FAIL，Stage 2 仍不得开始。
- 新增证据目录：`reports/gate1_shared_weights_v4/`，包含逐样本 labels/label_mask/probabilities/logits/disabled_experts、10000 样本审计和简单分类 baseline。
