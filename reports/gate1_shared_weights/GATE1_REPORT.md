# Gate 1：共享权重跨通道架构验证

## 结论

**Gate 1：FAIL，暂不进入阶段 2。**

共享权重的结构、跨通道前向/反向、混合 C padding、缺导屏蔽和通道置换等机制验证通过；C=8 短训练闭环通过。但当前架构的 C=1 完整回归中，两个 macro-F1 指标相对历史基线下降超过 0.03，因此不能把 Gate 1 判为通过。

## 验证范围与实际运行

本分支为 `codex/gate1-shared-weights`。本次任务先运行完整单元测试，再在当前 CPU-only 环境运行多通道 smoke、C=128 batch=1 smoke、C=8 两 epoch 短训练和 4 项定向变量通道测试。

完整 C=1 回归的轻量指标来自仓库中已存在的真实运行记录 `runs/gate1_shared_weights/regression_c1_seed42/`：该运行于 2026-09-22 使用共享权重架构提交 `b746c431` 完成，12 epoch、65,536 train samples、6,000 eval samples、seed 42。当前主机只有 CPU，未声称在本次任务中重新完成这项长回归；原始 checkpoint 和逐样本结果仍留在本地且未提交。

## Gate 清单

| 门禁项 | 结果 | 证据 |
|---|---|---|
| 100 项完整单元测试 | PASS | `100 passed in 74.51s` |
| 同一模型实例处理 C=1/3/8/34/64 | PASS | `smoke_multichannel.json`，输出 shape 正确，参数量均 15,388,896 |
| C=128 前向 + 反向（batch=1） | PASS | `smoke_multichannel.json`，参数量 15,388,896 |
| 混合 C batch、padding mask、masked loss | PASS | `mixed_channel_tests.json` |
| 缺导通道 mask | PASS | `mixed_channel_tests.json` |
| 通道置换等变 | PASS | `mixed_channel_tests.json` |
| C=8 两 epoch 训练闭环 | PASS（流程） | `c8_summary.json`；样本量仅 32/16，不作论文结论 |
| C=1 macro-AUROC 回归差异 < 0.03 | PASS | 0.8987 → 0.9112，差异 +0.0125 |
| C=1 macro-F1 @ 0.5 差异 < 0.03 | **FAIL** | 0.5683 → 0.5135，差异 −0.0547 |
| C=1 macro-F1 @ route threshold 差异 < 0.03 | **FAIL** | 0.5167 → 0.4633，差异 −0.0533 |

## 主要指标

### 当前任务真实运行的 smoke 与短训练

- 共享模型参数量：15,388,896；C=1、3、8、34、64、128 均保持不变。
- C=1/3/8/34/64 batch=2：前向、损失、反向全部通过。
- C=128 batch=1：前向、损失、反向通过；当前环境为 CPU，单步约 10.73 秒。
- C=8 短训练：2 epoch，32 train samples，16 eval samples，batch=8，最终训练损失约 4.016。
- C=8 短验证路由指标：macro-AUROC 0.6571，macro-F1@0.5 0.0937，macro-F1@route threshold 0.2412。该验证集很小，只用于流程诊断。

### C=1 完整回归比较

详见 `c1_regression_comparison.csv`。完整回归的路由检测指标为：macro-AUROC 0.9112、artifact-presence AUROC 0.9922、unknown-detection AUROC 0.9993、clean 上 Unknown 误激活率 0.0038、known 上 Unknown 误激活率 0.0454。两个 macro-F1 回归门槛失败。

## 路由实现与设计差异

设计文档 §3.5 用 `max(p_k)` 作为 clean-bypass 的 presence 判据：当 `max_k p_k < δ_p` 时旁路全部专家。当前代码在 `ArtifactTokenizer` 中增加了独立的 `artifact_presence` 线性头，输出 `artifact_presence_probability = sigmoid(artifact_presence_logit)`；`SparseRouter` 优先用这一独立 presence 概率判定 bypass，而不是直接取 `probabilities.max(dim=-1)`。这保留了一个可学习的总体存在性估计，但会使 clean-bypass 判据与各专家的 `p_k` 脱钩，是本次回归中需要重点校准和诊断的架构差异。

## 配置、环境与复现

- 当前任务分支 SHA：`9dc527dd228716f5c583e84fc275cd02e8812ac8`（代码提交后报告提交会再更新）。
- 当前环境：Windows 11，Python 3.13.9，PyTorch 2.14.0+cpu，CUDA 不可用。
- 完整测试：`KMP_DUPLICATE_LIB_OK=TRUE python -m pytest -q`
- 多通道 smoke：见 `c8_run_manifest.json` 与 `smoke_multichannel.json` 同目录原始运行记录。
- C=8 短训练命令和实际生效配置：见 `c8_run_manifest.json` 与 `effective_config.yaml`。
- 定向混合 C 测试命令：见 `mixed_channel_tests.json`。

## 未提交内容与限制

- 原始训练目录：`runs/gate1_shared_weights/`。
- 大型 checkpoint：`runs/gate1_shared_weights/regression_c1_seed42/*.pt`、`runs/gate1_shared_weights/smoke_multichannel_c8_bs8/*.pt` 等，继续保持忽略。
- 逐样本输出：`sample_metrics.csv` 等大型文件，继续保持忽略。
- 原始 EEG 数据：`data/raw/`，未提交。
- 当前 CPU-only 主机未重新跑 12 epoch C=1 完整回归；归档表中的 C=1 指标保留真实来源和运行 SHA，不冒充本次短运行。

## 后续建议

1. 先在验证集预注册并执行 presence / route threshold 校准，不查看测试集调参。
2. 重新封存测试集评估 macro-F1 两种口径和 clean-bypass 误激活率。
3. 若 F1 差异仍超过 0.03，定位独立 `presence_head` 与 `max(p_k)` 差异、类别混淆和阈值策略，再重训。
4. Gate 1 通过前，不开始阶段 2 或正式三种子训练。
