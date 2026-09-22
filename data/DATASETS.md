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
