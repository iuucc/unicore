# on004784 数据接入 Sanity Check

## 数据审计

- 根目录：`D:\codexwork\unicore\data\raw\on004784`
- 任务条件：Brain, Eyes, Facial, Neck, Walking, All
- Ground truth 文件：`data\raw\on004784\stimuli\GTdata_croppedToRisingEdge.mat`
- Ground truth 形状：`(153600, 21)`，来源通道组：`{'brain': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'ocular': [10, 11], 'neck_myogenic': [12, 13, 14, 15], 'facial_myogenic': [16, 17, 18, 19], 'trigger': [20]}`

## BIDS 记录

| task | family | labels | sfreq | channels | eeg_channels | samples | duration_s |
|---|---|---|---:|---:|---:|---:|---:|
| Brain | clean | clean | 512.0 | 264 | 128 | 182784 | 357.0 |
| Eyes | ocular | ocular_drift | 512.0 | 264 | 128 | 162304 | 317.0 |
| Facial | facial_myogenic | myogenic | 512.0 | 264 | 128 | 160256 | 313.0 |
| Neck | neck_myogenic | myogenic | 512.0 | 264 | 128 | 163840 | 320.0 |
| Walking | motion | motion_transient | 512.0 | 264 | 128 | 168448 | 329.0 |
| All | mixed | ocular_drift, myogenic, motion_transient | 512.0 | 264 | 128 | 159232 | 311.0 |

## 窗口样本检查

- 窗口长度：`1000` 点；目标采样率：`500` Hz；通道数：`1`
- 总窗口数：`971`

| task | shape | finite | mean | std | labels |
|---|---|---:|---:|---:|---|
| Brain | `(1, 1000)` | True | -0.0321 | 0.9526 | clean |
| Eyes | `(1, 1000)` | True | -0.0008 | 0.8259 | ocular_drift |
| Facial | `(1, 1000)` | True | -0.0544 | 0.8695 | myogenic |
| Neck | `(1, 1000)` | True | 0.0254 | 0.8614 | myogenic |
| Walking | `(1, 1000)` | True | 0.0458 | 1.0222 | motion_transient |
| All | `(1, 1000)` | True | 0.0002 | 1.0289 | ocular_drift, myogenic, motion_transient |

## 结论

- `.set/.fdt` 可由 `mne` 正常读取。
- 六类条件已映射到 UniCORE 六专家标签空间，其中 Facial 与 Neck 都属于 `myogenic`，但保留不同 family 便于后续分层统计。
- 当前检查只验证数据读取、标签映射、窗口切片和张量数值稳定性；暂不启动训练。
