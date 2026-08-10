# KPConvX 跨数据集训练实验汇总

本报告按“训练实验”组织结果，重点记录模型配置、训练状态、随机种子、最佳轮次和最佳指标。报告基于截至 2026-08-07 可核对的本地训练参数、训练日志、验证日志和 checkpoint 测试记录；不是新的训练运行。

## 先看结论

| 数据集 | 训练实验 | 训练状态 | Seed | 训练预算 | 训练监控选定的最佳验证 checkpoint | 已完成评估中的最佳结果 |
|---|---|---|---:|---:|---|---|
| S3DIS Area 5 | KPConvX-L + Grid | 完成 | 57106803 | 450 epochs | epoch 348，mIoU 75.7385% | epoch 450，10-vote full-cloud mIoU 71.4% |
| S3DIS Area 5 | KPConvX-L + Grid + FastAdapter | 中断 | 57106803 | 计划 450，实际至 epoch 2 | epoch 2，验证 mIoU 38.98% | 未测试 |
| ScanObjectNN main split | KPConvX-L + Grid | 完成 | 57106803 | 250 epochs | 未记录可用的训练验证 mIoU | epoch 199，10-vote OA 89.3%，mAcc 88.2% |
| ScanObjectNN main split | KPConvX-L + Grid + FastAdapter | 完成 | 57106803 | 250 epochs | 未记录可用的训练验证 mIoU | epoch 190，10-vote OA 88.7%，mAcc 87.3% |
| ScanObjectNN main split | KPConvD-L + Grid | 完成 | 57106803 | 250 epochs | 未记录可用的训练验证 mIoU | epoch 249/250，10-vote OA 88.9%，mAcc 87.3% |
| ScanObjectNN main split | KPConvD-L + Grid + FastAdapter | 完成 | 57106803 | 250 epochs | 未记录可用的训练验证 mIoU | epoch 250，10-vote OA 89.1%，mAcc 87.8% |

“最佳结果”必须结合指标来源理解：S3DIS 表中是 checkpoint 监控器选定的验证 checkpoint；ScanObjectNN 表中的最佳结果来自对多个 checkpoint 进行 10-vote 测试后的最高观测值，不是独立验证集选出的结果，因此存在测试集 checkpoint-selection bias。

## 配置说明

### S3DIS Area 5

两条 S3DIS 运行使用相同的 KPConvX-L 主体和训练协议，差异是是否启用 FastAdapter。

| 设置 | KPConvX-L + Grid | KPConvX-L + Grid + FastAdapter |
|---|---:|---:|
| Grid pooling | 开启 | 开启 |
| FastAdapter | 关闭 | joint training，开启 cross-layer 与 spatial attention |
| 计划训练轮次 | 450 | 450 |
| 每轮 steps | 300 | 300 |
| batch size | 4 | 4 |
| gradient accumulation | 6 | 6 |
| 有效 batch size | 24 | 24 |
| optimizer | AdamW | AdamW |
| 初始学习率 | 0.0001 | 0.0001 |
| weight decay | 0.05 | 0.05 |
| checkpoint 间隔 | 90 epochs | 90 epochs |
| 测试 votes | 10 | 尚未测试 |

FastAdapter 的 S3DIS 参数为：100 个 FPS anchors、geometry dimension 16、attention dimension 64、4 个 heads、chunk size 16384，cross-layer 和 spatial attention 均开启。

### ScanObjectNN main split

四条 ScanObjectNN 运行使用相同的 250-epoch 训练预算和优化器设置。FastAdapter 运行额外使用 64 个 FPS anchors、geometry dimension 16、attention dimension 64、4 个 heads、chunk size 4096，cross-layer 和 spatial attention 均开启。

| 设置 | 值 |
|---|---:|
| 数据划分 | main split |
| 计划训练轮次 | 250 |
| batch size | 32 |
| gradient accumulation | 2 |
| 有效 batch size | 64 |
| optimizer | AdamW |
| 初始学习率 | 0.0005 |
| weight decay | 0.01 |
| checkpoint 间隔 | 50 epochs |
| 测试 votes | 10 |

## 各训练实验的最佳记录

### 1. S3DIS：KPConvX-L + Grid

- 训练目录：`S3DIS_KPConvX-L-4090D-24G`
- 数据与划分：S3DIS，训练 Areas 1-4、6，Area 5 作为验证/评估区域
- Seed：`57106803`
- 状态：完成；训练曾中断并从 checkpoint 恢复
- 监控器选定的最佳 checkpoint：`best_mIoU_chkp.tar`
- 监控器选定轮次：epoch `348`
- 对应验证 mIoU：`75.7385%`
- 该 checkpoint 的 10-vote 结果：sub-cloud mIoU `70.9%`，full-cloud mIoU `71.1%`
- 最终 checkpoint：epoch `450`，10-vote full-cloud mIoU `71.4%`

这里的 epoch 348 是 `checkpoint_selection.csv` 记录的最佳 checkpoint；epoch 450 的 full-cloud 测试值更高，但它不是训练监控器选定的验证 checkpoint。Area 5 同时参与 checkpoint 选择和报告测试，因此这两个结果都不属于严格独立测试。

### 2. S3DIS：KPConvX-L + Grid + FastAdapter

- 训练目录：`S3DIS_KPConvX-L-G4-FA-seed57106803-4080S-32G-20260731T150900Z`
- 数据与划分：S3DIS Area 5
- Seed：`57106803`
- 计划训练轮次：`450`
- 实际状态：中断于内部 epoch `2`
- 当前已记录的最佳验证轮次：epoch `2`
- 当前已记录的最佳验证 mIoU：`38.98%`
- 正式测试结果：无

另外的 preflight 运行只完成 epoch `1`，验证 mIoU 为 `34.28%`，不作为正式训练结果。由于 FastAdapter 运行没有完成训练，也没有测试 checkpoint，不能据此判断 FastAdapter 相对于 Grid baseline 的最终收益。

### 3. ScanObjectNN：KPConvX-L + Grid

- 训练目录：`ScanObjectNN_KPConvX-L-official-seed57106803-4080S-32G`
- 数据与划分：ScanObjectNN `main_split`
- Seed：`57106803`
- 状态：完成 250 epochs
- 测试中观测到的最佳 checkpoint：`chkp_0200.tar`
- 对应内部 epoch：`199`
- 10-vote OA：`89.3%`
- 10-vote mAcc：`88.2%`
- 吞吐：`243.2 instances/s`

这里的“最佳”是 42 个候选 checkpoint 的 10-vote 测试最高值，不是训练验证最佳值；因此不能把 epoch 199 称为无偏的训练选择轮次。

### 4. ScanObjectNN：KPConvX-L + Grid + FastAdapter

- 训练目录：`ScanObjectNN_KPConvX-L-FastAdapter-seed57106803-4090D-24G`
- 数据与划分：ScanObjectNN `main_split`
- Seed：`57106803`
- 状态：完成 250 epochs
- 测试中观测到的最佳 checkpoint：`chkp_exact_0190.tar`
- 对应内部 epoch：`190`
- 10-vote OA：`88.7%`
- 10-vote mAcc：`87.3%`
- 吞吐：`110.6 instances/s`

该运行在训练早期发生过一次 CUDA out-of-memory，并自动降低 batch limit 后继续运行；因此报告保留这一训练过程异常，不能把它视为与 baseline 完全无条件一致的训练轨迹。

### 5. ScanObjectNN：KPConvD 对照组

为保留原跨数据集实验范围，报告同时记录两个 KPConvD 对照组：

| 训练实验 | Seed | 状态 | 最佳测试轮次 | 10-vote OA | 10-vote mAcc | 吞吐 |
|---|---:|---|---:|---:|---:|---:|
| KPConvD-L + Grid | 57106803 | 完成 250 epochs | 249/250 | 88.9% | 87.3% | 437.9/437.4 instances/s |
| KPConvD-L + Grid + FastAdapter | 57106803 | 完成 250 epochs | 250 | 89.1% | 87.8% | 135.1 instances/s |

## 结果文件和来源

- [metrics.csv](metrics.csv)：逐 checkpoint 的机器可读评估明细。
- [checkpoint_inventory.csv](checkpoint_inventory.csv)：checkpoint 文件名、内部轮次和训练/评估状态。
- S3DIS 训练汇总：[s3dis/kpconvx-l-area5-seed57106803/README.md](../s3dis/kpconvx-l-area5-seed57106803/README.md)
- ScanObjectNN checkpoint 测试汇总：[scanobjectnn/full-checkpoint-study-seed57106803/README.md](../scanobjectnn/full-checkpoint-study-seed57106803/README.md)

训练原始日志和 checkpoint 只保留在本地 `Standalone/KPConvX/results/`，不纳入版本化报告。

## 可复现性和限制

- 所有这里列出的主要运行都是单 seed `57106803`，不能代表多 seed 均值或置信区间。
- 运行期间的精确 Git commit 没有记录，不能用发布本报告时的 commit 代替。
- S3DIS Area 5 同时用于验证和测试，存在 checkpoint 选择偏差。
- ScanObjectNN 的最佳 checkpoint 是在测试多个候选 checkpoint 后选出，存在更明显的测试集选择偏差。
- S3DIS 原始 `val_IoUs.txt` 的第 172 行按三位小数行均值计算约为 75.7769%，高于 checkpoint 选择记录中的 75.7385%；该行与 checkpoint 的轮次映射目前未能独立确认，因此本报告以 `checkpoint_selection.csv` 的 epoch 348 记录为准，不把 75.7769% 宣称为 checkpoint-backed 最佳结果。
- S3DIS FastAdapter 训练未完成；其 epoch 2 的验证值只是当前已记录值，不是完整 450 epochs 的最佳结果。
- 不同实验的训练硬件标签不同；ScanObjectNN 的吞吐均在 NVIDIA GeForce RTX 4080 SUPER 测试流程中测得，不能直接当作训练速度比较。
