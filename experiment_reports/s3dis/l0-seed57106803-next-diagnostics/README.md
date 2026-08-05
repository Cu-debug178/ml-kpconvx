# S3DIS L0 结果判断、250 Epoch 方案与 Grid/FastAdapter 机制诊断

## 0. 报告来源与复现边界

- 数据集与划分：S3DIS，Area 5 验证/测试划分（来自用户提供的 checkpoint 汇总；本机未持有数据集核验）。
- 模型与配置：LitePT L0，encoder blocks `(3,3,9,12,3)`，heavy decoder，PointROPE 开启。
- Seed：`57106803`。
- 训练状态：用户材料报告训练至 450 epoch；存在 interrupted/resumed 训练与否未记录。
- 训练协议：450 epoch cyclic LR；其余训练参数以当次保存配置为准，本地未取得原始配置文件。
- 评估协议：full-room mIoU；投票次数、增强方式和逐类别指标未记录。
- Checkpoint 选择：在用户提供的已评估 checkpoint 中按最高 full-room mIoU 选择，存在 checkpoint-selection bias。
- 硬件、软件环境与资源用量：`not recorded`。
- 运行时 Git commit：`not recorded`；不得用当前仓库提交号事后替代。
- 数据来源：用户提供的 `S3DIS_L0_CHECKPOINT_SUMMARY.csv` 和 Pro 对话摘要；原始日志、checkpoint 与逐房间预测未在本机复核。
- 限制：这是单 seed 观察，不能据此证明其他配置或 seed 都会在 250 epoch 前收敛。

## 1. 对当前 L0 结果的判断

上传的 checkpoint 测试记录中，最高 full-room mIoU 为 **72.1%，出现在 epoch 210**。到 epoch 250 为止已经覆盖该最佳点；epoch 260–450 的最高值为 71.9%，没有刷新 epoch 210。150–250 区间均值约 71.46、标准差约 0.36；250 之后均值约 71.65、标准差约 0.16，表现为低学习率下的小幅稳定波动，而不是继续提升。

因此，对这一条 seed 的 L0 训练，250 epoch 足以覆盖目前最佳 checkpoint，后 200 epoch 的边际收益很低。但单个 seed 不能证明所有配置都应固定在 250 epoch。建议将 250 epoch 作为后续消融的统一预算，并保留 200、210、220、230、240、250 的 checkpoint；最终候选模型再用 3 个 seed 验证是否仍在 250 前达到平台。

## 2. 不能只把 450 截断为 250

原调度在 450 epoch 下，从峰值学习率开始每 120 epoch 衰减 10 倍。直接把 `max_epoch` 改成 250，会导致训练结束时学习率仍偏高。工程中提供的 250-epoch 脚本同时设置：

```text
max_epoch = 250
cyc_decrease10 = 62
checkpoint_start = 200
checkpoint_gap = 10
```

这样保持 30 epoch 上升和 5 epoch 峰值平台不变，并让 250 epoch 训练结束时的学习率与原 450 epoch 调度末端近似一致。它不是简单少训练 200 epoch，而是压缩退火阶段。

## 3. L0 后面的正确实验顺序

不要直接从 L0 跳到 L2。当前实验命名为：

| 实验 | Encoder blocks | Decoder | PointROPE | 目的 |
|---|---|---|---|---|
| L0 | `(3,3,9,12,3)` | heavy | 开 | 只替换后两级算子，保持 KPConvX-L 深度和 decoder |
| L0D | `(3,3,9,12,3)` | light | 开 | 单独测试轻量 decoder |
| L1 | `(2,2,2,6,2)` | light | 开 | 测试 LitePT-S 风格深度压缩 |
| L2 | `(2,2,2,6,2)` | light | 关 | 在 L1 上只消融 PointROPE |

所以桥接关系是：

```text
L0 -> L0D：改变 decoder
L0D -> L1：只改变 block 深度
L1 -> L2：只关闭 PointROPE
```

这里按现有 L0 作为工程参考，不额外补跑控制实验。已有 L0 使用 450/120 调度，L0D 使用 250/62，因此 L0 与 L0D 的差异不能解释成严格的 decoder-only 因果效应；正式论文表述需要披露这一限制。

L2 的层数不是一个新原理；它必须与 L1 完全相同，才能把差异归因于 PointROPE。`(2,2,2,6,2)` 来自 LitePT-S 的深度分配：高分辨率卷积级只保留少量局部块，低分辨率的第 4 个 stage 分配更多 attention blocks 用于上下文建模，瓶颈 stage 再收缩到 2 个 block。该数字在 KPConvX 上仍需通过 L0D→L1 验证，不能直接当成最优配置。

## 4. 新增的训练脚本

```bash
# L0 + light decoder
DATASET_PATH=/data/S3DIS \
./Standalone/run_S3DIS_litept_l0d_250_seed57106803.sh

# L1: small depth
DATASET_PATH=/data/S3DIS \
./Standalone/run_S3DIS_litept_l1_250_seed57106803.sh

# L2: L1 without PointROPE
DATASET_PATH=/data/S3DIS \
./Standalone/run_S3DIS_litept_l2_250_seed57106803.sh
```

用于 FastAdapter 的两条可选路径：

```bash
# 原 KPConvX grid hierarchy + FastAdapter
DATASET_PATH=/data/S3DIS \
./Standalone/run_S3DIS_kpconvx_fastadapter_250_seed57106803.sh

# L0 hybrid hierarchy + FastAdapter
DATASET_PATH=/data/S3DIS \
./Standalone/run_S3DIS_litept_l0_fastadapter_250_seed57106803.sh
```

第一条使用现有 B0 作为参考，检验 FastAdapter 对 grid sampling 的贡献；第二条使用现有 L0 作为参考，检验 FastAdapter 与后两级 token attention 是否互补。若参考实验训练协议不同，应在结果中披露，不能写成严格单变量消融。

## 5. Figure 4 风格的 Grid/FastAdapter 比较

新增工具：

```text
Standalone/KPConvX/tools/analyze_stage_representations.py
Standalone/analyze_S3DIS_grid_vs_fastadapter.sh
```

使用方法：

```bash
DATASET_PATH=/data/S3DIS \
BASELINE_LOG=/results/grid_baseline \
FASTADAPTER_LOG=/results/grid_fastadapter \
OUTPUT_DIR=/results/diagnostics/grid_vs_fa \
./Standalone/analyze_S3DIS_grid_vs_fastadapter.sh
```

单次工具对两个 checkpoint 使用同一个确定性测试房间和同一个点金字塔。默认还会增加第三行 `Grid+FastAdapter (bypass)`：加载同一个 FastAdapter checkpoint，但在推理时旁路 Adapter。这样可以区分“训练出一个不同模型”的收益与“Adapter 当前前向补偿”的直接贡献。若只需要两行，可设置 `INCLUDE_ADAPTER_BYPASS=0`。论文统计建议运行多房间聚合脚本：

```bash
DATASET_PATH=<DATASET_PATH> \
BASELINE_LOG=<GRID_LOG> \
FASTADAPTER_LOG=<FASTADAPTER_LOG> \
NUM_RUNS=10 \
./Standalone/analyze_S3DIS_grid_vs_fastadapter_multi.sh
```

多房间脚本默认显式分析 Area-5 的 scene index `0..NUM_RUNS-1`，避免依赖随机采样导致同一房间重复。也可以指定更均匀的房间集合：

```bash
SCENE_INDICES="0 5 10 15 20 25 30 35 40 45" \
DATASET_PATH=<DATASET_PATH> \
BASELINE_LOG=<GRID_LOG> \
FASTADAPTER_LOG=<FASTADAPTER_LOG> \
./Standalone/analyze_S3DIS_grid_vs_fastadapter_multi.sh
```

每次运行会记录 `cloud_id`，聚合结果位于 `summary/`。单次输出包括：

```text
stage_representation_comparison.png
grid_stage_0..4.ply
grid_plus_fastadapter_stage_0..4.ply
grid_degradation.csv
representation_metrics.csv
adapter_response.csv
degradation_task_metrics.csv
fastadapter_gain_by_degradation.csv
per_class_metrics.csv
single_cloud_metrics.csv
```

### 5.1 为什么采用 joint PCA

两个模型在每个 stage 的特征拼接后共同拟合一个 PCA 基底，再使用相同的 1%–99% 分位范围映射 RGB。若分别拟合 PCA，主成分轴可以任意交换或翻转，颜色差异可能只是坐标系差异，不具备可比性。

### 5.2 仅有 PCA 图不够

`grid_degradation.csv` 量化每个 stage 的：

- 点数和相对保留率；
- coarse cell occupancy；
- mixed-cell ratio；
- 原始标签在 coarse token 内的熵；
- minority-label fraction。

`adapter_response.csv` 再测试：

- A2P correction ratio；
- A2P gate 均值和方差；
- correction 与当前/下一次 grid cell occupancy 的 Spearman 相关；
- correction 与当前/下一次 cell entropy 的相关；
- mixed cell 与 pure cell 中的平均 correction 差异。

`degradation_task_metrics.csv` 将两个模型分别放到以下子集上评估：

- semantic boundary / non-boundary；
- pure grid cells / mixed-label grid cells；
- occupancy 最低与最高四分位；
- mixed cells 中 entropy 最高四分位；
- 同时属于 boundary 和 mixed cell 的困难点。

`fastadapter_gain_by_degradation.csv` 直接输出 FastAdapter 相对 grid baseline 的 mIoU、mAcc、OA 增益和 error-rate reduction。论文机制结论不应只依赖整体 mIoU，而应要求 FastAdapter 在 mixed/high-entropy/boundary 子集上的增益显著高于 pure/non-boundary 子集。

能够支持机制结论的现象应是：grid 层级越深，occupancy 和 label entropy 越高；FastAdapter correction 在即将进入 mixed/high-entropy cell 的点上更强；这些区域的分割错误同时下降。PCA 图只做定性展示，CSV 指标和多房间统计才是论文证据。

## 6. 新增训练监控

监控默认关闭，不改变原训练行为。250-epoch 新脚本默认每 50 个 optimizer step 开启一次诊断，只在该次梯度累积的最后一个 mini-batch 计算。

输出：

```text
optimization_monitor.csv
fast_adapter_monitor.csv
```

`optimization_monitor.csv` 包含：

- 当前参数组学习率范围；
- global gradient norm/RMS；
- backbone、LitePT attention、FastAdapter、head 的 gradient RMS；
- weight RMS；
- gradient max absolute value。
- 经过真实 `optimizer.step()` 后的 sampled update RMS 和 update/weight ratio。

`fast_adapter_monitor.csv` 每个 stage 包含：

- residual scale；
- P2A/A2P gate 均值和标准差；
- gate 饱和比例；
- correction ratio 均值与 P90；
- anchor occupancy；
- spatial output projection norm。

监控不保存完整激活，不在每一步执行，也不改变 loss、gradient clipping 或 optimizer。实际 update ratio 只对每个参数张量确定性抽样少量元素，而不是复制完整模型。主要影响是监控 step 上多一次参数遍历和少量 reduction；间隔为 50 时，整体训练开销通常远低于逐步记录。正式吞吐测试应关闭 `monitor_enabled`。

## 7. 在开始原创 attention 改进之前必须完成的工作

优先完成：

1. L0D、L1、L2 单 seed；
2. Grid vs Grid+FastAdapter 机制诊断，至少 10 个 Area-5 房间；
3. L1 与 L2 的 PointROPE 差异；
4. 记录 stage token 数、attention latency、gradient RMS；
5. 最佳两种配置做 3 seeds。

完成这些后，才有证据判断真正瓶颈是深层 attention、位置编码、grid 信息损失，还是 decoder。

## 8. 后续更适合 KPConvD 的原创改进方向

### 8.1 KPConvD-conditioned attention

用 stage 2/3 的 KPConvD 局部几何描述生成 attention 的门控或 Q/K 缩放，使后级 attention 不是完全从线性投影重新学习局部结构。需要与纯 PointROPE attention 对比，证明卷积先验确实改善边界或稀疏区域。

### 8.2 Kernel-point relative bias

将 KPConvD 的最近 kernel-point 分区、归一化 offset 或 shell index 转换为 attention bias，与 PointROPE 并行使用。这样位置编码不仅包含绝对 grid phase，还包含 KPConv 特有的局部几何分区。

### 8.3 Degradation-aware token attention

利用 grid occupancy、cell entropy 的无标签代理量（occupancy、半径内点间距、局部 PCA）自适应调整 patch size 或 attention budget。高退化区域使用更大 patch/更多 heads，规则平面区域保持轻量。

### 8.4 FastAdapter-anchor / token attention 合并

若实验发现 FastAdapter spatial attention 与 stage 3/4 token attention 冗余，可保留 P2A、cross-layer、A2P，但把 anchor feature 作为少量 memory tokens 加入深层 attention。这样避免两套独立 self-attention，同时将浅层局部几何直接送到深层语义阶段。

### 8.5 Geometry-preserving hand-over stage

在 C→A 切换处使用一个专门的 transition block：KPConvD 输出分成 local branch 与 semantic branch，后者进入 PointROPE attention，前者通过可学习 gate 保留。它比简单串联 `KPConvD + Attention` 更贴合当前 backbone，也更容易形成独立创新点。

这些方向应在完成 L0D/L1/L2 和 Grid/FastAdapter 诊断后开始。否则无法知道新模块修复的是哪个问题，也无法设计有说服力的消融。
