# S3DIS 三层证据诊断

工具入口：`tools/analyze_s3dis_difficulties.py`。

它回答三个彼此不同的问题：

1. S3DIS 本身包含哪些困难；
2. 一个模型具体在哪里犯错；
3. 新模型相对 baseline 在同一批点上修复了什么、又破坏了什么。

仅有 checkpoint 不能回答后两个问题。必须先生成 Area 5 的 full-resolution
逐点预测，并严格对齐 baseline、新模型的场景名、坐标和 GT。

所有大文件（逐点预测、概率和几何属性）都应放在
`Standalone/KPConvX/results/`，不要提交到 Git。

## 训练期间的安全规则

- 不要在训练期间运行 `infer`。工具默认检测其他 GPU compute process，发现后会拒绝
  启动；`--allow_gpu_contention` 只用于你明确接受显存/OOM和训练变慢风险的场景。
- 不要读取训练正在覆盖写入的 `current_chkp.tar`。等训练结束后使用稳定的 best/periodic
  checkpoint，或先由训练流程正常保存一个快照。
- `dataset` 的完整几何和 `profile --compute_geometry` 虽不占 GPU，但会竞争 CPU、内存和
  数据盘，也应等训练结束。训练期间可以只阅读文档、准备命令和修改代码。
- 不要为了诊断停止训练进程、DataLoader worker、screen 会话或本地代理。

## 1. 数据集审计（不需要权重）

在 `Standalone/KPConvX/` 下执行：

```bash
python3 tools/analyze_s3dis_difficulties.py dataset \
  --dataset_path <DATASET_PATH> \
  --output_dir results/s3dis_diagnostics/dataset_audit
```

默认对训练区域和 Area 5 统计类别点数、占比、房间覆盖率和每个房间的类别组成，
但只对 Area 5 计算较耗时的固定半径几何。默认几何尺度是 5、10、20 cm；
每个房间最多抽取 100000 个 query 点，但邻居始终从完整房间查找，因此不会把
抽样后的点数错误当成密度。PCA 默认使用半径内全部邻居；如果为了速度设置
`--max_pca_neighbors`，必须同时报告 `pca_neighbor_cap_saturation_ratio`，高饱和率的
结果只能解释为 capped-neighbour PCA。

重要输出：

- `dataset_class_stats.csv`：类别是否少、是否只出现在少数房间；
- `room_class_composition.csv`：训练区域与 Area 5 是否存在房间级分布偏移；
- `room_geometry_summary.csv`：固定物理半径密度、边界比例和 PCA 几何；
- `instance_stats.csv`：只统计文件中真实提供的 instance ID。没有 ID 时为空，
  不会把语义连通分量伪装成真实实例。

若只需要快速检查类别分布：

```bash
python3 tools/analyze_s3dis_difficulties.py dataset \
  --dataset_path <DATASET_PATH> \
  --geometry_split none \
  --output_dir results/s3dis_diagnostics/dataset_counts_only
```

## 2. 从 checkpoint 导出逐点预测

`--log_path` 必须是训练日志目录，因为模型结构配置保存在其中。checkpoint 可以
显式指定；不指定时会依次查找 best/current/最后一个周期权重。

```bash
python3 tools/analyze_s3dis_difficulties.py infer \
  --log_path <BASELINE_LOG> \
  --checkpoint <BASELINE_CHECKPOINT> \
  --dataset_path <DATASET_PATH> \
  --capture_hierarchy \
  --output_dir results/s3dis_diagnostics/baseline_export
```

对新模型重复一次：

```bash
python3 tools/analyze_s3dis_difficulties.py infer \
  --log_path <CANDIDATE_LOG> \
  --checkpoint <CANDIDATE_CHECKPOINT> \
  --dataset_path <DATASET_PATH> \
  --capture_hierarchy \
  --output_dir results/s3dis_diagnostics/candidate_export
```

每个房间保存一个 `.npz`，包含：

- full-resolution `points / labels / predictions`；
- `prediction_covered` 和访问次数；
- 使用 `--capture_hierarchy` 时，模型实际 pooling cell 的 occupancy、label entropy、
  mixed-cell，以及 LitePT attention/handover stage 的 patch-neighbor recall/cut ratio。

这里的 grid 和 patch 属性来自一次真实的模型层级 trace，不是事后按坐标随意切格子的
代理。工具同时输出 stage-0 标签统计和将实际 cell ID 投影到原始点后重新用 full GT
统计的 `full_cell_*`；论文中的原始点 mixed-grid 证据应使用后者。patch-cut 排除
self/shadow 邻居，并分别记录每种 Morton order 和多 order union。

必须先查看 `coverage.csv`。若不是所有房间都达到 100% coverage，不能直接使用导出预测
算 mIoU；应增加 `--in_radius` 或检查 regular centres。默认 `--in_radius 100` 适合把完整
S3DIS 房间作为输入，但显存不足时可以降低，工具会用重叠区域的平均概率合并。

`infer` 默认关闭随机测试增强，目的是让层级诊断可复现。`--votes` 可以增加覆盖次数，
但在无随机增强的设置下不等同于论文中的 TTA 投票。

如果数据和权重在另一台机器，只需在那里运行 `infer`，再复制两个 `predictions/`
目录回来；后续 `profile/compare` 不需要 GPU、日志或 checkpoint。

## 3. 单模型错误画像

```bash
python3 tools/analyze_s3dis_difficulties.py profile \
  --prediction_dir results/s3dis_diagnostics/baseline_export/predictions \
  --output_dir results/s3dis_diagnostics/baseline_profile
```

默认会补算 full-resolution 固定半径密度、边界和局部 PCA，并将带几何属性的逐点文件
缓存在输出目录的 `predictions_with_geometry/`。这一步可能耗时和占磁盘；如果导出文件
已经包含所需属性，或只想看已有的 hierarchy 属性，可加 `--no-compute_geometry`。

重要输出：

- `per_class_metrics.csv`：IoU、recall、precision 和 support；
- `confusion_matrix.csv`：beam/column 等到底被错分成什么；
- `subset_metrics.csv`：boundary/interior、低/高密度、高曲率、mixed/pure cell、
  高/低 patch-cut 子集；
- `per_class_subset_metrics.csv`：同一困难子集内逐类别统计，用来检查类别组成混杂；
- `per_room_metrics.csv`：错误是否集中于少数房间；
- `difficulty_thresholds.json`：密度/PCA/entropy/cut 的 q25/q50/q75 定义。

`mIoU_present` 只平均该子集实际出现的类别，适合描述子集，但不能冒充标准 Area 5
13 类 mIoU。必须同时查看 `point_count`、OA/error rate 和 confusion。

## 4. Baseline 与新模型成对比较

```bash
python3 tools/analyze_s3dis_difficulties.py compare \
  --baseline_predictions results/s3dis_diagnostics/baseline_export/predictions \
  --candidate_predictions results/s3dis_diagnostics/candidate_export/predictions \
  --baseline_name B0_KPConvD \
  --candidate_name B1_NewAttention \
  --output_dir results/s3dis_diagnostics/B0_vs_B1
```

比较前会强制检查两个目录的 room 集合、GT、点数和坐标；任一不一致都会停止，防止把
不同采样或不同投影的结果伪装成成对实验。困难标签只从 baseline 文件中的几何/层级属性
定义，不使用“新模型是否预测正确”来定义困难。

重要输出：

- `paired_summary.csv`：Area 5 总体差值；
- `gain_by_class.csv`：每类 IoU/recall/precision 的变化；
- `gain_by_subset.csv`：候选模型的增益是否集中在预定义困难子集；
- `gain_by_class_subset.csv`：困难子集内逐类别的成对增益；
- `transitions_by_subset.csv`：`fixed_by_candidate` 与 `regressed_by_candidate`；
- `gain_by_room.csv`：每个房间的成对差异；
- `paired_room_bootstrap.csv`：以房间为重采样单位的 95% 区间。

例如，只有同时看到以下证据，才适合说新注意力“缓解了 patch-cut”：

1. high-cut 子集的 paired gain 明显大于 low-cut；
2. high-cut 中 fixed points 明显多于 regressed points；
3. 改善不是由单个房间或单一大类支配；
4. 多个 seed 上方向一致。

如果只看到总 mIoU 上涨，不能确定改善来自边界、密度、grid 混合还是普通类别拟合。
如果只看到相关性，也不能证明 patch-cut 是因果机制；还需要改变 patch size/order 或加入
cross-patch 机制的受控消融。
