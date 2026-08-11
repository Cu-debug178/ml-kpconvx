# S3DIS L0 epoch 450 Stage 1 fixed-10 diagnostics

## Summary

- Status: `completed`
- Run ID: `l0-epoch450-stage1-fixed10-20260811`
- Exact training Git commit: `not recorded`
- Dataset and split: S3DIS Area 5，预先固定的 10 个房间子集
- Model: LitePT L0；legacy KPConvD encoder + KPConvX decoder
- Seed: `57106803`
- Checkpoint: `current_chkp.tar`，internal epoch `450`
- Training: 无；本次仅执行离线诊断
- Fixed-10 result: mIoU `70.941941%`，OA `90.251469%`

## Configuration

使用同一个 checkpoint 对 10 个固定房间执行确定性推理。房间列表为：
`Area_5_WC_1`、`Area_5_conferenceRoom_2`、`Area_5_hallway_2`、
`Area_5_hallway_8`、`Area_5_lobby_1`、`Area_5_office_1`、
`Area_5_office_21`、`Area_5_office_38`、`Area_5_pantry_1` 和
`Area_5_storage_3`。

| Setting | Value |
|---|---|
| Room count | 10 |
| Kernel entropy source | zero-based encoder Stage 2 |
| Attention query sampling | 每个 block 最多 256 个确定性 query |
| Latency repeats | 每个房间 3 次 warmed repeats |
| Device | CUDA device；具体 GPU 未记录 |
| Hardware / software versions | `not recorded` |
| Parameters / peak memory / elapsed time | `not recorded` |

## Protocol

- Token 数来自真实 forward 的各 encoder stage。
- latency 使用 CUDA event 测量模型计算，不含数据加载和 host-to-device
  传输；stage 值报告 30 次观测的均值，端到端值报告中位数。
- patch-KP overlap 是 serialized patch 内保留的实际 KP 邻接边比例，
  KP edge cut rate 是其补集。报告 `union` 顺序的房间均值。
- kernel occupancy entropy 使用 43-kernel occupancy signature；同时报告
  token 归一化 entropy 的房间均值与全局 kernel entropy 的房间均值。
- error/entropy 相关性为 10 个房间 Spearman 系数的算术平均，分别按 point
  error 与 token error rate 计算。
- sampled attention 只重建每个 block 配置的确定性 query 子样本，不是全量
  attention matrix。
- Stage 2->3 residual ratio 定义为第一个 token-attention block 中 branch norm
  除以 block-input norm。
- boundary 和 mixed-cell 指标在固定 10 房间的全分辨率预测上计算；mixed/pure
  使用覆盖整个 cell 的 `full_cell` 定义。
- checkpoint 在诊断前已固定，本报告没有根据这 10 个房间再次选择 checkpoint。

## Results

主要观察包括：Stage 3 的 union edge cut rate 为 `17.0170%`，Stage 4 为
`4.5367%`；error 与 kernel entropy 的 point/token Spearman 均接近零；5 cm
语义边界 error 为 `30.0149%`。Stage 2 mixed-cell error 为 `21.4007%`，高于
pure-cell 的 `7.3365%`；Stage 3 分别为 `15.7306%` 与 `5.7012%`。

完整的版本化数值见 [metrics.csv](metrics.csv)。原始逐房间日志、预测、点云和
checkpoint 保留在本机结果目录，不纳入 Git。

## Limitations and anomalies

- 这是固定 10 房间子集，不是标准完整 Area 5 验证或 10-vote full-cloud 测试；
  `70.941941%` 不能与完整 Area 5 的 `71.7%` 直接当作同协议结果比较。
- 只有一个 seed 和一个 checkpoint，不能估计训练方差。
- error/entropy 相关性是描述性统计，不构成 kernel geometry 的因果证据。
- sampled attention、3 次 latency repeat 和固定房间选择限制了外推范围。
- checkpoint 的训练 commit、诊断 GPU、软件版本和资源 telemetry 未在运行时捕获，
  因而标记为 `not recorded`，没有事后用当前环境替代。
- PCA shape profile 曾将邻居数上限设为 64，饱和率为 `99.9736%`；本报告不纳入
  这些近似 PCA 指标。boundary 与 density 统计不受该上限影响。
