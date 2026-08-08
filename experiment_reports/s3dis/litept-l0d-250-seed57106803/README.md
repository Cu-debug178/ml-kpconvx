# S3DIS LitePT L0D 全量 checkpoint 测试

## Summary

- Status: `completed`
- Run ID: `s3dis_litept_l0d_250_seed57106803`
- Exact training Git commit: `not recorded`
- Dataset and split: S3DIS Area_5；训练/验证/测试协议沿用该实验目录配置
- Model: LitePT L0D；encoder blocks `(3,3,9,12,3)`，C-C-C-A-A，PointROPE 开启，patch size `128`，light decoder，FastAdapter 关闭
- Seed: `57106803`
- Best validation result: epoch `144`，validation mIoU `75.4%`
- Best tested full-cloud result: epoch `150`，10-vote full-cloud mIoU `71.8%`

## Configuration

- Batch protocol: `batch_size=12`，`accum_batch=2`（effective batch 24）
- Training budget: `250` epochs；checkpoint archive from epoch 100 every 5 epochs
- Decoder: light LayerNorm decoder，额外 decoder layer 关闭
- FastAdapter: disabled
- KP operator: `kpconvx`; input features `5`; initial channels `64`; channel scaling `1.41`
- Input subsampling: grid，`in_sub_size=0.04`; input radius `2.1`; radius scaling `2.2`
- Grid pooling: enabled; neighbor limits `[12,16,20,20,20]`
- Optimizer: AdamW；初始 learning rate `1e-4`，weight decay `0.05`
- Validation best: `75.4%`（validation epoch 144）；epoch 250 validation 为 `75.4%`（日志原始精度约 `75.39%`）

## Protocol

每个不同 internal epoch 只测试一次，使用 full-cloud S3DIS Area_5 测试和 10 votes。共 31 个 checkpoint：epoch 100–250，每 5 epoch 一个；`chkp_exact_*`、普通 `chkp_*` 和 `current_chkp` 若对应同一 internal epoch 不重复计入。`metrics.csv` 中的 `fullcloud_mIoU_pct` 来自测试报告，`return_code=0` 表示测试成功。

## Environment and resources

- Training GPU、测试 GPU、CUDA/PyTorch 版本和峰值显存：`not recorded`
- 训练运行时 Git commit：`not recorded`
- 结果来源：本地 `eval_200_250_20260806/manifest.csv`、`eval_remaining_20260806/manifest.csv` 及对应 test reports

## Results

31 个 checkpoint 全部测试成功。full-cloud mIoU 范围为 `69.9%–71.8%`，算术平均 `71.24%`。指定的 late checkpoints 为：

| internal epoch | full-cloud mIoU |
|---:|---:|
| 200 | 71.3% |
| 210 | 71.5% |
| 220 | 71.3% |
| 230 | 71.1% |
| 240 | 71.3% |
| 250 | 71.6% |

完整逐 checkpoint 数据见 [metrics.csv](metrics.csv)。测试报告、manifest 和日志仍保留在本地结果目录；checkpoint 二进制在本次汇总核对后可清理，不纳入版本化报告。

## Limitations and anomalies

- 这是单 seed 结果，不能代表多 seed 均值或置信区间。
- 测试 checkpoint 是按训练过程规则覆盖的固定 epoch，不等同于独立验证集选择；若按 full-cloud 测试结果选最优，会产生 checkpoint-selection bias。
- validation mIoU 与 full-cloud test mIoU 不是同一指标，不能直接混用。
- 评估使用 10 votes；没有记录 13-TTA 结果，因此本报告不宣称使用 Pointcept 的 13-TTA。
- 训练后评估包装脚本曾有历史语法错误，但本次 31 个测试返回码均为 0；历史失败不影响本报告中的已完成测试结果。
