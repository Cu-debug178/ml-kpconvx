# S3DIS LitePT L0，seed 57106803

## Summary

- Status: `completed`
- Run ID: `s3dis_litept_l0_b12a2_seed57106803`
- Exact training Git commit: `not recorded`
- Dataset and split: S3DIS；训练/验证/测试为 Area 5 协议
- Model: LitePT L0，11,308,499 参数；高分辨率 3 个卷积阶段、低分辨率 2 个 PointROPE attention 阶段（C-C-C-A-A）
- Seed: `57106803`
- Best validation result: epoch `446`，validation mIoU `75.9846%`
- Best tested full-cloud result: epoch `210`，10-vote full-cloud mIoU `72.1%`

## Configuration

| Setting | Value |
|---|---|
| Encoder depth | `(3,3,9,12,3)` |
| LitePT hierarchy | C-C-C-A-A |
| KP operator / LitePT | `kpconvx` / enabled |
| PointROPE | enabled |
| Patch size | `128` |
| Decoder | heavy；`litept_light_decoder=false`，`decoder_layer=true` |
| FastAdapter | disabled |
| Input features | `5` |
| Initial channels / channel scaling | `64 / 1.41` |
| Input subsampling | grid，`in_sub_size=0.04` |
| Input radius / radius scaling | `2.1 / 2.2` |
| Grid pooling | enabled |
| Neighbor limits | `[12,16,20,20,20]` |
| Epochs / steps per epoch | `450 / 300` |
| Batch / accumulation | `12 / 2`，effective batch `24` |
| Optimizer | AdamW；初始 learning rate `1e-4`，weight decay `0.05` |
| Test protocol | S3DIS Area 5 full-cloud，10 votes |

## Results

| Result role | Epoch | Metric | Value |
|---|---:|---|---:|
| Best validation | 446 | validation mIoU | 75.9846% |
| Best tested full-cloud | 210 | 10-vote full-cloud mIoU | 72.1% |
| Final validation | 450 | validation mIoU | 70.5462% |
| Final tested full-cloud | 450 | 10-vote full-cloud mIoU | 71.7% |

共测试 33 个不同 internal epoch 的 checkpoint，full-cloud mIoU 范围为
`69.7%–72.1%`，算术平均 `71.53%`；完整数据见 [metrics.csv](metrics.csv)。

## Protocol and limitations

- 验证最佳 checkpoint 和 full-cloud 最佳 checkpoint 按不同指标确定，不能混称为同一个“最佳”。
- Area 5 同时承担验证和测试，存在 checkpoint-selection bias。
- 这是单 seed 结果；训练过程中有中断/恢复历史，精确训练 Git commit 未记录。
- checkpoint 文件、数据集和原始测试日志不纳入版本化报告。
