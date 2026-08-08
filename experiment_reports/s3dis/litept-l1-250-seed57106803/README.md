# S3DIS LitePT L1，seed 57106803

## Summary

- Status: `completed`
- Run ID: `s3dis_litept_l1_250_seed57106803`
- Exact training Git commit: `not recorded`
- Dataset and split: S3DIS；训练/验证为 Area 5 协议
- Model: LitePT L1，5,411,035 参数；C-C-C-A-A 的轻量 encoder 配置
- Seed: `57106803`
- Best validation result (secondary): epoch `159`，validation mIoU `73.0231%`
- Best tested full-cloud result: checkpoint epoch `130`（`chkp_0130.tar`），10-vote full-cloud mIoU `69.6%`

## Configuration

| Setting | Value |
|---|---|
| Encoder depth | `(2,2,2,6,2)` |
| LitePT hierarchy | C-C-C-A-A |
| KP operator / LitePT | `kpconvx` / enabled |
| PointROPE | enabled |
| Patch size | `128` |
| Decoder | light LayerNorm decoder；`litept_light_decoder=true`，`decoder_layer=false` |
| FastAdapter | disabled |
| Input features | `5` |
| Initial channels / channel scaling | `64 / 1.41` |
| Input subsampling | grid，`in_sub_size=0.04` |
| Input radius / radius scaling | `2.1 / 2.2` |
| Grid pooling | enabled |
| Neighbor limits | `[12,16,20,20,20]` |
| Epochs / steps per epoch | `250 / 300` |
| Batch / accumulation | `12 / 2`，effective batch `24` |
| Optimizer | AdamW；初始 learning rate `1e-4`，weight decay `0.05` |
| Checkpoint archive | epoch 100 起每 5 epoch |

## Results

| Result role | Epoch | Metric | Value |
|---|---:|---|---:|
| Best tested full-cloud | 130 | 10-vote full-cloud mIoU | 69.6% |

验证集最佳为 epoch 159（73.0231%），仅作为 checkpoint 选择参考；完整验证曲线仍保留在 [metrics.csv](metrics.csv)。

本次 full-cloud 测试覆盖 epoch 100–250（每 5 epoch 一个 checkpoint，另含已测试的 epoch 160），使用 Area 5、10 votes，测试过程正常完成。不能用 L1 的
validation mIoU 代替 full-cloud test mIoU。

## Protocol and limitations

- Area 5 同时用于验证和 full-cloud 测试，存在 checkpoint-selection bias。
- 这是单 seed 结果；精确训练 Git commit 未记录。
- checkpoint 文件、数据集和原始日志不纳入版本化报告。
