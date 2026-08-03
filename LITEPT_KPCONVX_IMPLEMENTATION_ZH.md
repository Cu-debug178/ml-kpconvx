# LitePT 思想在 Standalone KPConvX 中的可行性、工程实现与实验方案

## 1. 结论

**可行，且值得实现，但不能把 KPConvX 的 kernel attention 直接当作 LitePT 的深层 token attention。**

LitePT 的核心不是换一种采样方法，而是按 U-Net 层级分工：高分辨率浅层用卷积提取局部几何，低分辨率深层用自注意力建模语义与长程上下文；深层去掉卷积后，再用无参数 PointROPE 为 query/key 注入三维位置。这个原则与 KPConvX 当前采用 grid pooling 还是其他采样方式没有硬耦合。

工程上有三点需要处理：

1. KPConvX 的 kernel attention 是对固定 kernel-point 空间区域进行通道分组调制，仍属于卷积算子；LitePT 深层使用的是点 token 之间的 self-attention。只把 `first_inv_layer` 改成 3，并不能得到 LitePT。
2. LitePT 官方完整实现依赖 spconv、FlashAttention 和自己的 serialization 数据结构，不能无代价塞进当前 Standalone KPConvX 的 packed variable-length batch。
3. KPConvX 的深层通道宽度（默认约 192、256）不天然满足 PointROPE 的每头通道需按 x/y/z 拆分、且每轴成对旋转的约束。因此实现中使用独立 QKV 内部宽度，并自动舍入到 `6 × num_heads` 的整数倍。

本次已完成一个**依赖轻、可训练、可消融、与现有 FastAdapter 可组合**的工程版本。

---

## 2. 采用了哪些 LitePT 创新

### 2.1 分层算子分工

启用后，五层 encoder 默认变为：

```text
E1: KPConvD
E2: KPConvD
E3: KPConvD
E4: Serialized PointROPE Self-Attention
E5: Serialized PointROPE Self-Attention
```

即 `C-C-C-A-A`。其中：

- `C` 是关闭 kernel modulation 的 KPConvD residual block；
- `A` 是新增的点 token self-attention block，不再执行 KPConv；
- 可选 `X` hand-over stage 在同一 stage 内依次执行 KPConvD 和 PointROPE attention。

支持：

```text
C-C-X-A-A  --litept_conv_stages 2 --litept_handover_stage 3
C-C-C-X-A  --litept_conv_stages 3 --litept_handover_stage 4
```

### 2.2 PointROPE

对每个 attention head 的特征维度 `D`：

1. 强制 `D % 6 == 0`；
2. 按 x/y/z 均分成三个子空间；
3. 每个子空间再两两配对；
4. 分别用量化后的 x、y、z 网格坐标做旋转；
5. 只旋转 query 和 key，value 不变。

默认 base frequency 为 100。可用 `--litept_rope_enabled 0` 做严格消融。

### 2.3 局部 serialization attention

没有引入 spconv 或第三方 FlashAttention：

1. 每个 packed cloud 独立量化坐标；
2. 计算 Morton/Z-order code；
3. 按空间顺序排序；
4. 切成固定大小 patch；
5. 用 `torch.nn.functional.scaled_dot_product_attention` 计算局部注意力；
6. 用原始点索引还原 packed 顺序。

当前支持两种顺序：

```text
z
z-trans
```

不同 block 循环使用两种顺序，以改变 patch 边界。这里不是逐行复制 LitePT 官方 serialization；官方还支持 Hilbert 顺序。本实现优先保证 Standalone 可运行、依赖少和易消融。

### 2.4 serialization 缓存

排序不能在每个 attention block 重做。每个 stage 共享一个 `SerializedPatchCache`：

- 每个新 forward 开始时清空；
- 同一 stage 的不同 order 复用一次 cloud-local 坐标量化；
- 同一个 stage、同一种 order 只构建一次 patch metadata；
- 后续 block 复用索引、mask 和网格坐标。

因此默认 E4/E5 各自只量化一次，再构建 `z` 和 `z-trans` 两套排序，而不是按 block 数重复量化和排序。Morton 位宽调整保留在设备端计算，避免为每个 cloud/order 调用 `.item()` 引发 CUDA 同步。

### 2.5 轻量 decoder

S3DIS 默认脚本启用：

```text
--litept_light_decoder 1
--decoder_layer 0
```

decoder 只保留：上采样、skip concat、通道投影与 LayerNorm，不再额外堆 KPConvX residual block。这对应 LitePT 在语义分割中采用轻量 decoder 的原始定义；普通 KPConvX 或 heavy decoder 路径仍保留原有归一化设置。

### 2.6 与 FastAdapter 的组合

两条路径相互独立：

```text
stage local geometry / token context
    -> LitePT stage-tailored encoder
    -> FastAdapter P2A / cross-layer / anchor spatial / A2P
    -> pooling or head
```

可用环境变量直接打开：

```bash
FA_ENABLED=1 ./train_S3DIS_litept.sh
FA_ENABLED=1 ./train_ScanObjectNN_litept.sh
```

这不是默认认为二者一定叠加增益。深层 token attention 和 anchor spatial attention 都在补充上下文，可能互补，也可能冗余，必须用 2×2 因子实验验证。

---

## 3. 代码结构

### 新增

```text
Standalone/KPConvX/models/litept_blocks.py
Standalone/KPConvX/tests/test_litept_blocks.py
Standalone/KPConvX/tests/test_litept_kpnext_integration.py
Standalone/train_S3DIS_litept.sh
Standalone/train_ScanObjectNN_litept.sh
LITEPT_KPCONVX_IMPLEMENTATION_ZH.md
litept_experiment_matrix.csv
```

### 修改

```text
Standalone/KPConvX/models/KPNext.py
Standalone/KPConvX/utils/config.py
Standalone/KPConvX/experiments/S3DIS/train_S3DIS.py
Standalone/KPConvX/experiments/ScanObjectNN/train_ScanObj.py
Standalone/README.md
```

### 核心类

- `PointROPE`：纯 PyTorch 三维 rotary embedding；
- `SerializedPatchCache`：stage 级 serialization 复用；
- `SerializedPointROPEAttention`：packed cloud 的局部 token attention；
- `LitePointTransformerBlock`：pre-norm attention + MLP + packed DropPath；
- `LiteHandoverBlock`：KPConvD 后接 attention；
- `KPNeXt.get_encoder_block`：根据 stage 选择 C、A、X 或原 KPConvX block。

---

## 4. 默认架构与参数含义

两个 LitePT 启动脚本默认使用：

```text
layer_blocks = (2, 2, 2, 6, 2)
conv_stages = 3
handover_stage = 0
heads = 8
PointROPE base = 100
orders = z,z-trans
```

这沿用 LitePT-S 的 block 深度分配，但保留 KPConvX 的 stem、点金字塔、通道增长方式和任务 head。因此它是“LitePT 思想内化到 KPConvX”，不是声称复现官方 LitePT。

S3DIS 默认 patch size 128；ScanObjectNN 默认 64。原因是当前实现采用通用 PyTorch SDPA，而不是官方 FlashAttention CUDA kernel。patch 越大，上下文越广，但注意力计算和显存近似按 patch size 的平方增长。

主要参数：

| 参数 | 作用 |
|---|---|
| `litept_enabled` | 开启 stage-tailored encoder |
| `litept_conv_stages` | 前多少个 stage 只用 KPConvD |
| `litept_handover_stage` | 0 表示关闭；否则必须紧接纯卷积 stages，即 `handover_stage = conv_stages + 1` |
| `litept_patch_size` | serialization patch 大小 |
| `litept_num_heads` | 深层 token attention 头数 |
| `litept_attention_ratio` | QKV 内部宽度相对 stage 宽度 |
| `litept_mlp_ratio` | attention block 的 MLP expansion |
| `litept_rope_base` | PointROPE base frequency |
| `litept_rope_enabled` | PointROPE 开关，供消融 |
| `litept_orders` | `z` 或 `z,z-trans` |
| `litept_light_decoder` | 语义分割使用轻 decoder |

### 通道适配

KPConvX 默认深层通道约为：

```text
E4 = 192
E5 = 256
```

8 heads 时：

```text
192 -> attention inner dim 192 -> head dim 24
256 -> attention inner dim 240 -> head dim 30
```

24 和 30 都能被 6 整除，因此可以按三轴拆分并做成对旋转。attention 输出再投影回原 stage width，外部网络接口不变。

---

## 5. 启动方式

### S3DIS

```bash
cd Standalone
DATASET_PATH=<S3DIS_PATH> ./train_S3DIS_litept.sh
```

深度匹配现有 KPConvX-L，以隔离算子分配影响：

```bash
DATASET_PATH=<S3DIS_PATH> ./train_S3DIS_litept.sh \
  --layer_blocks 3 3 9 12 3 \
  --litept_light_decoder 0 \
  --decoder_layer 1
```

上面的 L0 同时保持 B0 的深度和 heavy decoder。仅追加 `--layer_blocks 3 3 9 12 3` 会得到 L0D，用来单独测量 light decoder 的影响。

hand-over：

```bash
HANDOVER_STAGE=3 ./train_S3DIS_litept.sh
HANDOVER_STAGE=4 ./train_S3DIS_litept.sh
```

关闭 PointROPE：

```bash
./train_S3DIS_litept.sh --litept_rope_enabled 0
```

LitePT + FastAdapter：

```bash
FA_ENABLED=1 ./train_S3DIS_litept.sh
```

### ScanObjectNN

```bash
cd Standalone
DATASET_PATH=<SCANOBJECTNN_PATH> ./train_ScanObjectNN_litept.sh
```

建议至少运行三个种子：

```bash
for seed in 1 42 1000; do
  SEED="$seed" LOG_PATH="<RESULT_ROOT>/litept_seed_${seed}" \
    ./train_ScanObjectNN_litept.sh
done
```

---

## 6. 推荐实验方案

完整矩阵见根目录 `litept_experiment_matrix.csv`。

### 6.1 第一阶段：先证伪/证实“分层分工”

S3DIS 固定训练协议，只改网络：

| ID | 架构 | 目的 |
|---|---|---|
| B0 | 原 KPConvX-L，`first_inv_layer=1` | 当前基线 |
| B1 | `first_inv_layer=3`，仍为 KPConvX kernel attention，heavy decoder | 检验仅延后 kernel attention 是否够用 |
| L0 | `C-C-C-A-A`，深度 `(3,3,9,12,3)`，heavy decoder | 与 B0 同深度、同 decoder，隔离 token attention |
| L0D | L0 + light decoder | 单独测量 decoder 简化的精度与效率影响 |
| L1 | `C-C-C-A-A`，深度 `(2,2,2,6,2)` | 主要效率模型 |
| L2 | L1 关闭 PointROPE | 验证位置编码贡献 |
| L3 | `C-C-X-A-A` | hand-over stage 3 |
| L4 | `C-C-C-X-A` | hand-over stage 4 |

不能只比较 B0 和 L1，因为它们同时改变算子、深度和 decoder。B0→L0 隔离算子，L0→L0D 隔离 decoder，L0D→L1 隔离深度压缩；这三个桥接比较缺一不可。

### 6.2 第二阶段：效率边界

以 L1 为中心：

```text
patch_size: 32, 64, 128, 256
orders: z vs z,z-trans
attention_ratio: 0.75, 1.0
mlp_ratio: 2, 4
light decoder: on/off（S3DIS）
```

记录：

- 参数量；
- 端到端训练显存峰值；
- 端到端推理显存峰值；
- 完整数据预处理、pyramid、serialization、网络和 head 的吞吐；
- 网络 forward-only latency；
- mIoU、mAcc、OA。

不要只报 attention block microbenchmark。Morton sorting、patch metadata 和恢复索引都属于真实成本。

### 6.3 第三阶段：与 FastAdapter 的 2×2 因子实验

| LitePT | FastAdapter | 解释 |
|---|---|---|
| 0 | 0 | 原始基线 |
| 1 | 0 | 分层卷积/attention 的独立贡献 |
| 0 | 1 | anchor 补偿的独立贡献 |
| 1 | 1 | 是否互补，及额外开销 |

若组合模型精度不升，优先消融 FastAdapter 的 `fa_spatial`，因为它与深层 token attention 的上下文功能最可能重叠；保留 P2A/A2P 和 cross-layer，测试其几何降采样补偿是否仍互补。

### 6.4 S3DIS 协议

- 主结果：Area-5；
- 指标：mIoU、mAcc、OA；
- 最少 3 个种子，最好沿用 KPConvX 原论文的多次测试思想；
- 训练和测试使用相同数据增强、epoch、optimizer、batch/accumulation；
- 完整房间推理与固定 15k 点 microbenchmark 分开报告；
- checkpoint 选择规则预先固定，避免只选单次最好结果。

### 6.5 ScanObjectNN 协议

- B0 使用当前配置默认的 KPConvD-L；另设 B0X 作为 KPConvX-L 次基线，不能把二者参数量或精度混写；
- 指标：OA、mAcc；
- 种子：1、42、1000；
- 报告 mean ± std；
- patch size 优先 32、64、128；
- 对分类任务同时比较 global pooling 前的 E5 token 数，避免某配置因异常强下采样获得虚假速度优势。

### 6.6 工程通过标准

建议用 Pareto 规则，而不是只追最高精度：

- L1 相对 B0 参数量明显下降；
- S3DIS mIoU 不低于 B0 超过 0.5 个点，或者在明显加速时允许小幅下降；
- 端到端吞吐不能只靠降低输入点数获得；
- 若 `patch_size=128` 比 64 精度增益不足 0.2，但延迟明显升高，默认用 64；
- LitePT+FastAdapter 只有在精度增益大于测量方差且速度成本可接受时才作为最终模型。

---

## 7. 当前代码级参数量检查

下面是本地按模型构造器统计的参数量，不是论文结果，也不是训练后的性能：

### S3DIS 配置形态

| 配置 | 参数量 |
|---|---:|
| 当前 KPConvX-L + heavy decoder | 18.289M |
| LitePT depth-matched + heavy decoder（L0） | 11.278M |
| LitePT depth-matched + light decoder（L0D） | 9.926M |
| LitePT `(2,2,2,6,2)` + light decoder | 5.402M |
| LitePT small + FastAdapter | 6.152M |

### ScanObjectNN 配置形态

| 配置 | 参数量 |
|---|---:|
| 当前默认 KPConvD-L encoder + classification head | 7.147M |
| 可选 KPConvX-L encoder + classification head | 16.814M |
| LitePT depth-matched | 9.803M |
| LitePT `(2,2,2,6,2)` | 5.278M |
| LitePT small + FastAdapter | 6.028M |

这些参数量已在当前源码上重新构造模型核对。S3DIS 的参数下降主要来自深层去掉 KPConvX convolution/kernel modulation、减少 block 深度和 light decoder；ScanObjectNN 当前默认基线实际是 KPConvD-L，因此 LitePT depth-matched 比 KPConvD-L 更大，不能宣称无条件减参，主要 Pareto 候选应比较 B0 与小深度 L1。是否能保持精度仍必须由正式训练确认。

---

## 8. 已完成验证

已实际执行：

- 新增与修改 Python 文件语法编译；
- 两个启动脚本 `bash -n`；
- PointROPE 零坐标恒等与反向传播；
- packed variable-length serialization 的点覆盖完整性；
- `z` / `z-trans` patch 构建；
- 192 和 256 通道 attention；
- 最后一块 192→256 通道过渡；
- serialization cache 跨 order 复用量化、同 order 复用布局并可按 forward 清空；
- S3DIS 形态的完整 synthetic pyramid 前向、decoder、head、反向；
- ScanObjectNN 形态的 hand-over stage、global pooling、head、反向；
- 跨 cloud attention 隔离、cloud-local 平移不变性与 CUDA 前反向（CUDA 可用时）；
- 与原有 FastAdapter 单元测试一起运行完整测试集。

尚未在容器里进行 S3DIS 或 ScanObjectNN 的完整正式训练，因此没有填写任何新 mIoU、OA、吞吐或 GPU 显存结果。

---

## 9. 已知边界与下一步

1. 当前只有 Morton `z` 与 `z-trans`，没有官方 Hilbert serialization。
2. 当前依赖 PyTorch SDPA；CUDA 环境可能自动走 fused backend，但不是 LitePT 官方 PointROPE/FlashAttention CUDA 实现。
3. 当前保留 KPConvX 的点金字塔和通道设计，不是官方 LitePT 的 spconv backbone。
4. 坐标量化和 patch sorting 已缓存，但每个 stage 每个新 batch 仍需构建两套 order；正式 profiling 必须确认这部分占比。
5. 正式训练前先跑 B0/L0/L0D/L1 的短程 smoke run，检查 loss、梯度、显存和吞吐，再投入完整 450/250 epoch。
