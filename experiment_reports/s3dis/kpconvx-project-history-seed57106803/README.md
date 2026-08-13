# KPConvX / LitePT / KTHA / GLSKF / DKS 项目全程实验总报告

## 0. 报告范围和读法

本报告整理从项目最初的模型路线判断，到截至 `2026-08-13` 的训练、诊断、
模块筛选和后续队列状态。主实验数据集是 S3DIS Area 5；早期工程还包含
ScanObjectNN main split 的对照实验，因此相关结果也一并记录。

报告把四种内容分开：

- **已证实**：能够由本地报告、CSV、验证日志或测试报告直接核对的事实。
- **合理推测**：由多个结果支持，但还没有多 seed、独立测试或因果证据确认的解释。
- **未证实/不能外推**：当前数据不能回答的问题，不能写成结论。
- **计划或进行中**：设计过但尚未完成，或当前队列仍在运行的工作。

原始训练日志、PLY、confusion 文件、进程状态和 checkpoint 只保留在本机
`Standalone/KPConvX/results/`；本报告不包含权重。运行时的精确 Git commit
没有被实验捕获，统一记为 `not recorded`，不能用发布本报告时的 commit 代替。

## 1. 最初的问题和研究假设

最初的问题不是“再换一种通用注意力”，而是：**KPConvD/KPConvX 的 kernel
几何怎样进入深层语义建模**。LitePT 的基本结构是高分辨率卷积、低分辨率
serialized PointROPE attention（`C-C-C-A-A`）。浅层卷积产生局部几何，后层
attention 通过 Morton patch、feature 和 PointROPE 建立 token 关系；在原结构中，
KP 邻域和 kernel response 没有作为显式 attention 输入持续传递。

初始机制假设是：Stage 2/3 到 Stage 3/4 的 handover 可能丢掉 kernel occupancy、
最近 kernel 分区、influence 和局部几何复杂度，造成 boundary/mixed-cell 区域的
语义重写。因此先做离线诊断，再做近似恒等初始化的短筛，最后才长训少数候选。

## 2. 项目阶段和实验协议

### 2.1 计划中的三阶段

1. **Stage 1：一天以内离线诊断**。固定 10 个 Area 5 房间，不训练新模型，记录
   每层 token 数、实际 latency、patch-KP overlap、KP edge cut、kernel occupancy
   entropy、error/entropy 相关、采样 attention entropy/distance、boundary、
   mixed-cell、per-class 和 Stage 2→3 residual ratio。
2. **Stage 2：短程 warm-start 筛选**。从 L0 checkpoint 初始化；先只训练新模块和
   head，再小学习率解冻全模型。候选为原 L0、M1 concat、M2 Q/K modulation、
   M3 relation bias，并加入等参数 MLP 和 shuffled geometry 控制。
3. **Stage 3：完整训练**。只对趋势、机制指标、参数和延迟均合理的两个候选做
   250 epoch。研发期不为每个变体执行长训。

### 2.2 评估协议的区别

- `full_identity`：Area 5 单视角、固定采样、无旋转/缩放/翻转；用于快速筛选，
  不是标准 10-vote full-cloud 成绩。
- 固定 10 房间：Stage 1 的机制诊断子集；不是完整 Area 5 验证。
- 标准 `10-vote full-cloud`：多视角投票后重投影到完整房间；用于 L0/L0D/L1 和
  M1 的最终对照。
- 同一 checkpoint 的 `true/shuffled/zero/room_mean/branch_off`：推理干预，
  用于判断模块在前向中是否真的被使用，不等价于重新训练模型。

Area 5 同时承担验证和测试，且 checkpoint 曾按测试结果或验证结果挑选，存在
checkpoint-selection bias。所有主要结果只有 seed `57106803`，不能估计训练方差。

## 3. 前置模型和工程实验时间线

### 3.0 完整运行清单

下表把“实验性运行”和“正式可比较结果”分开。失败启动、smoke 和中断运行保留
是为了说明工程过程，但它们不被当成模型性能结论。

| 时段/运行 | 目的 | 状态 | 是否进入正式比较 |
|---|---|---|---|
| L1 smoke、Morton 优化 smoke | 检查 LitePT 代码、序列化和数据流 | completed smoke | 否 |
| throughput baseline/async、batch 6/8/12、async-lean | 选择吞吐和有效 batch | completed probes | 否，只有配置经验 |
| L0 early interrupted/abandoned、L0D failed launch 两次 | 检查启动、恢复和异常路径 | failed/interrupted | 否 |
| L0 stability 300-step | 验证 `batch 12 + accum 2` 稳定性 | completed probe | 否 |
| KPConvX-L 450 epoch | 建立原始 KPConvX 基线 | completed after resume | 是，基线 |
| LitePT L0 450 epoch | 建立 KTHA 来源 checkpoint | completed | 是，主基线 |
| L0D/L1/L2 250 epoch | decoder、深度、PointROPE 路线消融 | completed | 是，路线参考 |
| L0 + FastAdapter S3DIS | 测试 grid/FastAdapter | interrupted/incomplete | 否 |
| Stage 1 fixed-10 | 机制诊断，不训练 | completed | 是，机制证据 |
| KTHA smoke、V1 warm10、shuffled、matched MLP | KTHA Stage 2 筛选 | completed/partial attempts | 是，筛选证据 |
| M1 joint20、同 checkpoint 干预、M1 10-vote | 检查 M1 是否真正使用 geometry | completed | 是，否定性证据 |
| M1 e180 | 低学习率长训尝试 | partial，只有 4 行 | 否 |
| KTHA V2 warm10 | 去除 semantic bypass 后重筛 | completed | 是，未晋级 |
| GLSKF warm10 五路筛选 | 测试深层语义到 KP kernel gate | completed | 是，否定性筛选证据 |
| GLSKF B1 scratch e180 | 排除 L0 warm-start 对新模块联合学习的约束 | completed，含 10-vote 和同 checkpoint 干预 | 是，否定性证据 |
| DKS Phase-A | 测试逐点动态 kernel scale 是否优于固定尺度和随机尺度 | preflight completed，3 seed × 6 arm 运行中 | 尚未，当前只有实现可用性证据 |

### 3.1 KPConvX-L 基线

最早先建立 KPConvX-L（`(3,3,9,12,3)`、heavy decoder、无 LitePT/PointROPE）作为
工程基线。450 epoch、AdamW、初始学习率 `1e-4`、weight decay `0.05`，运行曾中断
并从 checkpoint 恢复。epoch 348 的验证 mIoU 为 `75.7385%`，已测试 checkpoint 中
epoch 450 的 10-vote full-cloud mIoU 为 `71.4%`。它证明 KPConvX 主干本身可以稳定
运行，但不是后续 LitePT/KTHA 的严格单变量基线。

### 3.2 吞吐、采样和验证工程

早期做了 baseline、异步数据加载以及 `batch 6/8/12 + accumulation` 的吞吐试跑，
最后采用 `batch_size=12, accum_batch=2`（effective batch 24）作为许多 S3DIS 训练
的工程配置；另做了 300-step 稳定性试跑。后续验证优化加入了 bf16 AMP、显存/梯度
监控、周期 checkpoint、全量覆盖检查和可恢复队列。

验证慢的原因被定位为 `test.batch_size=1`、房间片段点数不稳定、邻域构建和 CPU/GPU
同步、概率回传 CPU 后投票/重投影/混淆矩阵以及写 PLY/confusion；显存占用低不等于
计算瓶颈在 GPU。`full_identity` 被明确作为筛选协议，避免把昂贵的 10-vote 误用在
每一个候选上。

### 3.3 LitePT L0/L0D/L1/L2/FastAdapter

LitePT L0（`C-C-C-A-A`、heavy decoder、PointROPE）是后续 KTHA 的统一来源。L0
450 epoch 的 33 个 checkpoint 标准测试范围为 `69.7%–72.1%`，均值 `71.53%`；
epoch 210 为最高 `72.1%`，epoch 450 为 `71.7%`。因此 KTHA warm-start 使用的
就是 **L0 epoch 210 checkpoint**，不是 450 checkpoint。

L0D 只把 decoder 改为 light，250 epoch、10-vote full-cloud 最佳为 epoch 150 的
`71.8%`；L1 使用 `(2,2,2,6,2)` 的轻量深度，最佳测试为 epoch 130 的 `69.6%`；
L2 在 L1 上关闭 PointROPE，250 个 checkpoint 全量测试范围约 `67.6%–68.7%`，
最高 `68.7%`，明显低于 L1/L0。这里 L0D/L1/L2 的训练协议和 decoder/深度同时变化，
只能作为路线消融，不能宣称每个差异都是单因素因果结果。

LitePT L0 + FastAdapter 的 S3DIS 训练曾多次中断，未完成计划训练，也没有正式的
最终 10-vote 结果；epoch 2 的验证值不能判断 FastAdapter 最终收益。ScanObjectNN
上 FastAdapter 有完整 250 epoch 记录（见第 10 节），但不能直接移植为 S3DIS 结论。

## 4. Stage 1 固定 10 房间诊断

诊断使用 L0 epoch 450 checkpoint，固定房间为 `WC_1`、`conferenceRoom_2`、
`hallway_2`、`hallway_8`、`lobby_1`、`office_1`、`office_21`、`office_38`、
`pantry_1`、`storage_3`。它不训练新模型，也没有根据这 10 个房间再次选 checkpoint。

### 4.1 已测数值

| 指标 | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---:|---:|---:|---:|---:|
| 平均 token 数 | 103889.7 | 21277.3 | 4405.3 | 906.6 | 173.6 |
| encoder latency (ms) | 19.08 | 6.19 | 5.00 | 23.76 | 7.73 |
| patch-KP overlap | - | - | - | 0.830 | 0.955 |
| KP edge cut rate | - | - | - | 0.170 | 0.045 |

其他机制值：token kernel occupancy entropy `0.698`，global entropy `0.947`；
point/error 与 entropy 的 Spearman 分别为 `0.020` 和 `0.025`；Stage 3 sampled
attention entropy 约 `0.943–0.959`、距离约 `2.13–2.17 m`，Stage 4 entropy `0.907`、
距离约 `3.34–3.37 m`；端到端 median latency `74.03 ms`。第一个 Stage 3 block 的
attention residual ratio `0.340`，MLP residual ratio `11.207`。

固定 10 房间 mIoU 为 `70.941941%`，OA 为 `90.251469%`；5 cm boundary error
`30.0149%`，10 cm `22.6164%`。Stage 2 mixed/pure cell error 为
`21.4007%/7.3365%`，Stage 3 为 `15.7306%/5.7012%`。

### 4.2 Stage 1 的作用和不能证明的事情

已证实：Stage 3 存在约 17% 的 KP 边被 serialized patch 切断，Stage 4 的切断率
较低；mixed-cell 和 boundary 是困难区域；Stage 3→4 是值得测试 geometry handover
的结构位置。

但 entropy 与错误几乎不相关，不能证明“高 kernel entropy 导致错误”，也不能证明
几何信息一定值得注入 attention。Stage 1 的正确作用是**缩小模块设计空间、指定
因果控制和目标 stage**，而不是直接给出新模块有效性的证据。

## 5. KTHA V1：三种模块和控制实验

KTHA V1 直接针对 Stage 3→4 handover：

- **M1 concat**：把 kernel geometry signature 与 token feature 拼接后送入 attention
  路径，近似恒等初始化。
- **M2 Q/K modulation**：用 geometry 调制 Q/K。
- **M3 relation bias**：用 geometry 生成 attention relation bias。
- **matched MLP**：只使用 semantic feature，匹配新增参数量，排除“参数增加”解释。
- **shuffled geometry**：同一房间内打乱 signature，保持 feature 和 signature 分布，
  破坏点与 geometry 的对应关系；只有 true 高于 shuffled 才支持空间对应关系有价值。

### 5.1 Stage 2 10 epoch warm-start

从 L0 epoch 210 初始化，先只训练模块和 head；bf16、batch 24、AdamW、learning
rate `5e-3`、weight decay `0.01`、`full_identity`。结果如下：

| 候选 | 最佳 epoch | 最佳 full_identity mIoU | 第 10 轮 |
|---|---:|---:|---:|
| M1 concat | 6 | 71.307692% | 70.823077% |
| M2 Q/K | 1 | 71.184615% | 70.984615% |
| M3 relation bias | 3 | 71.223077% | 71.061538% |
| M3 shuffled | 3 | 71.261538% | 71.176923% |
| matched MLP | 3 | 71.061538% | 70.923077% |

训练级 paired shuffled 的均值差：M1 true-shuffled `+0.018470` pp，M2
true-shuffled `-0.021530` pp。没有稳定的 geometry 因果优势。

### 5.2 M1 joint 20 epoch

M1 true 和 shuffled 从相同 warm-start 继续 joint 训练 20 epoch。true 最佳
`71.4538%`（epoch 7），shuffled 最佳 `71.7000%`（epoch 7）；final true
`71.1385%`，final shuffled `71.0615%`。20 轮中多数轮 shuffled 不低于 true，因而
不能把短期波动或微调轨迹解释为正确 geometry 的收益。

### 5.3 M1 同 checkpoint 推理干预

对同一个 true M1 joint-best checkpoint（internal epoch 7）执行五路单次
`full_identity`：true `71.447526%`、shuffled `71.448276%`、zero `71.451541%`、
room_mean `71.443162%`、branch_off `71.452649%`。总跨度仅约 `0.00949` 个百分点。
这是目前最强的前向因果证据：该 M1 checkpoint 在推理时几乎不依赖 geometry，甚至
旁路整个 residual 也几乎不改变结果。它不能排除模块在训练期提供正则化或改变优化
轨迹的可能性。

### 5.4 M1 标准 10-vote full-cloud

真几何 M1 joint-best 的标准 10-vote full-cloud mIoU 为 `71.869348%`，shuffled-trained
M1 为 `72.344162%`。相对 L0 epoch 210 的 `72.1%`：真 M1 `-0.230652` pp，
shuffled-trained `+0.244162` pp；shuffled 相对真 M1 高 `+0.474815` pp。

这不是 geometry 提升证据。shuffled-trained 的高分可能来自微调轨迹、额外容量、
checkpoint 选择或投票噪声；它还没有重复 seed，也没有独立 test selection。

### 5.5 180 epoch 长训尝试

曾尝试从 M1 joint-best 继续训练 180 epoch，配置为 joint、batch 12/accum 2、
bf16、learning rate `1e-4`、weight decay `0.01`、`full_identity`。队列只形成
4 个 validation row，未形成完整 180 epoch 产物，属于 partial/failed attempt，
不能写成 M1 的 180 epoch 结果，也不能用来判断低学习率是否有效。

## 6. KTHA V2：避免 semantic bypass 的设计和筛选

V2 的设计原则是让 geometry 只生成 pairwise attention bias，不再把 geometry
signature 直接 concat 到 semantic feature，减少 M1 的“绕过 geometry”路径。

- 每个 token 使用 `3K+2=131` 维 geometry-only signature：kernel occupancy、
  每 kernel residual-distance mean/std、valid-neighbor ratio、mean influence mass。
- 几何投影生成 per-head embedding，patch centering 去除常量几何。
- 零初始化 diagonal pairwise metric；zero/room-mean signature 结构上生成零 bias。
- `matched_mlp_v2` 每 attention block 精确匹配新增参数量；12 blocks 共新增
  `101,376` 参数。

V2 true、shuffled、matched MLP 从 L0 epoch 210 各做 10 epoch warm-start，三路队列
均成功。true 最佳 `71.035988%`（epoch 10），10 轮均值 `70.915385%`；shuffled
最佳 `71.245288%`（epoch 1），均值 `71.016923%`；matched MLP 最佳 `71.243080%`
（epoch 8），均值 `71.050769%`。true-shuffled 平均差 `-0.101538` pp，true-matched
平均差 `-0.135385` pp，true 仅在 10 轮中分别赢 `2/10` 和 `3/10`。

预设晋级门槛是 true-shuffled 至少 `+0.3` pp；实际方向相反，因此 V2 未通过短筛，
不建议进入联合/180 epoch。V2 的 768 个 pairwise weights 全部非零，mean absolute
weight `0.03552376`、最大 absolute weight `0.46712309`，说明它确实更新了，但
“学动”不等于“带来有效几何信息”。

## 7. GLSKF 后继实验

GLSKF 是 KTHA 之后的另一条机制假设：让 Stage 4/5 的深层语义特征调制 Stage 3
decoder skip 上的有效 KPConvD kernel gate。它与 KTHA 互斥，避免两条因果路径混杂。

本次 Stage 2 使用同一个 L0 epoch 210、bf16、batch 24、learning rate `5e-3`、
weight decay `0.01`、10 epoch、`full_identity`，并设置 head-only L0 control、
true kernel gate、shuffled context、room_mean context、matched MLP 五路。五路训练和
最终汇总均已完成，每一路都有 10 行确定性单视角全量 Area 5 验证结果。汇总任务首次
因直接执行无权限的 Python 文件而以 exit 126 失败；改为由项目 Python 解释器显式
执行后，可恢复队列跳过五个成功训练，只补做汇总。该工程故障没有改变训练权重或指标。

### 7.1 Stage 2 五路 warm10 结果

| job | 最佳 epoch | 最佳 full_identity mIoU | 10 轮均值 | 第 10 轮 |
|---|---:|---:|---:|---:|
| L0 head control | 2 | 71.276923% | 71.128462% | 71.130769% |
| GLSKF true | 2 | 71.215385% | 70.865385% | 70.669231% |
| GLSKF shuffled | 2 | 71.238462% | 70.890769% | 70.669231% |
| GLSKF room_mean | 6 | 71.200000% | 70.903077% | 70.676923% |
| GLSKF matched MLP | 6 | 71.200000% | 70.871538% | 70.638462% |

在 true 自身最佳的 epoch 2，true 相对 L0 head、shuffled、room_mean、matched MLP
分别为 `-0.061538`、`-0.023077`、`+0.061538`、`+0.123077` pp。按 10 个配对 epoch
求均值，true 相对四路分别为 `-0.263077`、`-0.025385`、`-0.037692`、
`-0.006154` pp；true 在 10 轮中没有一次超过 L0 head，只在 `3/10` 轮超过 shuffled。

因此，GLSKF true **没有通过**预设的筛选条件：它既没有超过 L0 head control，
`true-shuffled` 也远未达到 `+0.3` pp。当前结果不支持“正确对齐的深层语义上下文
给 kernel gate 带来短程正收益”。这属于单 seed、warm-start、按最佳 epoch 描述的
否定性筛选证据，不等价于证明 GLSKF 在所有训练方式下都无效。

### 7.2 B1 scratch 180 epoch：最终结果和 L0 差距

在 Stage 2 完成后启动了 B1：`kernel_gate + true context` 从随机初始化做 180 epoch
全模型联合训练，不加载 L0 或其他 checkpoint。配置为 seed `57106803`、300
steps/epoch、batch 12/accumulation 2、bf16、AdamW、learning rate `5e-3`、weight
decay `0.01`，每个 epoch 做一次 `full_identity` 单视角全量验证；训练完成后自动用
单视角全量最佳 checkpoint 做标准 10-vote full-cloud 测试。

该实验的主要目的不是推翻 Stage 2 门槛，而是检查一个未证实的替代解释：L0 已收敛
表示可能把近似恒等初始化的新 gate 限制在原有优化盆地，使 10 epoch warm-start
低估模块与主干共同学习的能力。B1 允许 KPConvX/LitePT 主干、Stage 4/5 语义上下文和
Stage 3 decoder kernel gate 从头协同形成，检验完整训练时的优化可行性和最终性能。

B1 已完成全部 `180` 个 epoch。训练日志中的最佳 `full_identity` checkpoint 是内部
epoch 81，mIoU 为 `65.730769%`；epoch 180 为 `65.261538%`。自动 10-vote 使用
epoch 81 的 best checkpoint，第 10 票 full-cloud mIoU 为 `66.854187%`，投票过程
最高为 Vote 8 的 `66.927478%`，最终报告按一位小数显示为 `66.9%`。

与 L0 的差距必须按相同协议分别计算：

| 协议 | L0 epoch 210 | B1 epoch 81 | B1-L0 |
|---|---:|---:|---:|
| deterministic `full_identity` 单视角 | 71.034802% | 65.735934% | -5.298868 pp |
| 标准 10-vote full-cloud | 72.1% | 66.854187% | 约 -5.245813 pp |

因此 B1 没有获得性能正收益，从头联合训练也没有支持“L0 warm-start 束缚了 GLSKF，
scratch 可以释放其收益”的替代解释。但这里仍有方法学边界：L0 来自 450 epoch 训练中
选出的 epoch 210，B1 是 180 epoch scratch，二者并非严格匹配的 scratch baseline 与
单变量模块对照，所以不能把全部 `5.25–5.30` pp 差距因果归因于 GLSKF 本身。

对同一 B1 epoch-81 checkpoint 的干预结果为：true `65.735934%`、shuffled
`65.739424%`、room mean `65.722318%`、zero context `65.703286%`、neutral gate
`65.662362%`、branch off `24.258799%`。true 与 shuffled 只差 `-0.003490` pp，
相对 room mean、zero context、neutral gate 也只有 `+0.013616`、`+0.032648`、
`+0.073572` pp；这些差异远小于预设的 `+0.3` pp 机制门槛。branch off 则下降
`41.477135` pp，说明模型强烈依赖整个 GLSKF correction branch，但几乎不依赖深层
上下文与空间位置的正确对应。更准确的结论是：B1 学会了使用该 correction 路径，
没有证据表明它学会了利用对齐的深层语义来动态调制 kernel geometry。

### 7.3 静态代码审查后的修复和解释边界

一次不含运行环境和实验产物的外部静态审查确认：KTHA/GLSKF 的张量方向、kernel
assignment、shadow 屏蔽、空间上采样、临时权重 gate 和恒等初始化没有发现原理性
错误；但发现了会影响对照解释的实现和方法学问题。下列确定缺陷已在不改变正在运行
B1 权重轨迹的前提下修复，并由聚焦测试覆盖：

- GLSKF baseline 关闭模块后仍保留 `module_head`，导致网络构造必然失败；现已强制
  `glskf_train_mode=joint`，并加入 baseline 构造回归测试。外部 L0 checkpoint 默认
  路径也改为真实的 L0 epoch 210，而不是错误假定 GLSKF run 内含该权重。
- KTHA/GLSKF shuffle 改为局部 RNG 作用域：生成可复现的 packed-cloud 内置换后恢复
  CPU/CUDA 全局 RNG，避免把后续模型随机流差异混入 true/shuffled 对比。
- 每次 `KPNeXt.forward` 开始时清除上一批 `shared_kp` 运行态几何，GLSKF 消费前检查
  本批 nearest-kernel assignment，返回前释放引用，避免相同 shape 的后续 batch
  静默复用陈旧缓存。
- 修正 GLSKF gate 初始化说明：第一个优化步先给外层 scale 梯度，gate 参数要等 scale
  离开零后才获得梯度。
- 补充 KTHA V1 shadow-neighbor、cached/recomputed 一致性、relation-bias 常量签名
  不变性、shuffle RNG 隔离，以及 GLSKF baseline/cache/RNG 测试。聚焦套件结果为
  `97 passed, 25 subtests passed`；CUDA attention 有一条已知 cuDNN stride warning，
  没有测试失败。

仍需保留的方法学限制如下，不能通过代码修复追溯性地改变已经完成的 Stage 2：

- 训练按约 2.1 m packed crop 做 shuffled/room-mean，`full_identity` 验证按完整房间
  分段，已训练控制臂存在干预尺度漂移。它们可作探索性筛选，不能单独建立空间对应
  关系的因果结论；B1 true 本身使用 `context_control=none`，不受该问题直接影响。
- KTHA V1 `relation_bias` 对 zero 或 patch 内常量签名只产生 softmax 可消去的常数
  bias，所以 zero/room-mean/branch-off 不是三个独立控制。这个限制不影响已推进的
  M1 concat 同 checkpoint 干预。
- 现有 KTHA `matched_mlp` 精确匹配 relation-bias 参数量，不匹配 M1 concat；真实
  模型统计为 concat `263,948` 对 matched MLP `66,144` 个 KTHA 参数。因此 M1 与
  matched MLP 不能被解释为严格容量消融。
- 单 seed、Area 5 checkpoint selection 和短筛 best-epoch 选择仍然存在。外部审查
  提到的具体效应量或 seed 方差范围没有本地数据或引用验证，未作为事实写入结论。

B1 后处理按训练主脚本 PID、`/proc` start ticks 和最终产物三重条件核验。180 行验证、
best/latest checkpoint、自动 10-vote、外部 L0 参考及六路同 checkpoint 干预均已完成。
后处理曾因 baseline job 缺少执行权限、公开标签 `shuffled` 与内部枚举 `shuffle` 不一致，
以及验证默认配置覆盖干预字段而失败；修复后只补跑失败项，已有成功结果没有重算或覆盖。

### 7.4 DKS：Dynamic Kernel Scale 新模块

DKS（Dynamic Kernel Scale，动态核尺度）直接作用于 KPConvD/KPConvX 的核几何，而不是
再增加一种 attention。它为每个 query point 生成一个有界各向同性尺度 `alpha`，在
保持预计算 KNN 邻居集合不变的情况下，用 `alpha` 除以中心化后的邻居相对坐标，从而
改变最近 kernel point 分配和 influence weight。当前只作用于 Stage 3 的第一个 KP
block，尺度范围为 `[0.5, 1.2]`；learned 路径初始化为精确 `alpha=1`，可从 L0 epoch
210 近似恒等 warm-start。DKS 与 KTHA、GLSKF 在同一 run 中互斥，避免混合机制归因。

Phase-A 使用 3 个 seed（`57106803`、`12345`、`98765`）和 6 个实验臂，每路训练
10 epoch，并只比较预先指定的 epoch-10 指标，而不是各自挑最佳 epoch：

| arm | 作用 | 要排除的替代解释 |
|---|---|---|
| `l0_head` | 不启用 DKS，只训练 L0 head | 优化器重启/head 微调 |
| `fixed_1.0` | 恒等尺度 | 加入 DKS 代码路径本身 |
| `fixed_0.8` | 全局缩小核尺度 | 固定半径调优 |
| `fixed_1.15` | 全局放大核尺度 | 固定半径调优 |
| `random` | 随机逐点尺度 | 任意扰动或正则化 |
| `learned` | 特征预测逐点尺度 | 待检验的动态几何机制 |

机制晋级条件不是 learned 单独超过 L0，而是 learned 在多 seed 下同时超过最佳 fixed arm
和 random，并超过预先声明的噪声门槛。`learned ~= fixed_best` 只支持全局 kernel radius
调优；`learned ~= random` 则说明随机扰动或正则化仍是充分解释。

GPU 预检已经完成：L0 与 fresh DKS `alpha=1` 的 full-room confusion 和 mIoU 精确一致；
100-step learned run 的 gate 从 0 移到 `0.112953`，最终 alpha std 为 `0.007550`。
这证明恒等初始化、梯度路径、checkpoint 和诊断产物可用，但 alpha 变化仍小，也不证明
DKS 有性能收益。截至 `2026-08-13 08:16 +08:00`，正式 Phase-A 队列完成 `1/18`，
第 2 个 arm 完成 `8/10` 个 epoch，仍在运行；完整结果出来前不做模型收益判断。

## 8. 目前能得出的结论

### 8.1 已证实

1. L0 epoch 210 是当前单 seed 标准 10-vote full-cloud 基准峰值 `72.1%`，后续 KTHA
   warm-start 确实从这个权重开始。
2. Stage 3 存在明显 KP edge cut（约 17%），但 kernel entropy 与 error 的相关性
   接近零；这支持“测试 handover”的动机，不支持“entropy 导致错误”。
3. M1 true 没有超过 L0 epoch 210；同 checkpoint 五路推理干预几乎不变，说明 M1
   前向几何依赖很弱。
4. M1 shuffled-trained 的 10-vote 分数高于 true M1，但这不能归因于正确 geometry。
5. V2 去除了 semantic concat bypass，参数正常更新，但 true 在短筛中低于 shuffled
   和 matched MLP；未达到预设晋级门槛。
6. GLSKF Stage 2 五路均已完成；true 低于 L0 head，且 true-shuffled 的最佳轮配对差
   为 `-0.023077` pp，没有达到 `+0.3` pp 门槛。
7. GLSKF B1 已完成 180 epoch 和 10-vote；相对 L0 在同协议单视角和 10-vote 下分别
   低 `5.298868` 和约 `5.245813` pp，scratch 训练没有产生正收益。
8. B1 同 checkpoint 的 true/shuffled/room-mean/zero-context 几乎不变，但 branch-off
   严重下降；模型依赖 correction branch，却没有可检测的对齐深层上下文依赖。
9. DKS 预检证明实现与恒等 warm-start 可用；Phase-A 尚未完成，性能结论未知。
10. L0D 的 light decoder 仍达到 `71.8%`，而 L1/L2 的轻量深度或无 PointROPE 结果
   更低；说明不能只按“更轻”推断更好。

### 8.2 合理推测

- M1 concat 可能主要学习普通 feature residual，geometry signature 在训练时只作为
  可绕开的附加容量或正则项。
- V2 虽然结构上更严格，但当前 signature、注入位置或 pairwise metric 可能冗余、
  噪声较大，或者被 L0 的 PointROPE/深层语义覆盖。
- observed gain 更可能来自优化器重启、head 微调、普通参数容量、checkpoint 选择和
  10-vote 波动，而不是已经证明的空间对应关系。
- Stage 1 指向的瓶颈也可能在 decoder 边界恢复或 patch grouping，而不一定是 Q/K
  公式；因此继续堆叠通用 attention 的理由不足。

### 8.3 不能确认

- geometry 在其他 seed、数据集、目标 stage、decoder 或 boundary 路径是否有效。
- shuffled-trained M1 的 `72.344%` 是否可复现，是否只是选择偏差。
- B1 的大幅落后中有多少来自 GLSKF、180/450 epoch 时长差异或训练调度差异；没有严格
  匹配的 scratch L0/shuffled/matched 对照不能分解。
- DKS learned 是否稳定超过最佳固定尺度和 random；Phase-A 尚未完成。
- 单 seed、Area 5 复用和不同评估协议下，任何约 `0.1–0.3` pp 差异是否统计显著。

## 9. 当前决策和建议

1. 不把 M1/V2 直接晋级为 Stage 3 长训候选；KTHA 的当前证据更适合写成“几何
   handover 未被当前实现证实”，而不是“几何没有作用”。
2. GLSKF Stage 2 和 B1 scratch 均无正收益，且空间对齐上下文没有通过干预门槛；不再
   把现有 GLSKF 作为长训候选，除非出现新的机制证据，而不是仅调整训练时长。
3. 当前优先完成 DKS 的 3 seed × 6 arm Phase-A。只有 learned 同时超过最佳 fixed 和
   random，才进入同 checkpoint 干预或更长训练。
4. 如果 DKS 也未过门槛，下一步优先检查 Stage 3 patch grouping、decoder boundary
   路径和 kernel 尺度/签名的可预测性，而不是继续变换 Q/K 形式。
5. 后续候选原则上只有通过预设因果门槛才做 180/250 epoch；最终候选需要独立 checkpoint
   选择、至少 3 seeds 和统一的 10-vote full-cloud 协议。

## 10. 复现和资源记录

- Seed：`57106803`。
- 主要环境：Python `3.10.20`、PyTorch `2.5.0`、CUDA `12.4`、NumPy `1.26.4`。
- 主要训练 GPU：NVIDIA RTX 4090 D，24 GB；早期部分运行使用 RTX 4080 SUPER，
  训练/验证资源不可直接横比。
- KTHA/GLSKF warm-start 常用：effective batch 24、bf16、AdamW、weight decay
  `0.01`；原 L0 为 batch 12/accum 2、bf16 关闭、weight decay `0.05`。
- 训练期间有中断/恢复和失败启动；队列脚本会保存 per-job 日志、exit code、时间和
  GPU/内存/磁盘快照，并对独立任务继续执行；依赖任务则在上游产物不可用时跳过。

## 10.1 早期 ScanObjectNN 对照

ScanObjectNN main split 使用 250 epoch、AdamW、learning rate `5e-4`、weight decay
`0.01`、batch 32/accum 2、10-vote。四个模型的最佳观测如下；它们与 S3DIS 的
mIoU 不是同一个指标，不用于证明 S3DIS 几何模块有效。

| 模型 | 最佳 checkpoint | 10-vote OA | 10-vote mAcc | 备注 |
|---|---:|---:|---:|---|
| KPConvX-L | epoch 199 | 89.3% | 88.2% | best observed |
| KPConvX-L + FastAdapter | epoch 190 | 88.7% | 87.3% | 有 OOM 后自适应 batch |
| KPConvD-L | epoch 249/250 | 88.9% | 87.3% | 对照 |
| KPConvD-L + FastAdapter | epoch 250 | 89.1% | 87.8% | 对照 |

这些 checkpoint 是在同一测试 split 上逐个测试后选择的，存在 test-set checkpoint
selection bias；单 seed 差异也不能说明稳定排名。

## 11. 证据索引

- L0 基线：[litept-l0-450-seed57106803](../litept-l0-450-seed57106803/)
- L0D：[litept-l0d-250-seed57106803](../litept-l0d-250-seed57106803/)
- L1：[litept-l1-250-seed57106803](../litept-l1-250-seed57106803/)
- Stage 1：[l0-epoch450-stage1-fixed10-20260811](../l0-epoch450-stage1-fixed10-20260811/)
- KPConvX-L：[kpconvx-l-area5-seed57106803](../kpconvx-l-area5-seed57106803/)
- 前置路线和后续计划：[l0-seed57106803-next-diagnostics](../l0-seed57106803-next-diagnostics/)
- KTHA 设计：[Standalone/KPConvX/tools/KTHA_V2.md](../../../Standalone/KPConvX/tools/KTHA_V2.md)
- GLSKF 设计：[Standalone/KPConvX/tools/GLSKF_EXPERIMENT.md](../../../Standalone/KPConvX/tools/GLSKF_EXPERIMENT.md)
- DKS 设计：[Standalone/KPConvX/tools/DKS_EXPERIMENT.md](../../../Standalone/KPConvX/tools/DKS_EXPERIMENT.md)
- 完整关键数值见同目录 [metrics.csv](metrics.csv)。
