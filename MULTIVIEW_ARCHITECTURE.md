# 五视角联合 Dinomaly 架构

本文档描述当前真正以“一个零件的五个固定相机视图”为一个样本的无监督异常检测
实现。它不使用类别 ID、异常训练标签、测试 mask 或 Test_A 统计量，也不要求不同
相机的像素坐标对齐。

## 1. 三代架构

### 原始 Dinomaly2

原始路径使用冻结的 DINOv2-register ViT 编码器、可训练 bottleneck、8 层
Transformer decoder 和 cosine reconstruction anomaly map。每张图片独立进入模型，
训练只重建正常特征。

### Category-generalized bottleneck

类别泛化版本把普通 bottleneck 替换为：

- `1024 → 384` 投影（ViT-L/14）；
- Compositional Reference Bank，只从正常特征学习可组合原语；
- Category-Free Router，不接收 category ID；
- local / component / global 三个空间尺度专家；
- reference balance、router balance、router entropy penalty 和 expert diversity
  正则。

Reference Bank 的 value 是学习到的正常原语。原始 encoder patch value 不会 residual
到 decoder，因此不存在把输入异常直接复制到输出的捷径。

### 五视角联合版本

训练和推理输入为：

```text
images:          [B_object, 5, 3, H, W]
view_ids:        [B_object, 5]，取值 0..4
valid_view_mask: [B_object, 5]
```

五张图先 reshape 为 `[B_object × 5, 3, H, W]`，共享同一个冻结编码器；patch
token 随后恢复成 `[B_object, 5, N_patch, 384]`。模型最终仍返回每个视角自己的：

```text
encoder_features: list[[B_object, 5, C, h, w]]
decoder_features: list[[B_object, 5, C, h, w]]
anomaly maps:     [B_object, 5, 1, H, W]
```

五张分割 mask 从各自的 feature pair 产生，不会生成公共 mask。

## 2. Robust context 与 Set Transformer

`MultiViewContextEncoder` 先按每个 patch token 的 L2 norm 排序，两端各 trim 一小部分，
再对中间 token 求均值。这样上下文不会被少量缺陷、强反光或轮廓离群 patch 主导。
每个 robust view context 加上可学习的固定相机 embedding，再经过 1–2 层
`SetAttentionBlock`。

缺失视角通过 attention key padding mask 排除；无效视角的 cross-view context 和
visibility 权重为零。`VisibilityAwareCrossViewFusion` 输出：

- 每个零件的 object-level set context；
- 每个视角独立的 cross-view context；
- 每个视角的 reliability / visibility soft weight；
- 用于诊断的 cross-view attention 权重。

cross-view context 只进入正常性 adapter：

- Category-Free Router 同时读取当前视角 robust context、cross-view context 与
  token dispersion；
- bounded FiLM 只调制 Reference Bank 重建出的 normal tokens；
- encoder 原始 patch value 不会跨过 Reference Bank；
- anomaly map 仍逐视角计算，所以只在一个相机可见的缺陷不会因为其他四个相机正常
  而被硬置零。

模型 forward 参数里只有图像、view ID、valid mask 和辅助损失开关，没有 category ID。

## 3. 数据分组与缺失视角

Real-IAD 根据 manifest 的 `object_id` 和文件名相机编号 `C1..C5` 分组，并在进入
DataLoader 前检查重复、缺失和越界。模型使用的 view ID 统一转换为 `0..4`。

`model.multi_view.missing_view_policy` 支持：

- `error`：发现缺失立即失败；
- `pad_and_mask`：补零图，并通过 `valid_view_mask` 从 attention、loss、metric 和
  prior 拟合中排除；
- `drop_incomplete`：整个不完整对象不进入数据集。

competition 的 `category/Sxxxx/0.png..4.png` 目录默认始终使用严格 `error`。

## 4. 无监督训练损失

主损失仍是每个有效视角的 normal feature reconstruction loss。新增项只使用正常
Train 对象：

- `context_consistency`：完整视角集合与随机 view-dropout 后的 object context 做
  cosine consistency；编码器只运行一次，第二次只运行 context 网络；
- `context_variance`：对 batch 内 set context 使用方差下界，防止样本全部塌缩到同一
  向量；
- `visibility_balance`：visibility 的 batch 使用率接近实际可用相机比例，防止永远只
  选择固定相机；
- `attention_entropy`：小权重惩罚完全塌缩的 cross-view attention。

这些项和已有四项辅助损失都以 graph-connected tensor 从 DataParallel replica 返回，
在主进程求均值；DDP 日志再做跨 rank reduce。YAML 权重为相对权重，最后统一乘
`training.generalized_regularization_weight`。

没有 anomaly-map 像素对齐损失，也没有要求五个视角 score 相等。

## 5. Train-only 正常边缘先验

训练完成后，流水线重新扫描正常 Train 数据，在原始 ViT/14 patch 网格上计算 raw
anomaly map。448 输入对应 `32 × 32`，artifact 不保存全部 448 分辨率浮点图。

统计项为：

```text
prior[category, view_id] = median_map, MAD_map
prior[view_id]           = seen Train 的 view-global median_map, MAD_map
```

seen / competition 类别优先使用 category-view prior。unseen 类别没有自己的 Train
数据，必须 fallback 到仅由 seen Train 汇总的 view-global prior；若 global 项也不存在，
该视角保持 raw map，不做静默伪造。

软校准公式为：

```text
z = (raw - median) / (MAD + eps)
gate = sigmoid((z - threshold) / temperature)
calibrated = raw * ((1 - blend) + blend * gate)
```

当 `blend < 1` 时，任何正的局部响应都不会被硬裁成相同的零值，局部排序能力得以
保留。校准发生在 patch 网格，之后才上采样、Gaussian、score 和 mask 标定。

`normal_prior.pt` 同时保存：format version、checkpoint 文件 SHA-256、训练语义 config
fingerprint、Train-only 标记、类别列表、view ID、网格尺寸、统计方法、样本数和校准
参数。恢复时 checkpoint 或 config 不匹配会直接报错，绝不静默复用。prior fitter 的
数据入口固定为 train dataset，不读取 test / Test_A。

## 6. Batch 与显存语义

多视角启用后，所有 training `micro_batch_size` 和 `effective_batch_size` 都表示对象数：

```text
effective view batch = effective object batch × 5
```

默认 unseen 配置为 12 个对象，即等效 60 张图；不是 64 个对象/320 张图。显存自动
探测会针对每个 candidate 真正运行 `[candidate,5,3,H,W]` 的 encoder、Set Attention、
decoder、loss、backward 和 optimizer step。`batch_tuning.json`、checkpoint 与日志同时
记录 object batch 和 equivalent view batch。

显存相对旧单图路径主要增加五倍 encoder 激活、五组 decoder token 和小型 context
网络；实际峰值必须以目标 GPU probe 为准。不要把旧实验的 view batch 直接与新实验的
object batch 比较。

## 7. YAML 接口

关键开关：

```yaml
model:
  multi_view:
    enabled: true
    num_views: 5
    context_dim: 384
    num_set_layers: 2
    num_heads: 6
    view_embedding: true
    view_dropout_probability: 0.2
    visibility_temperature: 1.0
    cross_view_dropout: 0.1
    missing_view_policy: error

training:
  multi_view_auxiliary_weights:
    context_consistency: 0.01
    context_variance: 0.001
    visibility_balance: 0.001
    attention_entropy: 0.001

evaluation:
  normal_prior:
    enabled: true
    resolution: patch
    category_view_enabled: true
    unseen_fallback: view_global
    statistic: median_mad
    threshold: 2.0
    temperature: 0.5
    blend: 0.8
    eps: 1.0e-6
```

单视角模型消融可设置：

```powershell
python run_unseen_pipeline.py `
  --set model.multi_view.enabled=false `
  --set evaluation.normal_prior.enabled=false `
  --set experiment.output_dir=outputs/unseen_single_view_ablation
```

关闭 multi-view 后，模型恢复 `[B,3,H,W]` forward、原 bottleneck/router 参数形状和旧
checkpoint state contract。normal prior 是独立消融项，所以要复现完全未经校准的旧
推理时也应关闭它。不同训练语义必须使用新 output directory。

## 8. 运行、恢复、推理和打包

Seen/unseen 全流程：

```powershell
python run_unseen_pipeline.py
```

执行顺序固定为：训练/恢复 normal model → 拟合或严格校验 normal prior → seen test →
unseen test → report。只评估已有 checkpoint：

```powershell
python run_unseen_pipeline.py --skip-train
```

Competition 分支执行：

```powershell
python run_competition_pipeline.py
python run_competition_pipeline.py --skip-train
python run_competition_pipeline.py --validate-only
```

competition 顺序为 Train normal model → Train prior → Test_A inference → per-view mask →
object score → `submission.csv` / `predicted_masks` / `submission.zip`。中断后只能复用同时
匹配 checkpoint SHA 和 config fingerprint 的 prior。

## 9. 建议消融

使用不同 output directory，逐项比较：

1. 单视角 generalized baseline；
2. 只开 Train-normal prior；
3. 只开 Set Attention，关闭四个 multi-view auxiliary 权重；
4. Set Attention + view dropout consistency；
5. 完整五视角 + prior；
6. category-view prior 与 view-global-only prior；
7. object score 的 legacy concat top-k、max、softmax 和 visibility-aware 聚合。

报告必须同时写 object batch、equivalent view batch、prior artifact fingerprint，并把
seen 与 unseen 分开。本文档不声称指标提升；只有实际完成对应 checkpoint 的正式评估
后才能报告收益。
