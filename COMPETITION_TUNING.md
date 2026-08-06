# 比赛分割调优与训练稳定性说明

五视角网络、数据 shape、辅助损失、正常先验 artifact 与恢复规则的完整说明见
[MULTIVIEW_ARCHITECTURE.md](MULTIVIEW_ARCHITECTURE.md)。

## 1. 为什么 P-AUROC 高，但 P-AUPR / P-F1max 只有 0.5–0.6

这是典型的像素类别极不平衡现象。背景像素远多于微小缺陷，少量物体边缘被打成高分时，P-AUROC 仍可能很高；但这些边缘像素的面积常常比真实缺陷还大，会直接降低 precision，进而压低 P-AUPR 和 P-F1max。

当前模型容易出现边缘假阳性的几个原因：

- 448 输入配合 ViT/14，原始异常响应只有 32×32 patch 网格；双线性上采样会把边界响应扩散开。
- 原配置 `gaussian_sigma: 4.0` 会进一步把窄边缘变成宽光晕。
- 两组特征异常图等权平均，其中较浅的一组更容易响应纹理、轮廓和位置抖动。
- Loose Loss 最终丢弃率 0.9，会长期聚焦最难的 10% 正常 token；在全正常 Train 上，这些 token 往往正是物体边缘。
- 若把训练步数从 6000 直接改成 15000，同时保持峰值学习率 1e-3，余弦退火会让模型在高学习率区间停留更久。

## 2. 已加入的低风险调优接口

`configs/competition.yaml` 已提供：

- 优化器：`stable_adamw`、`adamw`、`adam`。
- 学习率：`cosine`、`linear`、`polynomial`、`constant`、`step`、`multistep`，均可配 warmup。
- 异常图两组特征权重：`evaluation.anomaly_map_layer_weights`。
- 上采样方式开关：`evaluation.anomaly_map_align_corners`。
- 高斯平滑：建议先比较 sigma 0、1、2，原来的 4 通常不利于小缺陷 precision。
- 梯度保护：记录裁剪前总范数和每个参数组范数，超过阈值时跳过 optimizer step。

稳定版默认值是 LR 3e-4、Adam epsilon 1e-8、500-step warmup、cosine、Loose Loss 2000-step 渐进到 0.7、sigma 2、两层权重 `[0.35, 0.65]`。五视角版本使用独立的 `*_v3` 输出目录，避免恢复单视角 optimizer 状态。

建议一次只改变一个变量，优先顺序如下：

1. sigma：`0 / 1 / 2`。
2. 特征权重：`[1,1] / [0.5,0.5] / [0.35,0.65] / [0.2,0.8]`。
3. Loose Loss：final discard `0.5 / 0.7 / 0.9`，warmup `1500 / 2000 / 3000`。
4. LR：`1e-4 / 3e-4 / 5e-4`。

不要使用 Test_A 标签反向训练或按测试标签选阈值。若比赛允许本地验证，应该仅从 Train 的正常样本中留出少量 normal validation，用于监控正常边缘响应和训练稳定性；它不能直接估计缺陷 P-AUPR。

## 3. 已实现的五视角联合架构

当前默认实现不拼接 RGB，也不施加跨相机像素对齐约束。一个 `Sxxxx` 的五张图组成
一个 object batch 元素；共享冻结 DINO 编码器产生五组 patch token，再由带 view
embedding 的 robust pooling、两层 Set Attention 和 visibility head 产生 object context、
per-view cross-view context 与可靠性。跨视角信息只通过 router conditioning 和有界
FiLM 调制 normality adapter，原始 encoder patch value 不会 residual 到 decoder。

训练仍只使用正常 Train 的逐视角 feature reconstruction，并可配置
view-dropout consistency、context variance、visibility balance 和 attention entropy。
网络从不读取 category ID。推理继续生成五张独立异常图；visibility-aware object score
混入可配置 max 分量，因此某一视角独有的高异常不会因其余视角正常而被硬置零。

正常边缘先验也已实现：先在 32×32 patch 网格上，用 Train 正常图拟合每个
category/view 的 median 与 MAD，并同时保存 view-global fallback。校准使用平滑 sigmoid
gate 而不是 hard ReLU，保留局部排序。artifact 记录版本、checkpoint SHA、配置指纹、
类别、视角、尺寸与统计参数；任何不匹配都会报错。Test_A 从不进入 prior dataloader。

## 4. 这次梯度突然变高的解释

日志里的 `grad` 是 `torch.nn.utils.clip_grad_norm_` 的返回值，即**裁剪前总梯度范数**。原配置实际仍把梯度裁到 0.1，所以 `15234961` 不等于直接用一千多万的梯度更新参数。

不过这不是单纯的日志显示问题：loss 从约 0.018 持续升到 0.18–0.68，说明模型确实进入了不稳定状态。最可疑的组合是：

- 15k steps + 1e-3 LR，使 step 2140 时 LR 仍约 9.65e-4；
- BF16 下 Adam epsilon 设为 1e-10；
- Loose Loss 很早达到 0.9，把梯度集中到少量困难边缘 token；
- generalized reference/router 辅助项可能在该 batch 发生尖峰。

新日志会输出 `reconstruction_loss`、`regularization_loss`、`auxiliary_losses`、`gradient_group_norms`、`grad_norm_pre_clip`、`gradient_clip_scale` 和 `optimizer_step_skipped`，可以区分是 decoder 重建、bottleneck，还是广义路由项先爆炸。

当前运行建议停止，不使用 step 2140 之后的 checkpoint。step 2000 是日志中最后一个明确稳定的断点，但改变配置后不应恢复旧 optimizer 状态；稳定版应从新的输出目录重新训练。

## 5. 运行和覆盖示例

默认五视角配置（effective object batch 12，等效 view batch 60）：

```bash
python run_competition_pipeline.py
```

只跑一个消融，例如关闭高斯并恢复等权异常图：

```bash
python run_competition_pipeline.py \
  --set experiment.output_dir=outputs/competition_ablation_sigma0_equal \
  --set evaluation.gaussian_sigma=0.0 \
  --set evaluation.anomaly_map_layer_weights='[1.0,1.0]'
```

任何改变训练语义的实验都应使用新的 `experiment.output_dir`，防止自动恢复不兼容的 checkpoint。
