# Dinomaly2 在 Real-IAD Variety 上的评估报告

状态：正式 pinned-upstream 配置正在运行 100,000 步训练；训练完成后会
自动执行 160 类评估。当前不提供伪造、smoke 或外推指标。

## 评估口径

按照 Real-IAD Variety 论文，对每个类别分别计算并宏平均：
I-ROC、I-PR、I-F1max、P-ROC、P-PR、P-F1max、P-PRO。P-PRO
积分上限为 30% FPR。图像分数采用异常图最高 1% 像素的均值。

## 配置摘要

- 模型：Dinomaly2-B，DINOv2-register ViT-B/14
- 输入：280 x 280（适配 8 GB 显存的上游 Real-IAD Variety 配置）
- 有效/micro batch：16 / 16（实测 peak reserved 4,728 MiB）
- 训练：100,000 optimizer steps，StableAdamW，FP32
- 学习率：严格按 pinned multiview preview；首步 0，100 步 warmup，
  随后保持 2e-3
- 评估：FP32，逐类别动态 min/max 的 1,000-bin adeval
- mask：上游 bilinear + nonzero 语义

## 论文参考值（不是本次实测）

Real-IAD Variety 论文表 3 中的 **Dinomaly（不是 Dinomaly2）** 为：
85.4 / 97.2 / 94.5 / 91.5 / 42.8 / 45.8 / 75.6（百分数，顺序同上）。
本次模型、输入和训练实现不同，不能把这些数值当成本次结果。

## 架构改进观察框架

完整结果后将重点检查：

1. I-ROC 与 P-PR 的差距，用于判断小缺陷定位是否受 14 x 14 patch 和低输入分辨率限制。
2. 类别间方差和最差类别，用于判断统一 decoder 是否存在容量-多样性冲突。
3. 五视角的图像分数与对象分数差距，用于判断独立视角推理是否需要显式跨视角融合。
