# MoECLIP 官方方案复现说明

参考：

- 论文：<https://arxiv.org/abs/2603.03101>
- 官方代码：<https://github.com/CoCoRessa/MoECLIP>

## 关键边界

MoECLIP 的 zero-shot 是“测试类别未见”，不是“训练时只有正常图”。官方工业设置
以 VisA 等辅助类别的真实正常/异常图片、图像标签和像素 mask 训练，然后评估不重叠
类别。官方实现没有 CutPaste、拼接或生成式异常。本工程因此也不制造异常；若赛事禁止
外部带标注异常数据，这套训练协议不能作为合规方案使用。

## 网络

冻结的 OpenAI CLIP ViT-L/14@336 接收 518 × 518 RGB 图。第 6、12、18、24 个视觉块
之后各插入一个 patch 路由 MoE。每个 MoE 有 4 个 LoRA 专家，rank=8、alpha=16，
每个 patch 选择 top-2；FOFS 将 LoRA A 固定到互不重叠的正交特征子空间，只训练
LoRA B 和 router。MoE 输出做范数匹配，再以 0.1 权重和原 token 插值。

四层特征各做 1 × 1、3 × 3、5 × 5 PAA，得到 12 组 patch token。每层共享一个
1024→768 分割投影；最后一层的 1 × 1 token 进入 LayerNorm、深度卷积和 1 × 1
卷积组成的图像分类投影。文本侧使用官方 normal/abnormal 状态词与两个模板，冻结
CLIP 文本编码器，只训练 768→768 文本投影。

训练损失为图像二分类交叉熵，加上 12 组分割损失。每组分割损失包含 focal、
normal-channel Dice 和 anomaly-channel Dice；再加 0.01 router balance 与 0.01
ETF 专家分离损失。默认 Adam、lr=5e-5、betas=(0.5,0.999)、20 epochs。

## 辅助数据

`configs/competition.yaml` 默认要求：

```text
data/moeclip_aux/
├── VisA/
│   └── ... image and mask files ...
└── metadata/VisA/full-shot.jsonl
```

JSONL 每行格式：

```json
{"image_path":"candle/train/good/000.JPG","mask_path":"","label":0,"class_name":"candle"}
{"image_path":"candle/test/bad/001.JPG","mask_path":"candle/ground_truth/bad/001.png","label":1,"class_name":"candle"}
```

路径相对 `auxiliary_dataset.root`。异常样本必须提供 mask，且元数据必须同时含正常
和异常样本。训练几何增强与官方代码一致：±30°旋转、15%平移、水平翻转、垂直翻转。

## Test-B 评分

对每组 patch 特征计算 `100 × patch·[normal, abnormal]`。每张原始异常图为
`(abnormal + 1 - normal) / 2`，12 张图相加并用 7 × 7、sigma=1 高斯平滑。
图像分支使用 abnormal 相似度；工业评分按官方方式将局部最大分数和图像分数各占
0.5。为了写入比赛的 [0,1] 分数和 8-bit mask，本工程最后按当前测试类别做单调范围
映射；这是提交格式适配，不是额外训练或测试集原型学习。
