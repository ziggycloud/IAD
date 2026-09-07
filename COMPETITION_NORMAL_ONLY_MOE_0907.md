# Normal-only unseen Patch-MoE（0907）

该分支保持 seen 类 Dinomaly 路径不变，仅替换 Test_B 未见类别路径。训练只读取
`data/competition/Train` 的正常五视角图像；不读取异常图/Mask，不制造异常图，
不执行 CutPaste，也不加载第三方异常检测 checkpoint。唯一新增基础权重是公开的
OpenAI CLIP `ViT-L-14-336`，由 OpenCLIP 在首次运行时自动下载。

## 网络

```text
image -> frozen OpenCLIP multi-layer patches -> weighted layer fusion
      -> Top-2 / 4-expert low-rank Patch MoE -> PAA scales 1,3,5
      -> background / normal / broken category-conditioned text anchors
      -> foreground probability + semantic anomaly probability
      -> fixed 64-D projection -> category/view robust normal prototype
      -> prototype deviation + centered CLIP semantic evidence
      -> absolute anomaly mask and top-1% image score
```

四个 MoE 专家共享冻结 CLIP 输入，具有互相正交的冻结降维基；可训练部分只有
router、低秩升维、PAA 前的残差尺度、层权重和文本 adapter。训练损失为正常 patch
三分类、正常图分类、normal-vs-broken margin、五视角状态一致性、router balance、
expert diversity 和 prompt anchor。异常文本只作为语义负锚点。

## Test_B 类别级评分

推理先缓存一个类别全部图片的 64-D FP16 patch 特征。每个相机视角独立计算全局
几何中位中心，保留距离中心最近的 60% 样本作为伪正常集合，再建立空间 patch 原型。
局部余弦距离相对伪正常集合的 99% 分位数做绝对标定；CLIP 异常响应也先减去伪正常
集合的 95% 分位数。最终 mask 为：

```text
foreground^1.5 * sigmoid(4 * prototype_z + 0.5 * semantic_z - 3.0)
```

图像分数是低分辨率 mask 的 top-1% 均值，五视角仍由提交层的 max/mean 聚合。
任何 mask 都不会进行逐图 min-max 拉伸。

## 启动

```bash
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  run_competition_pipeline.py \
  --set runtime.multi_gpu_strategy=ddp \
  --test-b
```

权重写入独立的 `zero_shot_normalonly/checkpoints`，不会误读 0907 的旧异常监督
checkpoint；格式版本也已经升级为 3。
