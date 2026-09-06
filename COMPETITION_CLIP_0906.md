# 多视角 Dinomaly + CLIP 方案（0906）

本配置以 `codex/competition-pipeline` 的五视角模型为主干，并仅对 Train 中不存在的
Test 类别增加冻结 OpenCLIP 语义残差。目标是保留 seen 分类/分割和干净热力图，同时
补充 unseen 缺陷语义。

## 阶段划分

1. 用五视角对象 batch 训练 category-generalized Dinomaly。DINOv2-register ViT-L/14
   保持冻结，CLIP 不加载、不反向传播。
2. 用 Train 正常图拟合 DINO category/view median-MAD prior。
3. 只有目标测试集包含 unseen 类别时，才用同一批 Train 正常图拟合 CLIP
   view-global median-MAD prior。
4. seen 类沿用纯多视角 DINO 推理；unseen 类在 DINO prior 后增加 CLIP 残差。
5. 每个相机独立输出 mask，对象分数继续使用 visibility-aware 聚合。不同相机不做
   像素级平均，因为它们没有空间对齐。

## Unseen 融合

CLIP 使用 ViT-B/16 最后四个中间层。权重 `[0.1, 0.2, 0.3, 0.4]` 降低浅层纹理和
边缘噪声，突出深层语义。normal/broken prompts 分别聚合 top-3 相似度，并使用
`temperature=0.07` 得到 broken probability。

融合前依次执行：

- view-global CLIP normal prior，抑制训练正常图中反复出现的相机/背景响应；
- `broken_threshold=0.5`；
- 基于离 0.5 距离的置信度门控，默认平方衰减不确定 patch；
- 来自 prior-calibrated DINO map 的软前景门控，背景 floor 为 0；
- 在 DINO 量纲内增加语义残差，最后进行一次轻量 Gaussian smoothing。

CLIP 不会单独点亮没有 DINO 支持的背景，但仍能在 DINO 已定位到的物体区域内增强
scratch、missing part、wrong assembly 等 zero-shot 语义。

## 默认训练规模

```yaml
training:
  total_steps: 6000
  effective_batch_size: 12  # objects; 60 views per optimizer step
  amp_dtype: bfloat16
  optimizer:
    type: stable_adamw
  scheduler:
    type: cosine
```

CLIP 不增加训练 step 或反向传播显存。与单视图 0806 分支相比，本方案训练时间仍由
五视角 DINO forward/backward 主导。CLIP 只增加 prior 构建和 unseen 推理时间。

## 运行

```powershell
# 训练并生成 Test_A（若没有 unseen 类别，不构建或加载 CLIP）
python run_competition_pipeline.py

# 训练/复用 checkpoint，并对 Test_B unseen 类拟合 CLIP prior 后推理
python run_competition_pipeline.py --test-b

# 已有完整 checkpoint 时，只补 prior 和生成 Test_B 提交
python run_competition_pipeline.py --test-b --skip-train
```

两个 prior 和提交目录都包含 config/artifact fingerprint。CLIP prompts、融合参数、
prior 或 checkpoint 变化后，旧的按类别推理结果不会被错误复用。
