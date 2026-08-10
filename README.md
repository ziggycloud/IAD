# Dinomaly2 × Real-IAD Variety（RTX 3060 Ti）

这是从
[guojiajeremy/Dinomaly2](https://github.com/guojiajeremy/Dinomaly2)
固定版本 `1745c613` 中抽取的 Real-IAD Variety 基线。它保留
DINOv2-register encoder、Dinomaly2 bottleneck/decoder、Loose Loss 和
StableAdamW，重写了数据读取、8 GB 显存探测、断点续训和可恢复评估。

## 最短运行方式

在项目根目录打开 PowerShell，直接运行一条 Python 命令即可。若当前
Python 缺少依赖，脚本会自动定位本机的 IAD 环境并用正确解释器重新启动；
训练中断后重复运行同一命令，会自动从 `last.pt` 续跑：

```powershell
python run_pipeline.py
```

也可以显式激活路径型环境，或直接指定解释器。当前环境是按路径创建的，
因此不能使用 `conda activate IAD`：

```powershell
conda activate J:\project\IAD\data\.conda\iad
python run_pipeline.py

# 不激活环境也可以
J:\project\IAD\data\.conda\iad\python.exe run_pipeline.py
```

在其他机器上可以把 `IAD_PYTHON` 环境变量设为目标环境的 Python 路径，
一键脚本会优先使用它。

首次运行前建议先安装依赖并做一次数据抽检（约 40 秒）：

```powershell
.\setup_env.ps1
.\validate_data.ps1 -Mode sample
```

`run_pipeline.py` 会自动探测 micro-batch（16/8/4/2/1）、训练、保存断点，
再按论文口径做可恢复的 160 类评估。旧的 `train.ps1` 和 `evaluate.ps1`
继续保留作兼容入口。

## 3060 Ti 配置

一键训练/评估默认使用
[configs/rtx3060ti_strict_upstream.yaml](configs/rtx3060ti_strict_upstream.yaml)，
即本次正式 baseline。另提供显存更宽松、采用论文 LR 修正的
[configs/rtx3060ti.yaml](configs/rtx3060ti.yaml)：

- DINOv2-register ViT-B/14，输入 280 × 280；
- 100,000 optimizer steps，有效 batch 16；
- 修正版使用 BF16 AMP；两套配置都会实机探测 micro-batch，必要时自动
  梯度累积；
- 每 1,000 步原子保存 `last.pt`，重新运行同一命令即可续跑；
- 预留 700 MiB 显存给桌面和波动；
- 修正版采用 warmup + cosine LR（`2e-3 → 2e-4`），首层
  bottleneck 使用 `2e-4`；严格 preset 的 LR 见下文。

本机真实 forward/backward/optimizer 探测结果：

```text
默认 BF16：micro-batch=16, accumulation=1, peak reserved=3092 MiB
严格上游 FP32：micro-batch=16, accumulation=1, peak reserved=4728 MiB
```

仅重新做显存探测而不训练：

```powershell
python train.py --probe-only
```

若仍遇到 OOM，可直接覆盖配置，无需改代码：

```powershell
python run_pipeline.py `
  --set training.micro_batch_size=8 `
  --set evaluation.batch_size=1 `
  --set runtime.num_workers=1
```

上游当前 Real-IAD Variety multiview 脚本是 preview：所有可训练层均使用
`2e-3`，FP32，第一步 LR 为 0，100 步 warmup 后保持 `2e-3`。
显式命令等价于默认一键命令：

```powershell
python run_pipeline.py --config configs\rtx3060ti_strict_upstream.yaml
```

两个 preset 必须分开报告，不能混用 checkpoint。

## 可选的训练集缓存

默认严格配置直接读取原图，拿到项目后可以立刻一键训练。由于 100,000 步会
重复解码约 160 万张高分辨率图片，建议先建立仅含官方 19,955 张训练图的
1024 × 1024 缓存：

```powershell
# 可中断；重新运行会验证并跳过已完成图片
python prepare_cache.py

# 使用缓存训练，评估仍读取官方原图和 mask
python run_pipeline.py --config configs\rtx3060ti_strict_upstream_cached.yaml
```

缓存状态位于
`data\realiadvariety_1024\_cache_state.json`，逐步记录位于同目录的
`_cache_progress.jsonl`。只验证两张图的缓存流程：

```powershell
python prepare_cache.py --max-images 2
```

有限缓存不能用于完整训练；随后需不带 `-MaxImages` 重新运行补齐。
`rtx3060ti_cached.yaml` 则是 BF16 + 论文 LR 修正版的缓存 preset。

## 数据与评估口径

数据由官方 160 份 split JSON 枚举，不扫描文件夹猜标签：

```text
data\realiadvariety_jsons\Real-IAD_Variety_jsons
data\realiadvariety_raw
```

已核对的官方规模：

| split | 图像/视图 | 五视角对象 |
|---|---:|---:|
| train（全正常） | 19,955 | 3,991 |
| test | 178,995 | 35,799 |

评估对每个类别单独计算，再对 160 类做宏平均：

```text
I-ROC, I-PR, I-F1max, P-ROC, P-PR, P-F1max, P-PRO
```

- 图像分数：单视角异常图 top 1% 像素均值；
- P-PRO：积分到 30% FPR；
- 异常图先缩放到 256 × 256，再做 `5 × 5, σ=4` Gaussian；
- mask 使用 pinned 上游的 bilinear + nonzero 语义；若要做更规范的
  nearest-neighbor 消融，可覆盖
  `dataset.mask_resize_semantics=nearest_binary`；
- manifest 中异常对象但 `mask_path=null` 的视角，按 Dinomaly2 上游
  语义作为 view-normal；像素 mask 为全零；
- 另输出五视角对象级 O-ROC/O-PR/O-F1 作为架构诊断，不混入论文七项指标。

论文表 3 给出的 Dinomaly 数值不是 Dinomaly2，报告中仅作为量级参考，
不会冒充本次实测。

## 输出与中断恢复

默认输出目录：

```text
outputs\dinomaly2_realiad_variety_b_280\
├── resolved_config.yaml
├── batch_tuning.json
├── run_state.json
├── logs\
│   ├── train.log
│   └── progress.jsonl
├── checkpoints\
│   ├── last.pt
│   └── final_model.pt
└── evaluation\<签名>\
    ├── eval_state.json
    ├── metrics.json
    ├── metrics_per_category.csv
    ├── evaluation_report.md
    └── per_category\*.json
```

任务中断后，让 Codex 读取 [RUN_LOG.md](RUN_LOG.md) 和对应实验的
`run_state.json` 即可知道下一步。训练会从 `last.pt` 继续；评估会根据
checkpoint 与配置生成签名，并跳过同签名下已完成的类别。

默认评估只接受完成 100,000 步的 `final_model.pt`，避免把中间模型静默写成
正式报告。仅做诊断时可以显式运行：

```powershell
python run_pipeline.py `
  --skip-train `
  --checkpoint outputs\<实验>\checkpoints\last.pt `
  --allow-partial
```

中间断点评估会标记为 diagnostic，且不会覆盖正式总报告。

## 常用检查

```powershell
# 只检查全部 manifest 元数据与固定计数
.\validate_data.ps1 -Mode metadata

# 全量解码所有图像和 mask（耗时很长）
.\validate_data.ps1 -Mode full

# 两步训练 + 10 张图评估
python run_pipeline.py --config configs\smoke.yaml --resume never
```

预训练 backbone 首次使用时从 Meta 官方地址下载到
`third_party\Dinomaly2\backbones\weights`。本机已下载并完成真实 GPU
冒烟训练、评估和断点恢复测试。

## Seen/Unseen 类别泛化实验

新增的类别泛化架构、双卡 40 GB 配置和 50/50/60 评估协议使用独立入口：

```powershell
python run_unseen_pipeline.py
```

它只用固定的 50 个 seen 类 train-good 训练，随后分别评估 seen 和完全未参与训练的
50 个 unseen 类，并输出 `S_cls`、`S_seg`、`S_zs`、百分制总分及五视角对象延迟判定。
完整参数、公式和恢复方式见 [UNSEEN_PROTOCOL.md](UNSEEN_PROTOCOL.md)，任务交接状态见
[UNSEEN_RUN_LOG.md](UNSEEN_RUN_LOG.md)。

真正的五视角 object batch、Set Transformer、visibility-aware adapter、Train-only
正常边缘先验及消融方式见
[MULTIVIEW_ARCHITECTURE.md](MULTIVIEW_ARCHITECTURE.md)。

## 比赛 Train / Test_A 提交流水线

比赛数据直接按 `category/Sxxxx/0.png..4.png` 读取，不需要官方 JSON，也不读取
任何标签。默认配置使用 DINOv2-register ViT-L/14 和上游 DInomaly2 主体，只在
`1024 → 256 → 1024` bottleneck 中加入上下文困难度估计、`64/128/256` 连续嵌套
通道和低/中/高难度三个独立重建专家，并把输入对齐到提交 mask 的 448 × 448：

```powershell
# 一键：审计 → 正常 Train 训练 → 正常 Train prior → Test_A 推理 → 校验并打 ZIP
python run_competition_pipeline.py

# 只检查目录、类别和五视角完整性
python run_competition_pipeline.py --validate-only

# 已有 final_model.pt 时跳过训练，重新/继续按类别推理和打包
python run_competition_pipeline.py --skip-train

# 只训练，暂不推理
python run_competition_pipeline.py --skip-inference
```

默认数据路径和配置分别为 `data/competition/Train`、
`data/competition/Test_A` 与 `configs/competition.yaml`。每个 `Sxxxx` 的五张图仍
完整参与训练和提交，但网络按独立的 `[3,448,448]` view 运行，不重新启用旧的
Set Transformer 联合架构。分类分数聚合由
`submission.object_score_aggregation` 控制；`legacy_concat_topk` 可恢复旧基线，默认
`visibility_aware` 在无跨视角权重时使用均匀权重并保留 max 分量，避免单相机可见
缺陷被软共识抹掉。

困难度估计器只观察中心 patch 之外的 3×3/5×5 邻域和全局 special tokens，并显式
编码邻域方差与两个尺度的差异。前 1500 step 保持原始 256 通道和高容量专家，同时
将高专家的正常重建能力蒸馏给低/中专家；随后用 1000 step 平滑启用软路由，从
2500 step 开始执行稀疏 top-1 分派。重建与蒸馏梯度都不会反向操纵困难度路由。
日志中的 `difficulty_prediction`、`local_complexity_mean`、
`moe_expert_distillation` 和 `moe_*_usage` 用于监控复杂度判断与专家退化。

正常边缘先验仅扫描 Train 正常图，并把 checkpoint/config fingerprint 写入 artifact；
不匹配时拒绝复用。Test_A 只在 prior 保存后进入推理，不参与统计或阈值选择。

完成后读取：

```text
outputs/multiscale_density_moe_competition_vitl448_v1/
├── competition_data_audit.json
├── competition_pipeline_state.json
├── checkpoints/final_model.pt
├── normal_prior/normal_prior.pt
└── competition_submission/
    ├── latest.json
    └── <签名>/
        ├── result.json
        ├── package/
        │   ├── submission.csv
        │   └── predicted_masks/<类别>/Sxxxx/0_mask.png ... 4_mask.png
        └── submission.zip
```

`submission.zip` 已检查 CSV 行数/顺序、分数范围、全部 448 × 448 单通道 mask、
ZIP 根目录结构和 CRC，可直接手动上传。推理按类别落盘；中断后重复同一命令会跳过
已完整生成的类别。训练 batch 以单张 view 为单位，默认 effective batch 是 12；自动
显存探测执行真实单视角 difficulty-MoE forward/backward。单张 GPU 可用：

```powershell
python run_competition_pipeline.py `
  --set runtime.device_ids=[0] `
  --set submission.batch_size=2
```

分割边缘假阳性、五视角无监督建模、梯度爆炸诊断和 YAML 调参说明见
[COMPETITION_TUNING.md](COMPETITION_TUNING.md)。
