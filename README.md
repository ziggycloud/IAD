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
50 个 unseen 类，并输出 `S_cls`、`S_seg`、`S_zs`、百分制总分及单图延迟判定。
完整参数、公式和恢复方式见 [UNSEEN_PROTOCOL.md](UNSEEN_PROTOCOL.md)，任务交接状态见
[UNSEEN_RUN_LOG.md](UNSEEN_RUN_LOG.md)。

## 比赛 Train / Test_A / Test_B 提交流水线

比赛数据直接按 `category/Sxxxx/0.png..4.png` 读取，不需要官方 JSON，也不读取
任何标签。默认配置复用上面的 category-generalized Dinomaly、DINOv2-register
ViT-L/14，并把输入对齐到提交 mask 的 448 × 448：

```powershell
# 一键：审计数据 → 训练（可从 last.pt 恢复）→ Test_A 推理 → 校验并打 ZIP
python run_competition_pipeline.py

# 只检查目录、类别和五视角完整性
python run_competition_pipeline.py --validate-only

# 已有 final_model.pt 时跳过训练，重新/继续按类别推理和打包
python run_competition_pipeline.py --skip-train

# 使用同一训练配置推理 Test_B；允许类别和每类样本数不同于 Train
python run_competition_pipeline.py --test-b --skip-train

# 只训练，暂不推理
python run_competition_pipeline.py --skip-inference
```

默认数据路径和配置分别为 `data/competition/Train`、
`data/competition/Test_A` 与 `configs/competition.yaml`。传入 `--test-b` 后测试
路径切换到 `data/competition/Test_B`，并自动取消 Test_A 专用的类别及样本数约束。
分类分数把同一零件的
5 个视角合并后取异常热图 top 1% 均值，再用严格单调函数映射到 `[0,1]`；
mask 按类别做无标签、严格单调的 8-bit 分位数标定，以充分利用 PNG 灰度范围。

`0806-competition-clip` 分支会比较 Train 与测试目录中的类别名称。已见类别继续走
原始 Dinomaly 路径；仅未见类别会懒加载冻结的 OpenCLIP ViT-B/16，通过通用的
normal/broken 文本提示生成 patch 级破损语义图，并与去除全局类别偏移后的重建图
保守融合。CLIP 不参与训练，也不写入 Dinomaly checkpoint，因此已有该分支基线权重
可以继续使用。首次推理未见类别时会把 CLIP 权重下载到
`third_party/OpenCLIP/weights`。

完成后读取：

```text
outputs/generalized_dinomaly_competition_seen50_vitl448/
├── competition_data_audit.json
├── competition_pipeline_state.json
├── checkpoints/final_model.pt
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
已完整生成的类别。单张大显存 GPU 运行时可覆盖设备列表和推理 batch；训练仍用
梯度累计保持全局 effective batch 64：

```powershell
python run_competition_pipeline.py `
  --set runtime.device_ids=[0] `
  --set submission.batch_size=2
```
