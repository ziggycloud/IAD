# Seen/Unseen 一键实验说明

五视角联合网络、数据 shape、Set Attention、辅助损失、正常边缘先验和 object/view
batch 换算的完整说明见 [MULTIVIEW_ARCHITECTURE.md](MULTIVIEW_ARCHITECTURE.md)。

## 直接运行

在项目根目录执行：

```powershell
python run_unseen_pipeline.py
```

脚本会自动定位本机可用的 IAD Python，不依赖 `conda activate IAD`。默认配置为
`configs/unseen_2x40gb.yaml`，流程依次完成：固定类别划分、只用 seen-good 训练、
seen 测试、unseen 测试、加权评分、五视角对象延迟测试和报告生成。中断后重复同一命令即可
从 `last.pt` 和已完成的类别继续。

只生成并检查类别划分，不开始训练：

```powershell
python run_unseen_pipeline.py --split-only
```

已有完整 checkpoint 时只评估：

```powershell
python run_unseen_pipeline.py --skip-train
```

## 数据协议

- 对 160 个官方类别先排序，再使用种子 `20260803` 洗牌。
- 50 个 seen：训练只读取这些类别的 train-good；测试读取其 good 和 bad。
- 50 个 unseen：不参加训练，只在测试阶段读取 good 和 bad。
- 其余 60 类完全不用。
- 划分第一次生成后固化到
  `outputs/multiview_generalized_dinomaly_seen50_unseen50_vitl448_v3/protocol/category_split.json`。
  后续若种子或类别全集不一致，脚本会报错而不是静默换 split。
- 当前数据按该种子得到 6,240 张 seen 训练视图，检查结果全部为 good。

## 架构变化

新模型仍采用冻结的 DINOv2-register 编码器和 Dinomaly 重建异常分数，没有把 CLIP
分数与重建分数直接拼接。新增模块位于重建瓶颈内：

1. **可组合正常原型库**：只从 normal-good 特征学习 512 个正常原型。解码输入由原型
   重构得到，不保留把测试异常特征原样拷贝到输出的捷径。
2. **类别无关路由**：路由只读取稳健的视觉上下文与离散度，不读取训练或测试类别 ID。
3. **多尺度专家**：局部专家、部件尺度专家和全局布局专家共同重建正常结构；默认使用
   dense soft routing，避免某个专家在训练初期没有梯度。
4. **正常性正则**：原型使用均衡、路由均衡和专家差异正则仅由 seen-good 训练特征产生。

这套设计的目标是保留 Dinomaly 对细小缺陷的像素定位能力，同时把“记住某一类别外观”
改成“组合可复用的正常视觉原语”，从而提高新类别上的迁移能力。

## 双卡 40 GB 默认参数

| 参数 | 默认值 |
|---|---:|
| GPU | 2 张，`device_ids=[0,1]` |
| Backbone | DINOv2-register ViT-L/14 |
| 输入 | 5 × 448 × 448 |
| 精度 | BF16 |
| 单卡 micro-batch | 自动探测：3、2、1 个对象 |
| 全局有效 batch | 12 个对象（等效 60 张 view） |
| 梯度累计 | 自动；双卡 micro=3 时为 2 |
| Optimizer steps | 6,000 |
| 约合 seen 数据轮数 | 61.5 |
| 学习率 | `1e-3 -> 1e-4`，前 300 步 warmup |
| checkpoint 间隔 | 250 steps |

普通 `python` 启动时，`auto` 会在列出的两张 GPU 上使用单进程 DataParallel；通过
`torchrun --standalone --nproc_per_node=2 run_unseen_pipeline.py` 启动时使用 DDP。
Windows 下优先使用前一种一键方式。训练开始前会在一张卡上执行真实的
五视角 forward/backward/optimizer 显存探测，并预留 2.5 GB；最终选择会写入
`batch_tuning.json`。

如显存不足，将单卡 object micro-batch 调为 1；有效 object batch 仍为 12，双卡累计
步数会自动变成 6：

```powershell
python run_unseen_pipeline.py --set training.micro_batch_size=1
```

如果实测显存充足，可用 3，双卡累计 2 次：

```powershell
python run_unseen_pipeline.py --set training.micro_batch_size=3
```

注意：改变训练语义参数后不能续用旧 checkpoint，应同时指定新的
`experiment.output_dir`。

## 指标和总分

所有类别先各自计算指标，再做类别宏平均：

- `S_cls = mean(seen I-ROC, seen I-PR)`
- `S_seg = mean(seen P-ROC, seen P-PR, seen P-F1max)`
- `S_zs = mean(unseen I-ROC, unseen I-PR, unseen P-ROC, unseen P-PR, unseen P-F1max)`
- `总分 = 0.3 × S_cls + 0.5 × S_seg + 0.2 × S_zs`

分项和总分统一输出为 0–100。五视角版本用一个真实测试对象（五张图）进行 10 次预热
和 50 次计时，计时包含模型、异常图、正常先验、256 尺度后处理、Gaussian 和图像
分数，不包含磁盘解码。Mean 与 P95 都不超过 1 秒才标记延迟有效，同时报告 Max。
这个数值是 object latency，不能与旧单图 latency 直接比较。

## 输出和恢复

主要文件：

```text
outputs/multiview_generalized_dinomaly_seen50_unseen50_vitl448_v3/
├── unseen_pipeline_state.json
├── protocol/category_split.json
├── protocol/resolved_seen_config.yaml
├── logs/unseen_pipeline.jsonl
├── logs/progress.jsonl
├── checkpoints/last.pt
├── checkpoints/final_model.pt
├── evaluation/<seen签名>/...
├── evaluation/<unseen签名>/...
└── unseen_evaluation/
    ├── metrics_and_score.json
    └── evaluation_report.md
```

任务中断后先读 `UNSEEN_RUN_LOG.md`、`unseen_pipeline_state.json` 和
`run_state.json`。训练按 optimizer step 恢复；两个评估分别按已完成类别恢复。
