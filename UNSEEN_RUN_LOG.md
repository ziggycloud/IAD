# Unseen 实验交接日志

更新时间：2026-08-03（Asia/Shanghai）

## 当前状态

代码、配置、类别协议、评分公式、延迟检查和一键入口已实现；尚未在目标 2 × 40 GB
机器上启动长训练，因此当前没有可报告的最终 seen/unseen 指标。

已完成：

- 固定种子 50 seen / 50 unseen / 60 unused，split 会持久化并严格校验。
- 新增类别泛化 Dinomaly 架构：正常原型库、类别无关路由、局部/部件/全局专家。
- 双卡全局 batch 与梯度累计语义适配；普通 Python 可用 DataParallel，torchrun 可用 DDP。
- seen 与 unseen 独立、可恢复评估；按用户确认的 30%/50%/20% 公式输出百分制总分。
- 单图 Mean/P95 延迟 ≤ 1 秒校验。
- IAD 环境完整单元测试 38/38 通过，`compileall` 与 `git diff --check` 通过。
- 在本机 RTX 3060 Ti 上用真实 DINOv2-B 权重完成旧基线 2-step 训练回归，以及
  新架构 1-step 完整训练/checkpoint 回归；正则项与五个辅助量均进入训练日志。
  另完成 4 张测试图的评估回归和单图延迟函数回归。当前机器只有一张 GPU，尚未做
  真实双卡通信冒烟。

数据抽查：种子 `20260803` 对应的 50 个 seen 类共有 6,240 张 train 视图，异常标签数为
0；6,000 steps × global batch 64 约为 61.5 个数据轮次。

`python run_unseen_pipeline.py --split-only` 已实际通过，固化 split 和 resolved seen 配置
已经生成。长训练没有启动。

## 下一步

在能看到两张 40 GB GPU 的机器上执行：

```powershell
python run_unseen_pipeline.py --split-only
python run_unseen_pipeline.py
```

第一条用于人工查看固化的类别列表，第二条开始训练并在完成后自动评估。若中断，直接重复
第二条，不要更改 seed、类别、分辨率、架构或训练参数。

## 恢复时读取顺序

1. 本文件。
2. `outputs/generalized_dinomaly_seen50_unseen50_vitl392/unseen_pipeline_state.json`。
3. 同目录的 `run_state.json`。
4. `logs/progress.jsonl` 和 `logs/unseen_pipeline.jsonl` 的末尾。
5. 若状态为 evaluating，读取相应 evaluation 目录内的 `eval_state.json`。

状态文件的 `next_action` 是恢复入口；训练从 `checkpoints/last.pt` 恢复，评估自动跳过已完成
类别。不要手工删除 split、checkpoint 或 per-category 结果。
