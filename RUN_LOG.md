# Dinomaly2 / Real-IAD Variety 续接日志

最后人工更新：2026-08-01 00:21（Asia/Shanghai）

## 当前状态

正式 pinned-upstream baseline 的原后台进程已经退出，尚未完成训练或评估。
最后状态停在 step 13,620/100,000，完整的 `last.pt` 仍在，可以自动续训。

- 实验：`dinomaly2_realiad_variety_b_280_strict_upstream`
- 配置：`configs/rtx3060ti_strict_upstream.yaml`
- 原训练 PID `43740` 已不存在；旧自动评估监视器状态已失效
- 下一步：在项目根目录运行 `python run_pipeline.py`；脚本会自动切换到
  正确的 IAD Python，并从 step 13,000 的 `last.pt` 继续
- 最新训练步、loss、ETA 的唯一权威来源：
  `outputs/dinomaly2_realiad_variety_b_280_strict_upstream/run_state.json`

现有 `last.pt` 为 720,702,551 bytes，保存于 step 13,000；
`run_state.json` 的 step 13,620 是旧进程退出前的内存进度，因此续训会从
13,000 开始，最多回退 620 步。

## 已完成

- 固定上游源码：`guojiajeremy/Dinomaly2@1745c613`
- 环境：`J:\project\IAD\data\.conda\iad\python.exe`
  (`torch 2.9.0+cu128`)
- 该环境是路径型环境，不是名为 `IAD` 的环境；不要运行
  `conda activate IAD`。`run_pipeline.py` 会从 base Python 自动切换到它。
- GPU：RTX 3060 Ti 8 GB
- 数据 metadata/sample 校验通过：
  160 类、19,955 train views、178,995 test views
- 唯一数据警告：`thyristor/.../C5` 的原图为 2164×2164，mask 为
  2162×2162；评估变换会缩放到同一尺寸
- DINOv2-register ViT-B/14 权重 SHA-256：
  `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71`
- 10 项单元测试通过
- 两步真实 GPU 训练、FP32 评估、checkpoint 恢复测试通过
- loss、anomaly map、Gaussian、top-1%、动态 min/max adeval 和 mask
  语义均已与 pinned 上游交叉核对
- 正式 FP32 batch 探测：micro-batch 16、无梯度累积、peak reserved
  4,728 MiB

## 正式运行语义

- ViT-B/14，280×280，全部 160 类统一训练
- FP32，effective/micro batch 都是 16
- 100,000 optimizer steps
- StableAdamW，所有可训练层 `2e-3`
- 第一步 LR=0，100-step warmup，随后保持 `2e-3`
- Loose Loss + LA + LC=2 + CR，dropout=0.4
- 每 1,000 步原子保存
  `outputs/.../checkpoints/last.pt`
- 评估为 FP32；按类别动态 min/max、1,000 bins；P-PRO 到 30% FPR
- mask 使用上游 bilinear + nonzero 语义
- 七项结果逐类别计算后对 160 类宏平均

## 中断后怎么继续

1. 读取本文件。
2. 读取正式实验的 `run_state.json` 和 `pipeline_state.json`。
3. 若 `run_state.status == "training"`，先确认 PID 43740 或同路径 Python
   仍在运行，不要重复启动。
4. 若训练进程已退出且状态为 `interrupted`/`failed`，运行：

   ```powershell
   python run_pipeline.py --config configs\rtx3060ti_strict_upstream.yaml
   ```

   它会从 `checkpoints/last.pt` 继续；强制终止最多损失最近 999 步，
   Ctrl+C 会主动保存当前断点。

5. 若 `run_state.status == "trained"` 但自动评估未运行，执行：

   ```powershell
   python run_pipeline.py `
     --config configs\rtx3060ti_strict_upstream.yaml `
     --skip-train
   ```

6. 若状态为 `evaluating` 或 `evaluation_failed`，重复同一评估命令；
   已完成类别会按 checkpoint/config 签名跳过。
7. 若状态为 `complete`，读取：

   - `reports/evaluation_report.md`
   - `outputs/.../evaluation/latest.json`
   - 对应目录的 `metrics.json` 与 `metrics_per_category.csv`

## 日志位置

- 训练文本日志：`outputs/.../logs/train.log`
- 训练逐步 JSONL：`outputs/.../logs/progress.jsonl`
- 后台进程 stdout/stderr：`outputs/.../background_stdout.log` /
  `background_stderr.log`
- 自动流水线 stdout/stderr：`outputs/.../pipeline_stdout.log` /
  `pipeline_stderr.log`
- 缓存（可选）状态：
  `data/realiadvariety_1024/_cache_state.json`

## 尚未完成

- 正式 100,000 步训练
- 160 类完整评估
- 用实测七项指标替换 `reports/evaluation_report.md` 的 pending 内容
- 基于最弱类别和 I/P/O 差距给出最终架构改进结论

当前报告不会填入 smoke 数值，也不会伪造或外推正式指标。
