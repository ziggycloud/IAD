# Real-IAD Variety 双路线技术调研与 Test_C 提点框架

## 结论摘要

`codex/testc-eval-adaptclip` 当前不是一个统一模型，而是按类别是否出现在 competition Train 中进行硬路由：50 个 seen 类使用多视图 category-generalized Dinomaly，50 个 unseen 类仅使用本地 AdaptCLIP-inspired 分支。Test_C 总分为 `0.3 × S_cls + 0.5 × S_seg + 0.2 × S_zs`，所以现阶段最有杠杆的方向是先保护并提升 seen 分割，再修复 unseen 分类/分割中与官方 AdaptCLIP 不一致的实现。每提升 1 个百分点，seen 的单个分割子指标约贡献 0.1667 个总分点，seen 的单个分类子指标约贡献 0.15 个总分点，而 unseen 的任一子指标约贡献 0.04 个总分点。

本次真实 Test_C 报告已经给出足够证据优先修复 unseen 路线：总分 70.507，seen macro 的 C-AUROC/P-AP 为 0.9440/0.4678，unseen 只有 0.6718/0.1552。报告还直接显示 unseen 的五视角均值 C-AUROC 为 0.6908，高于实际提交的 0.6718；因此 max/mean 混合并没有保护单视角异常，反而造成约 1.9 个 C-AUROC 点的损失。代码优化应先消除这个确定性损失，再评估更大的模型模块。

## 本次真实 Test_C 诊断与已落地改动

| 路线 | C-AUROC | C-AP | C-F1 | P-AUROC | P-AP | P-F1 |
|---|---:|---:|---:|---:|---:|---:|
| seen / Dinomaly | 0.9440 | 0.9541 | 0.9341 | 0.9254 | 0.4678 | 0.4992 |
| unseen / AdaptCLIP-inspired | 0.6718 | 0.7519 | 0.7463 | 0.8006 | 0.1552 | 0.2445 |

分类、分割和 zero-shot 三部分分别贡献 28.47、31.54、10.50 分。seen 已经是稳定底盘，当前最主要的方差与失分都来自 unseen。最弱 unseen 分类类别包括 `volume_potentiometer`（0.19）、`rectangular_connector_accessories`（0.34）、`small_leaf`（0.39）和 `rotary_position_sensor`（0.43）；最弱 P-AP 类别包括 `recorder_switch`（0.0088）、`LED_indicator`（0.0131）、`small_leaf`（0.0143）和 `ceramic_wave_filter`（0.0154）。这些类别同时覆盖全局结构错误与微小局部缺陷，说明不能只靠局部 patch top-k 承担分类，也不能继续浪费 8-bit mask 的动态范围。

本分支据此落地四项修改：

1. unseen 对象分数的五视角 `max_blend` 从 0.5 改为报告实测更好的纯均值，并让训练期对象级 BCE 使用同一规则。
2. OpenCLIP 单次前向同时返回 spatial intermediates 与真正的 global image feature，取代 patch 均值伪全局特征；提交分数以 0.35 权重融合全局和局部证据，训练期同步使用完全一致的融合。
3. unseen mask 改为逐类别共享的 robust affine 8-bit 标定。该变换在类别内单调，不使用标签，不改变连续排序，但能减少窄概率区间直接映射到 `[0,255]` 时的大量 ties。
4. unseen 推理明确读取完整 5000 步的 `final_model.pt`，不再使用 synthetic training-loss EMA 选出的早期 best。本次 best 快照停在 1560/5000 步，而该指标没有真实 unseen 泛化含义；seen 仍优先使用其 `best_model.pt`，不存在时回退 final。

这些修改会使旧 zero-shot checkpoint 失配（format version 提升），需要重新训练 adapter；seen Dinomaly checkpoint 不受影响。报告同时警告 seen 实际只训练了 3000 步，而默认配置是 6000 步，因此下一次正式比较应先完成既定训练预算，避免把代码改动与训练未完成混为一谈。

## 任务和数据特征

Real-IAD Variety v2 包含 160 类、198,950 张高分辨率图像，覆盖 28 个行业、24 种材料、22 种颜色和 27 类缺陷。五相机系统由 1 个俯视相机和 4 个斜视相机构成，且缺陷只在可见视角标注像素 mask。超过 90% 的图像分辨率高于 2,000 像素，论文同时指出小缺陷占比较高，因此 448 输入经 ViT/14 后只有 32×32 patch 网格，天然存在细边界和微小缺陷定位瓶颈。[^1]

论文的核心观察与本项目高度一致：类别规模扩大时，多类无监督异常检测会出现 capacity-diversity conflict；VLM zero/few-shot 路线对类别规模更稳定。论文把冲突解释为统一模型趋向过于通用的特征空间，损失类别细节，同时 reconstruction 模型又可能出现 identity mapping，连异常也被重建。[^2]

## 当前分支架构审计

### Seen 路线

Seen 类现已完整恢复到 commit `71eebcc`：DINOv2-register ViT-L/14、compositional reference bank、category-free router 和三尺度 experts。五张相机图按独立视图训练，effective batch 为 64；不再使用 Set Transformer、visibility adapter、multi-view auxiliary loss 或 train-only normal prior。训练参数同该提交：6000 步、LR 3e-4、500 步 warmup、Loose Loss 在 2000 步渐进到 0.7、gradient clip 0.1。

这条路线的优点是全程只使用正常 Train，且与已知提交的模型结构、训练方式和 checkpoint fingerprint 完全一致。主要风险是：

- 512 个共享 reference 和统一 decoder 同时覆盖 50 个 Train 类，仍可能发生 capacity-diversity conflict。
- 32×32 patch residual 双线性上采样到 448，之后 Gaussian sigma=2，容易把轮廓错误扩展成光晕。
- 当前没有像 Dinomaly+ / OneNIP 那样的高分辨率监督 refiner，P-PR/P-F1max 可能先于 P-AUROC 触顶。
- 五视角只在对象分数阶段拼接聚合，网络本身不建模跨相机上下文。

### Unseen 路线

Unseen 类硬路由到 frozen OpenAI CLIP ViT-L/14@336，最后四层 patch feature 经可学习层权重融合，再经过 local/global visual adapter 与 textual adapter。训练缺陷完全由 Train 正常图在线合成，visual/textual 分支交替更新。推理 mask 为两分支 logit 的固定权重融合。

这不是官方 AdaptCLIP 的等价实现：

- 当前实现已恢复真正的 CLIP global image token，并将其与局部 top-k 证据联合用于图像级预测。
- 官方 textual adapter 学习 prompt token，并经 frozen text encoder 得到文本表示；当前实现是在两个已编码 prompt anchor 上加 embedding residual/MLP。
- 官方 zero-shot 推理平均 visual/textual 预测；当前使用固定 `visual_fusion_weight=0.65`。
- 官方 AdaptCLIP 在辅助数据集的真实异常标注上训练，并用另一数据集评估；当前仅在目标 competition Train 的正常图上使用简化 synthetic defects。因此官方成绩只能作为结构参考，不能作为当前 checkpoint 的预期值。[^3]
- 当前代码训练了 `visual_image_margin` 和 `textual_image_margin` 的 image BCE，但最终 unseen object score 完全丢弃它们，只从局部 mask 的 top-1% 聚合分类分数。这会系统性伤害 missing-part、错装、整体形变等更依赖全局语义的异常。
- 当前 unseen mask 固定用 `[0,1]` 写入 8-bit PNG；若概率集中在窄区间，像素只占少数灰度级，P-AUROC/P-AP 会因 ties 下降。Test_C 的像素指标按类别计算且阈值取最优，使用同一类别共享的单调 robust affine 拉伸不会改变连续分数排序，反而能减少量化损失。

## 两条路线的可比 SOTA 基线

### 多类正常训练 / seen

Real-IAD Variety v2 的 160 类 MUAD 结果中，Dinomaly 为 I-ROC 85.4、P-ROC 91.5、P-PR 42.8、P-F1max 45.8；Dinomaly+ 为 87.1、91.9、49.7、49.2。Dinomaly+ 的关键改动是使用 pseudo-anomaly 数据额外微调 refined segmentation head，而不是推倒 reconstruction backbone。[^4]

OneNIP 的 supervised refiner 直接消费低分辨率 reconstruction/restoration error，使用两层转置卷积恢复空间细节，并用 synthetic anomaly mask 的 Dice 等目标训练。这个设计与当前 Dinomaly residual map 兼容，且可以冻结 encoder、reference bank、router 和 decoder，把风险限制在像素头。[^5]

因此，若报告表现为 seen C 指标尚可但 P-PR/P-F1max 明显偏低，优先级应是 refiner，而不是扩大 router、增加 decoder 层数或延长基础训练。

### Zero-shot / unseen

Real-IAD Variety v2 的完整 160 类 zero-shot 结果为：AnomalyCLIP I-ROC/P-PR 69.3/35.3，AdaCLIP 71.1/32.6，VCP-CLIP 72.6/36.9，AdaptCLIP-Zero 72.9/36.1。它们随类别数增加的波动远小于 MUAD。[^6] 官方 AdaptCLIP 后续报告 1/2/4-shot 达到 84.3/86.4/88.1 I-AUROC 和 48.9/50.8/52.5 P-AUPR，但 Test_C unseen 路线没有合法的目标类正常 prompt 时只能参考 0-shot，不能把 few-shot 数字作为目标。[^7]

2025–2026 的可迁移模块包括：

- FAPrompt：compound abnormality prompts 与 sample-wise abnormality prior，针对粗粒度“damaged/defective”提示无法覆盖细缺陷的问题。适合替代当前单一 broken anchor。[^8]
- FB-CLIP：多策略文本特征融合、identity/semantic/spatial 三路 foreground-background separation、background suppression 和 semantic consistency regularization。适合背景与物体边界假阳性明显的类别。[^9]
- FE-CLIP：DCT frequency-aware feature extraction 与 local frequency statistics adapter。适合 scratch、纹理、涂层、细裂纹等高频缺陷，但其论文训练使用辅助真实异常，不能直接照搬训练结论。[^10]
- RareCLIP / MuSc：利用测试流或整批无标签图像的 rarity/mutual scoring。它们是有价值的诊断或规则允许时的 transductive 方案，但在未确认竞赛规则前不应默认启用。[^11]

## 收到 Test_C 报告后的决策树

1. 先校验结果来源：run id、commit、checkpoint/config fingerprint、seen/unseen 各 50 类、每类 10 normal + 10 anomaly；拒绝把 smoke、partial 或旧 signature 混入比较。
2. 分开计算 seen/unseen 的 C-AUROC、C-AP、P-AUROC、P-AP、P-F1max，并同时看 category median、bottom-10、标准差和类别级相关性。
3. 若 unseen C 低而 P 尚可，先恢复真正的 CLIP global token，并让 object score 配置化融合 global/local，而不是重训像素头。
4. 若 unseen P 指标整体低且预测灰度范围窄，先做单调 per-category 8-bit 校准消融；这无需重训。
5. 若 unseen C/P 都低，比较 hard route、Dinomaly-only 和 scale-free rank ensemble。只有报告证明 Dinomaly 对 unseen 是负贡献时才保留硬路由。
6. 若 seen P-ROC 高而 P-AP/P-F1低，检查边缘 false positives、Gaussian sigma、浅/深层权重和 normal-prior gate，然后加入冻结 backbone 的 Dinomaly+ refiner。
7. 若 seen 与 unseen 都在细纹理/划痕类失败，再评估 frequency adapter；若都在背景/轮廓类失败，再评估 foreground-background disentanglement。

## 建议实验顺序

第一批应全部是低成本推理消融：unseen 动态范围校准；zero-shot local/global 分类分数；unseen Dinomaly/CLIP rank ensemble；seen object aggregation 的 max/visibility-aware；sigma 0/1/2。它们不应改变已有训练 checkpoint。

第二批只重训 zero-shot adapter：使用真实 CLIP global token；修正 image/local 两套 loss 与推理一致性；加入 fine-grained compound prompts；为 synthetic defect 生成增加 cut/pit/thin-crack/missing-part/edge-contact 等更接近 27 类缺陷的形态，并保留 hard-normal photometric augment。

第三批再训练 seen refiner：冻结 Dinomaly 主干，缓存正常/合成异常的多层 residual features，以轻量 1×1 + 两级上采样 head 训练 4 个 epoch 左右；同时保留 raw residual 与 refined map 的可配置融合，防止 refiner 对未覆盖的异常类型过拟合。

所有实验都要写独立 output directory 和完整 fingerprint。选择 checkpoint 时不应继续只用训练 loss EMA；至少建立固定、按 category/object 隔离的 synthetic validation，用独立随机种子和不同 anomaly generator 参数选择 zero-shot/refiner checkpoint。真实 Test_C 只能用于最终模型比较，不能反向调阈值或训练。

## Sources

[^1]: Zhu et al., “[Real-IAD Variety: Pushing Industrial Anomaly Detection Dataset to a Modern Era](https://arxiv.org/html/2511.00540v2),” v2, 2026, Sections 3–4.
[^2]: Zhu et al., “[Real-IAD Variety](https://arxiv.org/html/2511.00540v2),” 2026, MUAD scalability analysis.
[^3]: Gao et al., “[AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection](https://arxiv.org/html/2505.09926v2),” AAAI 2026, Sections 3.2 and 4.1.
[^4]: Zhu et al., “[Real-IAD Variety](https://arxiv.org/html/2511.00540v2),” 2026, Tables 3–4 and Dinomaly+ setting.
[^5]: Gao, “[Learning to Detect Multi-class Anomalies with Just One Normal Image Prompt](https://arxiv.org/abs/2505.09264),” ECCV 2024 / arXiv version 2025; [official OneNIP repository](https://github.com/gaobb/OneNIP).
[^6]: Zhu et al., “[Real-IAD Variety](https://arxiv.org/html/2511.00540v2),” 2026, zero-/few-shot table.
[^7]: Gao et al., “[official AdaptCLIP repository](https://github.com/gaobb/AdaptCLIP),” Real-IAD Variety results updated December 2025.
[^8]: Zhu et al., “[Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection](https://arxiv.org/abs/2410.10289),” ICCV 2025.
[^9]: Hu et al., “[FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement](https://arxiv.org/abs/2603.19608),” CVPR 2026.
[^10]: Gong et al., “[FE-CLIP: Frequency Enhanced CLIP Model for Zero-Shot Anomaly Detection and Segmentation](https://openaccess.thecvf.com/content/ICCV2025/papers/Gong_FE-CLIP_Frequency_Enhanced_CLIP_Model_for_Zero-Shot_Anomaly_Detection_and_ICCV_2025_paper.pdf),” ICCV 2025.
[^11]: He et al., “[RareCLIP: Rarity-aware Online Zero-shot Industrial Anomaly Detection](https://openaccess.thecvf.com/content/ICCV2025/papers/He_RareCLIP_Rarity-aware_Online_Zero-shot_Industrial_Anomaly_Detection_ICCV_2025_paper.pdf),” ICCV 2025; Li et al., “[MuSc](https://arxiv.org/abs/2401.16753),” ICLR 2024.
