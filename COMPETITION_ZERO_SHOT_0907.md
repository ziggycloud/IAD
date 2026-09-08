# Real-IAD Variety unseen zero-shot branch

This branch keeps the trained Dinomaly path unchanged for categories present
in `Train`. Categories absent from `Train` are hard-routed to an independent
good-only CLIP path conditioned by the Test_B folder name.

The implementation is inspired by AdaptCLIP's visual/textual adapters and
alternating optimization, but it is a clean local implementation. It does not
load AdaptCLIP, AnomalyCLIP, VCP-CLIP or any other anomaly-detection checkpoint.
The only external pretrained parameters are the public OpenAI CLIP
`ViT-L-14-336` base weights loaded by OpenCLIP.

## Architecture

```text
category appeared in Train?
├─ yes -> existing five-view Dinomaly reconstruction path
└─ no  -> locally trained unseen path
         ├─ RGB -> luminance -> three repeated gray channels
         ├─ frozen public OpenAI CLIP ViT-L/14@336
         ├─ weighted fusion of the last four patch-feature layers
         ├─ trainable four-expert top-2 patch MoE
         ├─ visual branch
         │  ├─ residual local visual adapter
         │  └─ residual global visual adapter
         ├─ textual branch
         │  ├─ generic + folder-class normal/broken prompt ensemble
         │  ├─ learned prompt residual
         │  └─ residual textual adapter
         ├─ weighted visual/textual pixel logits -> anomaly mask
         └─ weighted global logits + local top-1% mean
            -> per-view anomaly score
```

The visual and textual adapters are optimized on alternating steps. Each step
is supervised directly on that branch's logits, so a weak detached branch
cannot suppress its gradient through a harmonic mean. Dice is evaluated only
for non-empty anomaly masks; focal loss uses explicit positive alpha. The
frozen CLIP backbone remains unchanged throughout. Training supervision comes
from defects synthesized from the competition's normal training views and
includes pixel focal loss, Dice loss, image-level BCE, clean-image suppression
and a prompt-anchor regularizer.

No real anomalous image or external mask is required. Pseudo anomalies are
made only from Train-good images using cross-image texture paste, local
luminance inversion, background-valued missing material, spatial displacement,
local blur and scratches/blobs. Random RGB colour blocks are not used. The
same image pipeline supplies exact synthetic masks.

After training, clean Train-good logits are used only to fit robust normal
pixel/image quantiles. This gives unseen categories an absolute normal
reference without reading Test_B labels or estimating a Test_B prototype.

At inference, no Dinomaly/CLIP heatmap blending is performed for unseen
categories. Five per-view image scores are aggregated as
`0.5 * max + 0.5 * mean`.

## Weights and mirror

No weight is downloaded while preparing the code. On the first training or
inference run, OpenCLIP downloads the public OpenAI CLIP base weight into
`third_party/OpenCLIP/weights`. If Hugging Face access in the runtime requires
a mirror, set it before launching:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

This is only a download endpoint override. No gated repository or task-specific
checkpoint is required. The revised training run writes `last.pt` for resume,
`final_model.pt` for the last step, and an EMA-loss-selected `best_model.pt`
under `outputs/.../zero_shot_gray_moe_goodonly/checkpoints/`. Inference uses the best model
by default. The previous 8000-step checkpoint is not resumed because its
loss/fusion semantics are incompatible.

## Run

Train the unchanged seen branch, train the unseen adapter branch, infer Test_B
and package the submission:

```bash
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  run_competition_pipeline.py \
  --set runtime.multi_gpu_strategy=ddp \
  --test-b
```

The zero-shot adapter stage is intentionally executed once on rank 0 after the
distributed Dinomaly stage. `--skip-train` expects both local checkpoints to
already exist and performs inference/package generation only.
