# Real-IAD Variety unseen zero-shot branch

This branch keeps the trained Dinomaly path unchanged for categories present
in `Train`. Categories absent from `Train` are hard-routed to an independent
category-agnostic CLIP path.

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
         ├─ frozen public OpenAI CLIP ViT-L/14@336
         ├─ weighted fusion of the last four patch-feature layers
         ├─ visual branch
         │  ├─ residual local visual adapter
         │  └─ residual global visual adapter
         ├─ textual branch
         │  ├─ normal/broken prompt ensemble
         │  ├─ learned prompt residual
         │  └─ residual textual adapter
         ├─ harmonic visual/textual pixel probability -> anomaly mask
         └─ harmonic visual/textual global probability + local maximum
            -> per-view anomaly score
```

The visual and textual adapters are optimized on alternating steps. On a
visual step, the textual branch is detached; on a textual step, the visual
branch is detached. The frozen CLIP backbone remains unchanged throughout.
Training supervision comes from defects synthesized from the competition's
normal training views and includes pixel focal loss, Dice loss, image-level
BCE, clean-image suppression and a prompt-anchor regularizer.

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
checkpoint is required. The pipeline then trains its own lightweight adapter
checkpoint at `outputs/.../zero_shot/checkpoints/final_model.pt`.

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
