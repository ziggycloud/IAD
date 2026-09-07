# Competition zero-shot branch (0907)

This branch replaces inference-only CLIP suppression with a separately trained,
category-agnostic anomaly segmenter.

## Routing

```text
                       category appeared in Train?
Test image ─────────────────────┬────────────────────────
                               │
                         yes   │   no
                               │
          five-view Dinomaly   │   frozen CLIP ViT-B/16
          + normal prior       │   intermediate patch features
                               │          │
                               │   trainable residual adapter
                               │          ├─ learned normal/broken prompts
                               │          └─ dense segmentation decoder
                               │                    │
                  reconstruction map       absolute P(broken) map
                               └──────────┬─────────┘
                                  submission output
```

There is no blend between these paths. Seen classes preserve the 0906 Dinomaly
baseline. An unseen class never consumes the reconstruction anomaly map or its
normal prior.

## Training

Stage 1 trains the existing multiview Dinomaly model. Stage 2 freezes all CLIP
weights and trains only the layer weights, visual residual adapter, two
object-agnostic prompt offsets, and dense decoder. Training normals are changed
into masked synthetic defects (scratch, missing material, foreign texture,
colour/contamination and displaced material); unchanged hard augmentations are
also included as negative examples.

The zero-shot loss is:

```text
L = focal(mask) + dice(mask)
  + 0.25 focal(CLIP semantic map)
  + 0.25 BCE(image label)
  + 0.20 clean false-positive penalty
  + 0.01 prompt anchor penalty
```

The default schedule is AdamW, 8,000 steps, 400-step linear warmup followed by
cosine decay from `2e-4` to `1e-5`, BF16 and gradient norm clipping at 1.0.
Parameters with one dimension and prompt/layer logits receive no weight decay.

## Commands

Full two-stage training and Test_B packaging:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  run_competition_pipeline.py \
  --set runtime.multi_gpu_strategy=ddp \
  --test-b
```

Resume training with the same command. After both final checkpoints exist,
inference-only packaging is:

```bash
CUDA_VISIBLE_DEVICES=0 python run_competition_pipeline.py \
  --test-b --skip-train
```

The CLIP weights are cached under `third_party/OpenCLIP/weights`. The learned
zero-shot checkpoint is written beneath the experiment output directory at
`zero_shot/checkpoints/final_model.pt`.
