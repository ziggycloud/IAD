# Dinomaly2 + OpenCLIP competition fusion

This branch starts from `codex/completion_clear`. It keeps the original
Dinomaly2 encoder, bottleneck, decoder, losses, and 6,000-step training
schedule. Only the `competition` fitting/inference path adds a second, frozen
OpenCLIP branch.

## Architecture

The two branches solve different parts of the localization problem:

- **Dinomaly2 reconstruction:** strong patch-level appearance difference and
  fine spatial detail, but complex normal structure can also reconstruct badly.
- **OpenCLIP zero-shot semantics:** category-conditioned normal/abnormal
  prompts recognize states such as damaged, deformed, bent, twisted,
  misaligned, or missing parts. It is less sensitive to whether a complex
  normal wire can be reconstructed exactly.

OpenCLIP uses frozen `ViT-L-14-336` OpenAI weights. Dense tokens from layers
6/12/18/24 are projected into the text space and aggregated over 1x1, 3x3, and
5x5 neighbourhoods. Each category name is expanded into normal and anomalous
prompt ensembles, mixed with an object-agnostic prototype to reduce dependence
on unusual class names.

## Fusion instead of a fixed average

Before Test_A inference, the pipeline samples eight **normal Train images per
category and camera view**. For both Dinomaly2 and CLIP it stores the median,
95th percentile, and 99.5th percentile. Test_A is never used to tune these
statistics.

At inference the two maps are normalized against their own Train-normal tails:

1. CLIP-normal evidence softly suppresses broad Dinomaly reconstruction heat.
2. At least 35% of the reconstruction score is retained, protecting tiny
   defects that CLIP may miss.
3. CLIP-only evidence can recover shape or logical anomalies.
4. An agreement term boosts pixels where both independent branches respond.

This is deliberately not a raw `0.5 * map_a + 0.5 * map_b`: the two score
ranges are different and a fixed average would be dominated by whichever map
has the wider numerical distribution.

## Setup and weights

Install the added pinned dependency:

```powershell
./setup_env.ps1
```

With network access, `pretrained: openai` downloads the official CLIP weight on
first use. For an offline server, place `ViT-L-14-336px.pt` locally and set:

```yaml
evaluation:
  clip_fusion:
    weights_path: /absolute/path/to/ViT-L-14-336px.pt
```

The local weight is SHA-256 identified in the fusion artifact. An artifact is
also bound to the exact Dinomaly checkpoint and fusion configuration, so stale
calibration cannot be silently reused.

## Recommended competition run

For a clean comparison with the 77.3520 baseline, reuse its completed
checkpoint. This skips Dinomaly training and performs only Train-normal fusion
calibration plus Test_A inference:

```powershell
python run_competition_pipeline.py `
  --skip-train `
  --checkpoint outputs/dinomaly2_competition_vitl448/checkpoints/final_model.pt
```

If the clear checkpoint is unavailable, run the complete competition pipe:

```powershell
python run_competition_pipeline.py
```

Outputs are isolated under:

```text
outputs/dinomaly2_clip_fusion_competition_vitl448_v1/
|-- clip_fusion/clip_fusion.pt
|-- clip_fusion/clip_fusion.json
`-- competition_submission/<signature>/submission.zip
```

Other training and evaluation pipes are unchanged.

## Design references

- WinCLIP, CVPR 2023: category/state prompt ensembles, local multi-scale CLIP
  features, and complementary normal-reference evidence.
  https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html
- AnomalyCLIP, ICLR 2024: object-agnostic normal/anomalous prompts and
  multi-level dense CLIP features.
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/d7b50b8ac2c781a12f26155f48310d8d-Abstract-Conference.html
- APRIL-GAN, VAND 2023: projected CLIP patch features and visual memory as
  complementary anomaly segmentation signals.
  https://arxiv.org/abs/2305.17382
