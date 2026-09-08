# Model evolution: Test_C AdaptCLIP 0907

## Fixed starting point

- Branch base: `competition-pipeline-adaptclip-0907`
- Base commit: `35f6437`
- Architecture: category-generalized Dinomaly for seen categories plus an independent AdaptCLIP-inspired learned zero-shot segmenter for unseen categories.
- Seen/unseen routing: the fixed 50 seen categories use Dinomaly and the normal prior; the fixed 50 unseen categories use only the zero-shot path. Unseen heatmaps are not mixed with Dinomaly heatmaps.
- Training protocol: competition `Train` only — the fixed 50 seen classes, 20 good objects per class. The pipeline rejects non-good, test-split, unknown-class, or wrong-count inputs.
- Evaluation protocol: `configs/testc_protocol.json` version 1, seed 20260909, 10 normal and 10 anomalous five-view test objects per class.
- Checkpoint rule: use the complete Dinomaly checkpoint plus the zero-shot best checkpoint selected by minimum training-loss EMA after warmup. Both fingerprints must match the resolved config; `--allow-partial` is diagnostic-only for Dinomaly.

## Run record

Every run captures commit/branch/dirty state, resolved config, protocol and Test_C hashes, checkpoint hash, environment, progress logs, per-category metrics, aggregate score, and a Markdown bottleneck report. Append material experiment decisions here as well as relying on generated logs.

| Date | Commit / run_id | Architecture or training change | Checkpoint selection | Test_C score | Finding / next action |
|---|---|---|---|---:|---|
| 2026-09-09 | AdaptCLIP 0907 protocol | Initial reproducible Test_C evaluation pipeline | complete Dinomaly + best zero-shot | — | Compare against reconstruction-only reference |

## Experiment template

```text
Date:
Commit / branch / run_id:
Hypothesis:
Architecture delta:
Training delta:
Important config overrides:
Checkpoint and SHA-256:
S_cls / S_seg / S_zs / total:
Weakest categories and classification-localization gap:
Latency and one-second status:
Conclusion:
Next experiment:
```
