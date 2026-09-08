# Model evolution: Test_C baseline

## Fixed starting point

- Branch base: `origin/codex/competition-pipeline`
- Base commit: `7b2318b`
- Architecture: category-generalized Dinomaly reconstruction with five-view context.
- Seen/unseen routing: all 100 Test_C categories use the Dinomaly reconstruction predictor. Seen categories use category-aware normal prior statistics; unseen categories fall back to the existing view-global prior.
- Training protocol: competition `Train` only — the fixed 50 seen classes, 20 good objects per class. The pipeline rejects non-good, test-split, unknown-class, or wrong-count inputs.
- Evaluation protocol: `configs/testc_protocol.json` version 1, seed 20260909, 10 normal and 10 anomalous five-view test objects per class.
- Checkpoint rule: use a complete `final_model.pt` whose semantic configuration fingerprint matches the run, unless `--allow-partial` is explicitly supplied for diagnostics.

## Run record

Every run captures commit/branch/dirty state, resolved config, protocol and Test_C hashes, checkpoint hash, environment, progress logs, per-category metrics, aggregate score, and a Markdown bottleneck report. Append material experiment decisions here as well as relying on generated logs.

| Date | Commit / run_id | Architecture or training change | Checkpoint selection | Test_C score | Finding / next action |
|---|---|---|---|---:|---|
| 2026-09-09 | baseline protocol | Initial reproducible Test_C evaluation pipeline | complete final checkpoint | — | Establish reference run |

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
