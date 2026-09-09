# Test_C baseline audit — 2026-09-09

## Evidence and comparability

- Evaluation signature: `e89b55e7e2f9843ad7e2a45d5ce464be08e8f65c47db7ffbd6ac39ef58844ba3`.
- Dataset manifest SHA-256: `a8d33b82b283dcd4e776b84598e5caae4fa80f350d685516a1cb20de4c6a7d62`.
- The report contains 50 seen and 50 unseen categories, each with 10 normal and 10 anomalous objects.
- The evaluated checkpoint contains only 1,000 optimizer steps. The baseline protocol is 6,000 steps and Loose Loss warmup alone is 1,500 steps. This is therefore a short-run diagnostic, not the final baseline.
- Training progress, resolved config, checkpoint hash, and run manifest were not included with the three supplied metric files. Gradient stability, learning curves, resume provenance, and the exact CLI overrides cannot be verified from these files alone.

## Score decomposition

| Component | Value | Weight | Contribution to total |
|---|---:|---:|---:|
| Seen classification (`S_cls`) | 91.0842 | 0.3 | 27.3252 |
| Seen segmentation (`S_seg`) | 56.3020 | 0.5 | 28.1510 |
| Unseen combined (`S_zs`) | 53.0336 | 0.2 | 10.6067 |
| Total |  |  | **66.0830** |

Seen classification is already strong. The largest recoverable deficit is localization: seen pixel AUROC is 0.9003, but pixel AP and F1max are only 0.3689 and 0.4199. The large AUROC-to-AP/F1 gap indicates that broad normal/background responses are ranked mostly below defect pixels, yet still create too many false-positive pixels at useful thresholds.

The same effect is stronger for unseen categories: pixel AUROC remains 0.8668 while pixel AP/F1max fall to 0.1654/0.2240. Twenty unseen categories have pixel AP below 0.10. This is consistent with image-wide semantic novelty in reconstruction maps rather than complete loss of local defect signal.

## Object-level generalization

- Seen object AUROC/AP: 0.9026/0.9191.
- Unseen object AUROC/AP: 0.6524/0.7431.
- Ten unseen categories have object AUROC below 0.50. The worst are `flower_velvet_fabric` and `ceramic_fuse` (both 0.11), followed by `audio_jack_socket` (0.20).
- Mean-view diagnostic AUROC is 0.9142 seen and 0.6700 unseen, versus 0.9026 and 0.6524 for the submitted visibility/max blend. Max-view aggregation is worse at 0.8792 and 0.6242. The learned visibility weights are trained only on normal objects and are not reliable defect-visibility probabilities.

The baseline now uses unweighted five-view mean aggregation. This is label-free at inference time and matches the stronger diagnostic on both partitions. It avoids letting a single noisy camera or a normal-trained visibility head dominate the object score.

## Training audit

The data gate is sound: the Test_C entry point audits the fixed Train set before training, limits training to the 50 seen categories, verifies 20 good objects per category against the official JSON train split, and uses Test_C only after checkpoint/prior creation.

The main training-process issue was Loose Loss selecting hard patches over each local micro-batch. Under DDP plus gradient accumulation, this changes the selected patch population with GPU count and memory fallback. The new `training.loose_loss_scope: object` computes selection independently for each five-view object, so the loss semantics no longer depend on micro-batch size or world size. Because this setting changes the training fingerprint, the improved baseline must start a new 6,000-step run; it must not resume the 1,000-step checkpoint.

The architecture remains the same category-generalized Dinomaly reconstruction model with frozen DINOv2-register ViT-L/14 features, a learned normal reference bank/router, and five-view context. This keeps the experiment focused: the next score change measures training semantics and inference calibration, not a simultaneous backbone replacement.

## Inference and localization changes

For unseen categories only, the predictor now removes part of each frame's median reconstruction floor while retaining local excess and 25% of the global component. This happens after the Train-only normal prior and before upsampling/smoothing. It does not inspect Test_C labels, masks, or category-specific thresholds. The purpose is to reduce category novelty that raises every pixel while preserving localized defect contrast.

The prediction signature previously included the training fingerprint and submission settings but omitted evaluation-time map settings. Changing Gaussian smoothing, layer weights, normal-prior calibration, or novelty debias could therefore resume stale per-category outputs. The signature now includes the complete evaluation configuration, forcing regeneration whenever prediction semantics change.

The next report records normal/anomalous score means, their margin, foreground-pixel prevalence, component contributions, category dispersion, aggregation diagnostics, and a warning when the checkpoint is below 6,000 steps. These fields make high-AUROC/low-AP failures distinguishable from object-score inversion without opening per-category prediction files manually.

## Submission artifact boundary

Before this change, `run_testc_pipeline.py` reused the competition packager against `Test_C/images`, so it produced a structurally valid ZIP under `competition_submission/`. That ZIP has 2,000 Test_C object rows and 10,000 Test_C masks. It is not a Test_B result and must not be submitted.

Test_C predictions now go to `testc_predictions/` and no ZIP is built by default. The optional exported archive is named `testc_predictions_not_for_submission.zip` and carries `competition_submit_ready: false`. The unchanged official Test_B entry point writes `competition_submission/<signature>/submission.zip` only after scanning `data/competition/Test_B`.

## Next controlled run

Use a new output directory and `--resume never`; the old 1,000-step checkpoint has different training semantics and total-step fingerprint. After the full run, compare against signature `e89b55e7e2f9`. If pixel AP/F1 remain low while AUROC stays high, the next isolated ablation should compare smaller Gaussian sigma and deeper-layer weighting, one change at a time. Do not select per-category thresholds from Test_C for Test_B inference.

## Primary references

- Dinomaly, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html
- Official Dinomaly2 repository and multi-view Real-IAD Variety entry point: https://github.com/guojiajeremy/Dinomaly2
- Real-IAD multi-view dataset, CVPR 2024: https://realiad4ad.github.io/Real-IAD/
- Real-IAD Variety benchmark: https://arxiv.org/abs/2511.00540
