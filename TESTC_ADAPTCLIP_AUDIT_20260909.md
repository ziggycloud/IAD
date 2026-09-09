# AdaptCLIP Test_C audit — 2026-09-09

## Evidence and comparability

The supplied `metrics_and_score (1).json` and 100-row per-category files are
treated strictly as evaluation data. The run scored **70.9619** with
`S_cls=94.1462`, `S_seg=62.7033`, and `S_zs=56.8320`. Its weighted
contributions are 28.24 classification, 31.35 segmentation, and 11.37
zero-shot points.

The available reconstruction-only comparison scored 66.0830, but it used a
shorter 1000-step checkpoint. AdaptCLIP's +4.8789 total delta is therefore not
a controlled architecture ablation. A valid comparison requires the same
Train data, optimizer schedule, Dinomaly steps, seed policy, and Test_C
manifest, changing only the unseen route.

## Metric diagnosis

- Seen classification is already strong: AUROC 0.9358 and AP 0.9471.
- Seen localization is the main weighted bottleneck: pixel AUROC 0.9238 is
  high, but pixel AP 0.4623 and F1max 0.4950 show poor precision/threshold
  concentration around the true defect area.
- AdaptCLIP materially improves unseen object ranking (AUROC 0.7428, AP
  0.8055), while unseen pixel AP is only 0.1948 and F1max 0.2597. The
  classification-to-pixel-AP gap is 54.80 points.
- Against the short reconstruction baseline, unseen classification AUROC
  improves by 0.0904, but unseen pixel AUROC drops by 0.0280. The learned path
  is semantically useful but its local maps are not spatially reliable enough.
- Worst pixel-AP categories include `LED_indicator` (0.0133),
  `recorder_switch` (0.0227), `power_jack` (0.0246), `gear_motor` (0.0277),
  `lilypad_led` (0.0338), and `ethernet_connector` (0.0392).
- Unseen category variance is high (classification AUROC standard deviation
  0.1583); three unseen categories are below 0.5 AUROC. This is not a single
  global threshold problem because AUROC itself is rank-based.

The per-category changes are heterogeneous. `flower_velvet_fabric`,
`audio_jack_socket`, and `button_switch` gain strongly in classification,
whereas `lego_propeller`, `angled_toggle_switch`, `miniature_motor`, and
`rotary_position_sensor` regress. This pattern points to synthetic-defect and
local-map generalization, not simply insufficient score calibration.

## Training and inference audit

1. Dinomaly Loose Loss selected hard locations across the whole micro-batch.
   With DDP and gradient accumulation, changing per-GPU batch size changed the
   selection threshold. Object-scoped selection now preserves semantics across
   hardware fallbacks.
2. The zero-shot image BCE supervised a separate global CLIP margin, while the
   actual CSV score deliberately comes only from the local masks. The global
   visual adapter therefore received classification supervision that could not
   directly improve the submitted score. The primary image loss now computes
   local top-1% scores for five views, aggregates them with the same max/mean
   rule as inference before inference-only resizing/smoothing, and applies
   object BCE. Global BCE remains a lower-weight auxiliary signal.
3. Synthetic anomalies were sampled independently per view with probability
   0.60. Under an any-view object label this makes about 99% of five-view
   objects anomalous (`1 - 0.4^5`), so the new object loss would otherwise be
   severely imbalanced. Synthesis now samples balanced objects first, permits
   view-level invisibility, and guarantees at least one visible defect for an
   anomalous object.
4. Seen submitted AUROC was 0.9358; the diagnostic five-view mean was 0.9418
   and max was 0.9214. Seen Dinomaly now defaults to mean aggregation. The
   zero-shot max blend remains 0.5 because its submitted unseen AUROC 0.7428
   was marginally above pure mean 0.7426; its setting is now decoupled from
   Dinomaly visibility aggregation.
5. `best_model.pt` may contain weights captured before the final step while
   also recording `training_completed_steps` after the run finishes. The old
   result code compared the snapshot step (2660) to the configured total and
   incorrectly emitted `partial_diagnostic`. Completion now uses
   `training_completed_steps`, and reports both values for Dinomaly and the
   zero-shot checkpoint.
6. Test_C reused the official packaging namespace and created a ZIP, which was
   easy to mistake for Test_B output. Test_C predictions now live under
   `testc_predictions`, are not zipped by default, and always report
   `competition_submit_ready=false`. A submission-ready ZIP is produced only
   by the competition pipeline scanning the configured official split.

## Next controlled experiments

Retrain because both Dinomaly and zero-shot training fingerprints changed.
Use the fixed 6000-step Dinomaly and 5000-step zero-shot schedules, keep the
Test_C protocol unchanged, and compare against this run per category. First
acceptance targets are improved unseen pixel AUROC/AP without losing more than
0.01 unseen classification AUROC, and improved seen pixel AP/F1. If geometric
categories still regress, the next isolated experiment should expand synthetic
thin-edge, displacement, and missing-part defects; do not introduce Test_C GT
into training, checkpoint selection, calibration, or category-specific rules.
