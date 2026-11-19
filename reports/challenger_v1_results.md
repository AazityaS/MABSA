# Challenger V1 Development Results

## Executive summary

This report contains **development-split results only**. It does not reproduce, recalculate, or compare against any previously reported test metrics. The test split was not loaded or evaluated for this analysis.

Increasing the unmatched-slot loss weight from 0.1 to 0.5 reduced development cardinality for both challengers. The text-only challenger also improved the primary development F1 substantially, from 0.1145 to 0.2727, while reducing recall from 0.4286 to 0.3277. The multimodal challenger reduced mean output size more sharply, but its primary F1 increased only from 0.1330 to 0.1405 while recall fell from 0.3950 to 0.1429. Thus V1 is a promising text-only result; the multimodal result is primarily a precision/cardinality trade-off and remains weak for implicit aspect-sentiment detection.

Neither challenger has solved set cardinality. The text challenger still predicts a mean of 12.45 items against 1.90 gold items; multimodal predicts 8.32. The early multimodal epoch with threshold 1.0 and zero outputs was not selected: the final multimodal checkpoint is epoch 5 with threshold 0.60.

## Experimental setup and integrity

The original experiments remain frozen. Both challenger runs used the original architecture and 23 slots, cached XLM-R/CLIP features, matching and other loss weights, optimizer, learning rate, weight decay, batch size 32, seed 1729, epoch limit and patience. The only intended training parameter change was:

`loss_weights.objectness_empty: 0.1 → 0.5`

The challenger config was compared recursively with the baseline config; this was the sole difference. Both use cache key `9072ce0b71cd3470` (text features 768 dimensions; image features 512 dimensions). Runs were sequential. Both completed without warnings or errors and stopped via patience. The trainer's opt-in instrumentation records per-epoch total and component losses, matched/unmatched objectness probability quantiles, development metrics/cardinality, and pre-clipping prediction-head gradient norms. It does not alter loss computation, model outputs, threshold selection, or optimizer updates.

The evaluation loaded the four saved checkpoints on the **same 1,895 development segments**. Segment IDs, ordered gold target sets, and language labels were asserted equal across all four conditions. All metrics use the existing `decode_outputs`, `evaluate_structured`, and `evaluate_by_language` definitions; no matching rule or threshold was changed. Objectness summaries divide slots by the existing Hungarian assignment: matched slots are positive and unmatched slots are negative.

| Model | Checkpoint | Best epoch | Saved dev threshold | Training duration |
|---|---|---:|---:|---:|
| Original text-only | `runs/structured/text/best_checkpoint.pt` | 4 | 0.70 | 708.619 s (11m 49s) |
| Original multimodal | `runs/structured/multimodal/best_checkpoint.pt` | 3 | 0.65 | 702.672 s (11m 43s) |
| Challenger V1 text-only | `runs/structured/challenger_v1/text/best_checkpoint.pt` | 15 | 0.45 | 1,623.680 s (27m 04s) |
| Challenger V1 multimodal | `runs/structured/challenger_v1/multimodal/best_checkpoint.pt` | 5 | 0.60 | 814.967 s (13m 35s) |

The challenger thresholds are the saved development-selected thresholds. In the multimodal challenger log, epoch 1 selected threshold 1.0 and emitted no items, yielding zero primary F1; the best checkpoint instead comes from epoch 5 at threshold 0.60.

## Four-model development comparison

Primary implicit category–sentiment precision/recall/F1 use the approved aspect-implicit gate and exact category–sentiment one-to-one scoring. Complete-set metrics use the approved complete structured item matching. `Exact` is exact-set accuracy over segments.

| Model | Primary P | Primary R | Primary F1 | TP / FP / FN | Complete P | Complete R | Complete F1 | Exact-set accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original text-only | 0.0661 | 0.4286 | 0.1145 | 51 / 721 / 68 | 0.0927 | 0.9394 | 0.1687 | 0.0000 |
| Original multimodal | 0.0799 | 0.3950 | 0.1330 | 47 / 541 / 72 | 0.0869 | 0.9319 | 0.1590 | 0.0000 |
| Challenger V1 text-only | 0.2335 | 0.3277 | **0.2727** | 39 / 128 / 80 | 0.1378 | 0.9028 | 0.2391 | 0.0000 |
| Challenger V1 multimodal | 0.1382 | 0.1429 | 0.1405 | 17 / 106 / 102 | 0.1740 | 0.7622 | **0.2834** | 0.0005 (1/1,895) |

### Cardinality comparison

Gold cardinality is shared by all four conditions: **3,600 quads over 1,895 segments**, mean **1.8997** gold quads per segment. The table reports all decoded slots at each checkpoint's saved threshold. Positive signed bias means overprediction.

| Model | Predicted mean | Median | SD | Min–max | Gold mean | Cardinality MAE | Signed bias | Segments with ≥20 outputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original text-only | 19.2623 | 20 | 3.7328 | 10–23 | 1.8997 | 17.3625 | +17.3625 | 1,011 (53.35%) |
| Original multimodal | 20.3641 | 22 | 2.9277 | 4–23 | 1.8997 | 18.4644 | +18.4644 | 1,173 (61.90%) |
| Challenger V1 text-only | 12.4480 | 12 | 6.3725 | 3–23 | 1.8997 | 10.5483 | +10.5483 | 393 (20.74%) |
| Challenger V1 multimodal | 8.3208 | 6 | 5.5026 | 1–23 | 1.8997 | 6.4222 | +6.4211 | 151 (7.97%) |

Relative to its corresponding baseline, text mean cardinality fell 35.4% and MAE fell 39.2%; multimodal mean cardinality fell 59.1% and MAE fell 65.2%. Both are still substantially above the gold count on average. The multimodal signed bias differs slightly from MAE because a small number of segments underpredict, while absolute error cannot cancel.

Predicted cardinality histograms are `emitted slots: segment count`; omitted cardinalities have zero count.

- **Original text-only:** 10:10, 11:47, 12:14, 13:50, 14:125, 15:67, 16:223, 17:250, 18:10, 19:88, 20:180, 21:19, 22:33, 23:779.
- **Original multimodal:** 4:2, 5:1, 6:1, 7:1, 8:4, 9:1, 10:2, 11:18, 12:1, 13:11, 14:30, 15:79, 16:20, 17:60, 18:312, 19:179, 20:76, 21:17, 22:529, 23:551.
- **Challenger V1 text-only:** 3:18, 4:50, 5:288, 6:171, 7:43, 8:74, 9:51, 10:99, 11:24, 12:341, 13:78, 14:20, 15:83, 16:52, 17:45, 18:55, 19:10, 20:30, 21:35, 22:24, 23:304.
- **Challenger V1 multimodal:** 1:1, 2:4, 3:71, 4:436, 5:93, 6:609, 7:74, 8:16, 9:43, 10:96, 11:29, 12:63, 13:78, 14:13, 15:1, 16:4, 17:53, 18:17, 19:43, 20:34, 21:21, 22:13, 23:83.

## Objectness score analysis

Development objectness quantiles for Hungarian-matched positive slots and unmatched slots:

| Model | Slot group | Mean | SD | P05 | P25 | Median | P75 | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original text-only | Matched positive | .9426 | .0681 | .8072 | .9224 | .9669 | .9908 | .9950 | .9960 |
| Original text-only | Unmatched | .8281 | .1489 | .5044 | .7623 | .8732 | .9390 | .9838 | .9947 |
| Original multimodal | Matched positive | .9369 | .0729 | .7907 | .9106 | .9663 | .9887 | .9949 | .9971 |
| Original multimodal | Unmatched | .8270 | .1461 | .5220 | .7586 | .8654 | .9384 | .9856 | .9947 |
| Challenger V1 text-only | Matched positive | .7928 | .1952 | .3873 | .6856 | .8634 | .9523 | .9804 | .9859 |
| Challenger V1 text-only | Unmatched | .4588 | .2712 | .0530 | .2142 | .4560 | .6803 | .9016 | .9597 |
| Challenger V1 multimodal | Matched positive | .7592 | .2065 | .3501 | .6164 | .8319 | .9430 | .9715 | .9793 |
| Challenger V1 multimodal | Unmatched | .4953 | .2339 | .1533 | .3067 | .4681 | .6724 | .9098 | .9691 |

The positive-minus-unmatched mean gap increased from **0.1145 to 0.3340** for text and **0.1099 to 0.2639** for multimodal. Median gaps increased from **0.0938 to 0.4074** and **0.1009 to 0.3638**, respectively. This is materially better separation, but overlap remains, particularly in the high-score unmatched tail (unmatched P95 is about .90 for both challengers). The score change is consistent with the much smaller predicted sets.

## Implicitness detection

Positive-class scores treat implicit as positive. Counts are under the existing structured metric assignment.

| Model | Aspect implicit P / R / F1 (TP/FP/FN) | Opinion implicit P / R / F1 (TP/FP/FN) |
|---|---|---|
| Original text-only | .0855 / .5546 / .1481 (66/706/53) | .0000 / .0000 / .0000 (0/9/48) |
| Original multimodal | .1054 / .5210 / .1754 (62/526/57) | .1538 / .0417 / .0656 (2/11/46) |
| Challenger V1 text-only | .2994 / .4202 / .3497 (50/117/69) | .0667 / .0417 / .0513 (2/28/46) |
| Challenger V1 multimodal | .1789 / .1849 / .1818 (22/101/97) | .0000 / .0000 / .0000 (0/12/48) |

The end-to-end implicit component macro-F1 values use the primary implicit prediction gate and fixed label sets:

| Model | Category macro-F1 | Sentiment macro-F1 |
|---|---:|---:|
| Original text-only | .06484 | .05407 |
| Original multimodal | .06192 | .09695 |
| Challenger V1 text-only | .16803 | .12802 |
| Challenger V1 multimodal | .07098 | .08621 |

Text-only improved on both component macros. Multimodal improved slightly in category macro-F1 but declined in sentiment macro-F1. Opinion implicitness remains poorly detected in all conditions; these results do not support a claim of broad improvement across every structured target.

## Per-language results and support

Support is the same for all four models: English (`en`) has 1,542 segments, 2,859 gold quads, and 38 gold implicit quads; Vietnamese (`vi`) has 298 segments, 617 gold quads, and 77 gold implicit quads; `other` has 55 segments, 124 gold quads, and 4 gold implicit quads. The language metadata is used only to group evaluation; it is not an added model feature.

| Language | Model | Primary TP / FP / FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| en | Original text-only | 14 / 246 / 24 | .0538 | .3684 | .0940 |
| en | Original multimodal | 4 / 13 / 34 | .2353 | .1053 | .1455 |
| en | Challenger V1 text-only | 12 / 34 / 26 | .2609 | .3158 | .2857 |
| en | Challenger V1 multimodal | 0 / 0 / 38 | .0000 | .0000 | .0000 |
| vi | Original text-only | 36 / 378 / 41 | .0870 | .4675 | .1466 |
| vi | Original multimodal | 42 / 504 / 35 | .0769 | .5455 | .1348 |
| vi | Challenger V1 text-only | 26 / 80 / 51 | .2453 | .3377 | .2842 |
| vi | Challenger V1 multimodal | 17 / 104 / 60 | .1405 | .2208 | .1717 |
| other | Original text-only | 1 / 97 / 3 | .0102 | .2500 | .0196 |
| other | Original multimodal | 1 / 24 / 3 | .0400 | .2500 | .0690 |
| other | Challenger V1 text-only | 1 / 14 / 3 | .0667 | .2500 | .1053 |
| other | Challenger V1 multimodal | 0 / 2 / 4 | .0000 | .0000 | .0000 |

Per-language end-to-end category and sentiment macro-F1 are withheld by the existing support safeguard: every language has at least one fixed class with fewer than five gold examples. In particular, `other` has only four gold implicit quads. The language-specific scores are descriptive and should not be treated as stable comparative evidence.

## Interpretation

### Text-only

The text-only challenger reduced primary false positives from 721 to 128 (82.2%) and increased precision from .0661 to .2335. Recall fell by 10.1 percentage points (.4286 to .3277), while primary F1 rose by .1582 (.1145 to .2727). Complete-set F1 also rose (.1687 to .2391), and cardinality MAE fell by 6.8143 items. This is evidence of improvement on the development prediction task, not merely a smaller output set: primary F1 and complete-set F1 both rose despite lower recall.

### Multimodal

The multimodal challenger reduced primary false positives from 541 to 106 (80.4%), but recall fell by 25.2 percentage points (.3950 to .1429). Primary F1 increased only .0075 (.1330 to .1405). Complete-set F1 improved (.1590 to .2834), and cardinality MAE fell by 12.0422. The primary-task gain is modest and comes with a severe recall trade-off. The model still averages 8.32 predictions against 1.90 gold, so substantial overprediction remains even after this reduction.

The selected thresholds differ across models because the same development F1 selection rule was reapplied: .70 and .65 for the originals; .45 and .60 for V1. The controlled change was the no-object coefficient, but this is one training seed and threshold selection is part of the fixed experimental procedure. These outcomes support the hypothesis that stronger no-object supervision can improve objectness separation and precision in these runs; they do not establish that 0.5 is generally optimal or that the change alone caused the full score differences.

## Limitations and threats to validity

- All results here are development results used during model selection. They are not independent final estimates and must remain distinct from any prior test report.
- The dev set has 119 implicit quads, with uneven distribution across languages and classes. Language-level estimates are noisy; macro-F1 is correctly withheld where class support is insufficient.
- This is a single-seed comparison. The findings may reflect seed or threshold-grid variation.
- A large precision increase can coexist with a recall collapse. The multimodal V1 primary F1 gain is small, and its implicit recall is low.
- Cardinality is still materially above gold. The metric is averaged per segment; mean predicted cardinality does not imply every segment has the same error.
- The saved development threshold is selected for primary implicit F1, not complete-set cardinality. Cardinality changes reported here are measured at that protocol-selected operating point.
- No hotel-clustered interval was computed for dev. No statistical-significance claim is made.

## Recommendation for next step

Treat Challenger V1 text-only as promising, but not yet as a final replacement: inspect its development errors and consider a separately approved seed replication before any test comparison. Treat Challenger V1 multimodal as inconclusive for the primary question: it lowers false positives and cardinality but sacrifices substantial recall and gains only 0.0075 primary F1. Continue to report its cardinality and precision/recall trade-off rather than calling it a clear multimodal improvement. Do not change thresholds or start another training run from this report alone.

## Artifacts and validation

- Challenger config: `configs/structured_challenger_v1.json`; copied run config: `runs/structured/challenger_v1/config.json`.
- Per-epoch training diagnostics: `runs/structured/challenger_v1/text/training_log.jsonl` and `runs/structured/challenger_v1/multimodal/training_log.jsonl`.
- Run summaries/checkpoints: under `runs/structured/challenger_v1/{text,multimodal}/`.
- Four-checkpoint dev-only metrics: `runs/structured/challenger_v1/dev_metrics.json`.
- Verification confirmed all four checkpoints load, thresholds and best epochs match their saved training logs, cache keys match, and the four evaluation signatures (development segment IDs, gold target sets, and languages) are identical.
- The transient formatting failure in an earlier dev-only helper occurred before any data was loaded; it produced no artifact. The corrected dev-only evaluation completed successfully. No model was retrained, no existing report or baseline artifact was overwritten, and no test split was accessed.
