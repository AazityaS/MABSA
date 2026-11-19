# HAMOS Prediction Cardinality and Objectness Audit

## Scope and method

This audit evaluates the frozen text-only and multimodal checkpoints on **training and development only**. It does not load or evaluate test examples, labels, or predictions. Checkpoints, configurations, caches, source annotations, split files, and prior reports were not modified. The feature cache key used by both checkpoints is `9072ce0b71cd3470`.

The diagnostic reconstructed Hungarian assignments with the current loss, treated assigned slots as objectness positives and all remaining slots as no-object slots, then inspected sigmoid probabilities before thresholding. Decoded cardinality is the number of the 23 slots whose objectness probability is at least the checkpoint threshold. “Nearly all slots” is operationally defined here as at least 20 of 23.

Split support observed: train **8,813** segments; dev **1,895** segments. Every segment in these splits has at least one gold quad. There are 16,784 gold quads in train and 3,600 in dev.

## 1. Prediction cardinality

Statistics are population standard deviation. The same gold set cardinality distribution applies to each mode because eligibility and annotations are shared.

| Split | Gold mean | Gold median | Gold SD | Gold min–max |
|---|---:|---:|---:|---:|
| Train | 1.90446 | 1 | 1.42864 | 1–23 |
| Dev | 1.89974 | 1 | 1.38765 | 1–15 |

| Mode / split | Threshold | Predicted mean | Median | SD | Min–max | Segments above gold | At least 20 slots | Exactly 23 slots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text / train | 0.70 | 19.25712 | 20 | 3.73506 | 6–23 | 8,812 / 8,813 (99.99%) | 4,661 (52.89%) | 3,607 (40.93%) |
| Text / dev | 0.70 | 19.26227 | 20 | 3.73281 | 10–23 | 1,895 / 1,895 (100.00%) | 1,011 (53.35%) | 779 (41.11%) |
| Multimodal / train | 0.65 | 20.28628 | 22 | 3.00985 | 3–23 | 8,812 / 8,813 (99.99%) | 5,319 (60.35%) | 2,618 (29.71%) |
| Multimodal / dev | 0.65 | 20.36412 | 22 | 2.92769 | 4–23 | 1,895 / 1,895 (100.00%) | 1,173 (61.90%) | 551 (29.08%) |

### Gold cardinality histograms

The following entries are `gold quads per segment: number of segments`.

- **Train:** 1:4541, 2:2505, 3:927, 4:419, 5:167, 6:104, 7:54, 8:40, 9:19, 10:10, 11:9, 12:5, 13:9, 14:1, 15:1, 20:1, 23:1.
- **Dev:** 1:984, 2:519, 3:210, 4:87, 5:44, 6:23, 7:11, 8:7, 9:4, 10:2, 12:1, 13:2, 15:1.

### Predicted cardinality histograms at each saved checkpoint threshold

Entries are `emitted slots per segment: number of segments`; omitted cardinalities have count zero.

- **Text / train, threshold .70:** 6:2, 7:1, 8:4, 10:28, 11:209, 12:75, 13:244, 14:558, 15:388, 16:1057, 17:1045, 18:78, 19:463, 20:787, 21:74, 22:193, 23:3607.
- **Text / dev, threshold .70:** 10:10, 11:47, 12:14, 13:50, 14:125, 15:67, 16:223, 17:250, 18:10, 19:88, 20:180, 21:19, 22:33, 23:779.
- **Multimodal / train, threshold .65:** 3:1, 4:11, 5:12, 6:13, 8:7, 9:3, 10:19, 11:71, 12:8, 13:110, 14:154, 15:324, 16:76, 17:328, 18:1452, 19:905, 20:332, 21:93, 22:2276, 23:2618.
- **Multimodal / dev, threshold .65:** 4:2, 5:1, 6:1, 7:1, 8:4, 9:1, 10:2, 11:18, 12:1, 13:11, 14:30, 15:79, 16:20, 17:60, 18:312, 19:179, 20:76, 21:17, 22:529, 23:551.

The overprediction is already present on train and dev, so it is not an artifact unique to the final test split. Text predictions emit around ten times the mean gold cardinality; multimodal predictions emit slightly more. The model's logits are not constrained to produce one slot per gold item; thresholding alone currently leaves most slots active.

## 2. Objectness probabilities and development threshold sweep

Each probability distribution is over Hungarian-assigned positive slots or unmatched slots. Quantiles below are probability values; `Q05` etc. denote percentiles. Large overlap between distributions is evidence of weak separation.

| Mode / split | Group | N | Mean (SD) | Q05 | Q25 | Q50 | Q75 | Q95 | Q99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text / train | Matched positive | 16,784 | .9439 (.0685) | .8079 | .9257 | .9680 | .9907 | .9950 | .9962 |
| Text / train | Unmatched | 185,915 | .8284 (.1496) | .5033 | .7617 | .8740 | .9404 | .9842 | .9948 |
| Text / dev | Matched positive | 3,600 | .9426 (.0681) | .8072 | .9224 | .9669 | .9908 | .9950 | .9960 |
| Text / dev | Unmatched | 39,985 | .8281 (.1489) | .5044 | .7623 | .8732 | .9390 | .9838 | .9947 |
| Multimodal / train | Matched positive | 16,784 | .9389 (.0720) | .7965 | .9150 | .9669 | .9887 | .9950 | .9971 |
| Multimodal / train | Unmatched | 185,915 | .8257 (.1486) | .5164 | .7563 | .8655 | .9392 | .9863 | .9947 |
| Multimodal / dev | Matched positive | 3,600 | .9369 (.0729) | .7907 | .9106 | .9663 | .9887 | .9949 | .9971 |
| Multimodal / dev | Unmatched | 39,985 | .8270 (.1461) | .5220 | .7586 | .8654 | .9384 | .9856 | .9947 |

At both saved thresholds, the **median unmatched-slot probability exceeds the threshold** (.873 text vs .70; .865 multimodal vs .65). The positive and unmatched distributions overlap substantially. Raising the threshold can reduce count, but the approved primary metric deteriorates sharply once positive slots are removed.

The complete development sweep is the configured global grid, 0.00 through 1.00 in increments of 0.05. F1 is the development primary implicit category–sentiment micro-F1. Mean predictions count all emitted slots, not only aspect-implicit ones.

| Threshold | Text P | Text R | Text F1 | Text mean slots | Multimodal P | Multimodal R | Multimodal F1 | Multimodal mean slots |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| .00 | .04158 | .47899 | .07651 | 23.000 | .06298 | .41176 | .10925 | 23.000 |
| .05 | .04158 | .47899 | .07651 | 23.000 | .06298 | .41176 | .10925 | 23.000 |
| .10 | .04158 | .47899 | .07651 | 23.000 | .06314 | .41176 | .10950 | 22.999 |
| .15 | .04158 | .47899 | .07651 | 23.000 | .06323 | .41176 | .10962 | 22.998 |
| .20 | .04158 | .47899 | .07651 | 22.998 | .06364 | .41176 | .11024 | 22.985 |
| .25 | .04158 | .47899 | .07651 | 22.982 | .06414 | .41176 | .11099 | 22.960 |
| .30 | .04161 | .47899 | .07656 | 22.937 | .06499 | .41176 | .11226 | 22.893 |
| .35 | .04164 | .47899 | .07661 | 22.825 | .06452 | .40336 | .11124 | 22.791 |
| .40 | .04179 | .47899 | .07687 | 22.631 | .06566 | .40336 | .11294 | 22.655 |
| .45 | .04216 | .47899 | .07750 | 22.351 | .06685 | .40336 | .11470 | 22.450 |
| .50 | .04328 | .47899 | .07939 | 21.986 | .06906 | .40336 | .11794 | 22.124 |
| .55 | .04653 | .47899 | .08482 | 21.521 | .07154 | .40336 | .12152 | 21.687 |
| .60 | .05234 | .47059 | .09420 | 20.941 | .07367 | .39496 | .12417 | 21.126 |
| .65 | .05876 | .46218 | .10427 | 20.225 | .07993 | .39496 | .13296 | 20.364 |
| .70 | .06606 | .42857 | **.11448** | 19.262 | .08073 | .36975 | .13253 | 19.355 |
| .75 | .06609 | .31933 | .10951 | 18.050 | .07943 | .32773 | .12787 | 17.948 |
| .80 | .07143 | .20168 | .10549 | 16.297 | .08235 | .29412 | .12868 | 15.985 |
| .85 | .07874 | .08403 | .08130 | 13.731 | .08434 | .23529 | .12417 | 13.185 |
| .90 | .16667 | .00840 | .01600 | 10.147 | .06787 | .12605 | .08824 | 9.673 |
| .95 | .00000 | .00000 | .00000 | 5.489 | .04110 | .02521 | .03125 | 5.409 |
| 1.00 | .00000 | .00000 | .00000 | 0.000 | .00000 | .00000 | .00000 | 0.000 |

The approved development-only procedure selects text **0.70** (F1 .11448, TP 51 / FP 721 / FN 68) and multimodal **0.65** (F1 .13296, TP 47 / FP 541 / FN 72). These are exactly the saved checkpoint thresholds and are the maxima on the configured grid, with the protocol's highest-threshold tie break. They are reasonable for the defined primary F1 objective, but plainly not adequate for overall set cardinality. Do not replace them for the frozen experiment or use test data to recalibrate them.

## 3. Loss and matching investigation

### Assignment and supervision

`StructuredSetLoss` forms a rectangular slot-by-gold cost from category negative log likelihood, sentiment negative log likelihood, and the two implicitness BCE costs. SciPy's linear-sum assignment then matches every gold quad to a distinct slot (the model has 23 slots and the data validator ensures no training set exceeds that). Objectness is **not** part of the Hungarian matching cost. Matched slots receive positive objectness BCE; all other slots receive no-object BCE. Duplicate-valued gold quads remain separate columns/items and can receive separate slots.

This is consistent with the approved cost design. Because matching ignores objectness, a slot can be assigned as a target based on its task heads even if its objectness score was initially low; after training it still receives a positive objectness label. That coupling is standard for set prediction and is not, by itself, a code defect.

### Reduction and relative weights

The implementation takes a mean over all matched target losses, a separate mean over matched positive-objectness BCE, and a separate mean over unmatched-slot BCE. Total loss is:

`matched_mean + 1.0 * positive_objectness_mean + 0.1 * unmatched_objectness_mean`.

No-object BCE is averaged, not summed across slots. The weighted no-object term therefore does not grow with the approximately 21 unmatched slots per segment. Since there are on average only 1.90 gold quads, the coefficient and independent means make the per-unmatched-slot influence much smaller than the per-positive-slot influence. The measured no-object probabilities remain high despite a substantial raw BCE.

| Checkpoint / split | Matched loss mean | Positive-objectness mean | Raw no-object mean | Weighted no-object contribution | Total loss mean |
|---|---:|---:|---:|---:|---:|
| Text / train | .11682 | .06142 | 2.20404 | .22040 | .39864 |
| Text / dev | .13982 | .06152 | 2.19144 | .21914 | .42048 |
| Multimodal / train | .13233 | .06688 | 2.18988 | .21899 | .41820 |
| Multimodal / dev | .15957 | .06816 | 2.18196 | .21820 | .44592 |

So the no-object term is **not numerically absent**: after weighting, it is the largest individual term in these checkpoint measurements. The design concern is its 0.1 coefficient combined with a mean over all unmatched slots, not a missing or summed/averaged implementation bug. The data supports testing stronger no-object supervision, but does not by itself establish the optimal coefficient.

### Head gradient check

For an additional directional check, gradients of the full loss were measured through each prediction-head parameter group on the first 32-item training batch, with the frozen checkpoint weights and no optimizer step. These are L2 norms and are batch-specific, not an epoch-wide attribution.

| Mode | Objectness head | Category head | Sentiment head | Aspect-implicit head | Opinion-implicit head | Slot embeddings |
|---|---:|---:|---:|---:|---:|---:|
| Text | .2833 | .0937 | .3289 | .5443 | .0519 | .4063 |
| Multimodal | .3093 | .0538 | .2613 | .7388 | .1050 | .2467 |

The gradients are not balanced uniformly across heads. The aspect-implicit head has the largest measured head norm in both batches; text opinion-implicit and multimodal category heads are notably smaller. This does not prove that the loss is dominated by a particular task across training, but it is a reason to log per-component losses and head-gradient norms in a challenger run. No defect was found in how these heads are wired to the total loss.

### Loss implementation findings

- **Confirmed implementation bug causing the observed over-emission:** none found.
- **Objective-design weakness:** each slot receives a shared context plus learned slot embedding, and objectness no-object BCE has coefficient 0.1 with a separate mean over the many unmatched slots. This can favor a high-recall/high-cardinality solution.
- **Empty-target edge path:** if a minibatch mixes zero-target and nonempty-target items, `matched_losses` can contain a scalar zero alongside per-item vectors before `torch.cat`; that mixed shape may fail. Train/dev currently have minimum gold cardinality 1, so this path did not cause the measured behavior. It should be fixed and tested before using this pipeline with empty-target examples, but is not required to explain this audit's train/dev counts.
- The current training log records aggregate loss but not its positive/no-object subcomponents or head gradient norms. Consequently the report's component values are checkpoint reevaluations, while the gradient comparison is a single-batch diagnostic, not a reconstruction of historical per-epoch gradients.

## 4. Forward-to-evaluation path

The model returns objectness **logits**. `decode_outputs` applies sigmoid exactly once, uses `probability >= threshold`, and emits at most one item per slot. Category and sentiment use argmax; the two flag logits receive sigmoid and use the saved flag threshold. Training's development threshold selection and evaluation use this same decoder.

No slot-level deduplication is performed. This preserves the approved multiset/set-prediction semantics: two slots with equal labels remain two predictions, and one-to-one evaluation counts extra duplicates as false positives. Cardinality is computed as the length of the emitted slot list, so the measured count correctly corresponds to slots passing the threshold. No double sigmoid, inverted comparison, stale threshold, or off-by-one slot filter was found.

The reason nearly all slots emit is the actual score distribution: unmatched medians are .86–.87, well above both saved thresholds. It is not an inference-time counting bug.

## 5. Development-only improvement options

Ranked by expected benefit for cardinality and implementation risk:

1. **Increase no-object supervision while keeping 23 slots.** Use a separate challenger config and test `objectness_empty=0.5` as a conservative first change (five times the baseline coefficient); keep other losses, data, seed, architecture, and threshold-selection rule fixed. This directly addresses the weak per-unmatched-slot influence. Risk: too much negative pressure can suppress true quads and reduce primary recall. Compare dev primary F1, complete-set F1, cardinality MAE/bias, predicted-count histogram, and objectness positive/negative quantiles. Retain the existing development-only primary F1 threshold rule.
2. **Add an explicit expected-cardinality regularizer if option 1 does not control counts.** Penalize the difference between the sum of slot objectness probabilities and the gold quad count during training, using only training annotations. This gives count supervision directly and does not alter inference inputs. Tune its coefficient on train/dev only. Risk: probability sums need not equal thresholded count, and a poorly scaled penalty may harm set recall or implicit prediction.
3. **Improve calibration/reporting, not by test tuning.** On development predictions, report precision/recall and cardinality over the full score curve, and optionally fit a calibration transform on a training calibration partition or cross-validation. Any selected operating point must be fixed before test use and remain consistent with the approved endpoint. This may improve threshold behavior but cannot repair a poorly trained ranking if positive and negative scores overlap; a calibration-only change may also trade recall for count without improving the primary endpoint.

Reducing slot count is not recommended: 23 is required because a training segment has 23 gold quads. Changing the Hungarian cost is lower priority: matching is maximum-cardinality and uses the approved category/sentiment/implicitness costs; no mismatch with the current protocol was found. Consider revisiting it only after controlled loss experiments.

## 6. Challenger training plan and estimate

After approval, create a **new** challenger config and output directory; do not alter the frozen baseline config, logs, cache, or checkpoints. Train text-only first and multimodal second, sequentially, with the same validated feature cache, official train/dev splits, seed 1729, batch size 32, 23 slots, early-stopping rule, development primary metric, threshold grid, and matching/evaluation protocol. The only change in the first challenger pair should be `objectness_empty=0.5`. Save separate best checkpoints and per-epoch component-loss/cardinality diagnostics. Select/check thresholds from dev only. Do not inspect test data until both challenger runs are complete and separately authorized for a final comparison.

The prior frozen training summaries report **708.619 s** for text-only and **702.672 s** for multimodal (about **23 min 31 s combined** on the configured RTX 3050 6 GB, excluding setup). A first challenger pair with the same epoch and patience settings is therefore estimated at **about 24 minutes**, with practical allowance of **25–30 minutes** for startup and variation. This is an estimate from actual previous run durations, not a new training run.

## Conclusion

The cardinality problem is real on both train and dev. At saved thresholds, the models emit a mean of 19–20 items while gold averages about 1.9; 99.99–100% of segments overpredict. Objectness positives score highly, but unmatched slots also score highly, and their score distributions overlap. Threshold selection correctly maximizes the approved dev primary F1, but that endpoint does not reward correct overall cardinality. The implementation applies sigmoid, thresholds, filters, preserves duplicate items, and counts emitted slots correctly. The leading actionable issue is objective design: stronger relative no-object supervision is the lowest-complexity controlled challenger, with explicit cardinality regularization as the next option if needed.

No fixes or training were performed in this task. The original experiments and artifacts remain frozen. No test examples, test labels, or test predictions were accessed.
