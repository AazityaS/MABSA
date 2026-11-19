# Final Test Evaluation

## Evaluation integrity and support

The single approved final evaluation was run with `.venv-hamos-cuda\\Scripts\\python.exe` on CUDA. Both models were evaluated sequentially by the documented evaluator. Checkpoints were not changed and no test threshold selection or tuning was performed.

| Item | Text-only | Multimodal |
|---|---:|---:|
| Checkpoint | `runs/structured/text/best_checkpoint.pt` | `runs/structured/multimodal/best_checkpoint.pt` |
| Saved development threshold | 0.70 | 0.65 |
| Feature cache key | `9072ce0b71cd3470` | `9072ce0b71cd3470` |
| Eligible test segments | 1,890 | 1,890 |
| Gold quads | 3,606 | 3,606 |
| Gold implicit aspect quads | 116 | 116 |
| Hotels | 514 | 514 |

Evaluator integrity checks confirmed identical eligible test segment IDs, gold targets, language labels, and hotel IDs between modes. The saved thresholds match the checkpoints. The result JSON is [final_test_evaluation.json](../runs/structured/final_test_evaluation.json).

## Primary endpoint

The primary endpoint is end-to-end implicit category–sentiment micro-F1: a prediction must be emitted as aspect-implicit, and the category–sentiment pair must match under the approved one-to-one matching. False aspect-implicit predictions contribute false positives; missed gold implicit items contribute false negatives.

| Mode | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Text-only | 0.05125 | 0.31897 | 0.08831 | 37 | 685 | 79 |
| Multimodal | 0.06305 | 0.31034 | 0.10480 | 36 | 535 | 80 |

The multimodal primary F1 is 0.01650 higher on this test set. The paired cluster-bootstrap interval for the difference includes zero; this result alone does not establish a statistically reliable improvement.

## End-to-end implicit component macro-F1

These scores use the primary aspect-implicit prediction gate and the approved component matching. Macro-F1 averages over the fixed six aspect categories or three sentiment labels.

| Mode | Category macro-F1 | Sentiment macro-F1 |
|---|---:|---:|
| Text-only | 0.03028 | 0.08911 |
| Multimodal | 0.04918 | 0.05329 |

Per-class supports and scores are available in the JSON result. In particular, a class with no predicted items can have zero F1 despite nonzero gold support; the fixed class order comes from the actual label mapping.

## Gold-implicit-conditioned diagnostic

This diagnostic matches gold implicit items to all emitted predictions, regardless of predicted aspect implicitness. It prioritizes exact category-and-sentiment pairs, then minimizes category/sentiment Hamming distance. Predicted aspect/opinion implicitness does not affect this assignment. Duplicate items retain multiset semantics. Unmatched gold items count as misses; unmatched predictions are ignored for this diagnostic. “Exact pair rate” is the fraction of gold implicit quads assigned an exact category-and-sentiment prediction.

| Mode | Exact pair rate | Category P / R / F1 (TP/FP/FN) | Sentiment P / R / F1 (TP/FP/FN) |
|---|---:|---|---|
| Text-only | 0.97414 | 0.99138 / 0.99138 / 0.99138 (115/1/1) | 0.98276 / 0.98276 / 0.98276 (114/2/2) |
| Multimodal | 0.93966 | 0.95690 / 0.95690 / 0.95690 (111/5/5) | 0.98276 / 0.98276 / 0.98276 (114/2/2) |

Confusion matrices use gold labels as rows. Prediction columns follow the category/sentiment order and end with `MISS`; unmatched predictions are ignored, as specified above.

**Text-only category** — columns: AMENITY, BRANDING, EXPERIENCE, FACILITY, LOYALTY, SERVICE, MISS

| Gold \\ predicted | AMENITY | BRANDING | EXPERIENCE | FACILITY | LOYALTY | SERVICE | MISS |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMENITY | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| BRANDING | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| EXPERIENCE | 0 | 0 | 37 | 1 | 0 | 0 | 0 |
| FACILITY | 0 | 0 | 0 | 50 | 0 | 0 | 0 |
| LOYALTY | 0 | 0 | 0 | 0 | 9 | 0 | 0 |
| SERVICE | 0 | 0 | 0 | 0 | 0 | 16 | 0 |

**Multimodal category** — same column order

| Gold \\ predicted | AMENITY | BRANDING | EXPERIENCE | FACILITY | LOYALTY | SERVICE | MISS |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMENITY | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| BRANDING | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| EXPERIENCE | 0 | 0 | 33 | 4 | 0 | 1 | 0 |
| FACILITY | 0 | 0 | 0 | 50 | 0 | 0 | 0 |
| LOYALTY | 0 | 0 | 0 | 0 | 9 | 0 | 0 |
| SERVICE | 0 | 0 | 0 | 0 | 0 | 16 | 0 |

**Text-only sentiment** — columns: negative, neutral, positive, MISS

| Gold \\ predicted | negative | neutral | positive | MISS |
|---|---:|---:|---:|---:|
| negative | 6 | 0 | 2 | 0 |
| neutral | 0 | 2 | 0 | 0 |
| positive | 0 | 0 | 106 | 0 |

**Multimodal sentiment** — same column order

| Gold \\ predicted | negative | neutral | positive | MISS |
|---|---:|---:|---:|---:|
| negative | 7 | 0 | 1 | 0 |
| neutral | 1 | 1 | 0 | 0 |
| positive | 0 | 0 | 106 | 0 |

## Complete structured-set results

Complete-set matching requires the full structured item to match, including category, sentiment, aspect implicitness, and opinion implicitness. Extra predictions are false positives and missing gold items are false negatives. Exact-set accuracy requires the full predicted and gold multisets to be identical for a segment.

| Mode | Precision | Recall | F1 | TP | FP | FN | Exact-set accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text-only | 0.09376 | 0.94370 | 0.17058 | 3,403 | 32,890 | 203 | 0.00000 |
| Multimodal | 0.08880 | 0.93927 | 0.16225 | 3,387 | 34,757 | 219 | 0.00000 |

The low exact-set accuracies mean neither model produced an exactly correct complete set for any evaluated segment under this matching definition.

## Implicitness metrics

Positive-class metrics treat “implicit” as positive. Denominators are the matched component counts under the approved component evaluation. Aspect implicitness is relevant to the primary endpoint; opinion implicitness is reported as a secondary structured output.

| Mode | Aspect implicit P / R / F1 (TP/FP/FN) | Opinion implicit P / R / F1 (TP/FP/FN) |
|---|---|---|
| Text-only | 0.07618 / 0.47414 / 0.13126 (55/667/61) | 0.00000 / 0.00000 / 0.00000 (0/8/41) |
| Multimodal | 0.07706 / 0.37931 / 0.12809 (44/527/72) | 0.00000 / 0.00000 / 0.00000 (0/13/41) |

## Cardinality

Signed bias is predicted set cardinality minus gold set cardinality, averaged over eligible segments; positive values indicate over-prediction.

| Mode | Cardinality MAE | Signed bias |
|---|---:|---:|
| Text-only | 17.29471 | +17.29471 |
| Multimodal | 18.27407 | +18.27407 |

## Per-language results

The language label is metadata used only to group evaluation results, not a model input. Support is reported as eligible segments and gold quads. End-to-end component macro-F1 is withheld in every language because at least one fixed class has fewer than five gold examples; class-level results remain in the JSON.

| Language | Segments | Gold quads | Gold implicit | Text primary TP/FP/FN; F1 | Multimodal primary TP/FP/FN; F1 | Text diagnostic exact pair; category F1; sentiment F1 | Multimodal diagnostic exact pair; category F1; sentiment F1 |
|---|---:|---:|---:|---|---|---|---|
| en | 1,540 | 2,822 | 44 | 7/234/37; 0.04912 | 3/12/41; 0.10169 | 0.95455; 1.00000; 0.95455 | 0.93182; 0.95455; 0.97727 |
| vi | 295 | 657 | 61 | 27/354/34; 0.12217 | 30/504/31; 0.10084 | 0.98361; 0.98361; 1.00000 | 0.93443; 0.95082; 0.98361 |
| other | 55 | 127 | 11 | 3/97/8; 0.05405 | 3/19/8; 0.18182 | 1.00000; 1.00000; 1.00000 | 1.00000; 1.00000; 1.00000 |

Per-language positive implicitness metrics (P/R/F1), with denominators in the JSON:

| Language | Text aspect | Multimodal aspect | Text opinion | Multimodal opinion |
|---|---|---|---|---|
| en | 0.03734 / 0.20455 / 0.06316 | 0.20000 / 0.06818 / 0.10169 | 0 / 0 / 0 | 0 / 0 / 0 |
| vi | 0.07116 / 0.62295 / 0.12773 | 0.07116 / 0.62295 / 0.12773 | 0 / 0 / 0 | 0 / 0 / 0 |
| other | 0.07000 / 0.63636 / 0.12613 | 0.13636 / 0.27273 / 0.18182 | 0 / 0 / 0 | 0 / 0 / 0 |

The aggregate “other” language bucket has only 55 segments and 11 gold implicit quads; its perfect diagnostic scores should not be interpreted as stable evidence. Even the larger language groups have small support in several categories and sentiments.

## Paired hotel-clustered bootstrap

The approved paired bootstrap resampled hotels (2,000 replicates, seed 1729), applying the same sampled hotel IDs to both modes in each replicate. The reported difference is multimodal minus text-only primary F1.

| Quantity | Result |
|---|---:|
| Text-only primary F1 95% interval | [0.05747, 0.12276] |
| Multimodal primary F1 95% interval | [0.06842, 0.14658] |
| Mean paired difference | +0.01642 |
| Paired difference 95% interval | [-0.00802, 0.04415] |
| Replicates multimodal > text-only | 1,783 / 2,000 |
| Ties | 0 / 2,000 |
| Replicates text-only > multimodal | 217 / 2,000 |
| Multimodal win proportion | 0.8915 |

The bootstrap win proportion is descriptive and is not, by itself, a significance test. The paired difference interval includes zero.

## Reporting conventions and limitations

- The saved development-selected thresholds (0.70 text-only; 0.65 multimodal) were frozen before test evaluation.
- Primary and diagnostic metrics use their distinct approved assignment rules; duplicates are preserved with multiset semantics. Diagnostic confusion matrices include `MISS` columns for unmatched gold items and ignore unmatched predictions.
- End-to-end implicit category/sentiment component macro-F1 uses the primary implicitness gate. Fixed classes are included in the aggregate macro average, with zero-support classes contributing zero as specified by the protocol. Per-language macros are suppressed unless every fixed class has at least five gold examples.
- Aspect and opinion implicitness positive metrics are secondary; opinion implicitness is not part of the primary endpoint.
- The gold implicit test sample is only 116 quads. Language-specific implicit support is 44 English, 61 Vietnamese, and 11 other. These limited counts, class imbalance, and broad confidence intervals constrain conclusions.
- The primary result is a modest observed multimodal F1 increase (+0.01650), but the paired interval spans negative and positive values. Avoid a claim of established improvement from this one test evaluation.
- Test data was evaluated once, after checkpoint and threshold selection were complete. No training, tuning, image-only evaluation, or additional test experiment was run.
