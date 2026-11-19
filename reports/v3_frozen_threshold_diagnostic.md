# V3-Frozen Threshold Diagnostic

## Scope and data boundary

This is a post-training diagnostic of the existing V3-Frozen epoch-1 checkpoint. It uses official development labels and the verified train/dev cache reader; no test split, test labels, test predictions, or test embedding values were loaded or inspected. The checkpoint, configuration, original implementation, and prior results were not changed. No training was performed.

Checkpoint: `runs/structured/challenger_v3/frozen/best_checkpoint.pt` (epoch 1); stored threshold `0.00`. Development sample: 1,895 segments, 98 positive and 34,012 negative label cells across 18 labels.

The threshold comparison is **probability >= threshold**, matching `src/v3_presence_metrics.py`. Equal scores are included together. Exact-sweep F1 ties are resolved by choosing the highest threshold, consistent with the approved global-threshold tie rule.

## Probability distributions

All statistics are on raw sigmoid probabilities from the saved epoch-1 classifier. Quantiles use NumPy's default linear interpolation. The positive/negative rows pool cells across labels; the 18-row table gives a separate distribution for each output label over all 1,895 development segments.

| Gold-cell group | N | Min | Max | Mean | Median | P01 | P05 | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All label cells | 34110 | 0.000909 | 0.227627 | 0.087184 | 0.080606 | 0.012158 | 0.029676 | 0.040428 | 0.058476 | 0.080606 | 0.110328 | 0.147949 | 0.160910 | 0.191340 |
| Positive gold cells | 98 | 0.002701 | 0.162138 | 0.059448 | 0.058447 | 0.003017 | 0.011045 | 0.015304 | 0.039986 | 0.058447 | 0.077520 | 0.096933 | 0.111930 | 0.137449 |
| Negative gold cells | 34012 | 0.000909 | 0.227627 | 0.087264 | 0.080691 | 0.012353 | 0.029795 | 0.040496 | 0.058528 | 0.080691 | 0.110418 | 0.148021 | 0.160941 | 0.191370 |

| Output label | Gold positives | Min | Max | Mean | Median | P01 | P05 | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMENITY / negative | 1 | 0.001860 | 0.122490 | 0.077665 | 0.079916 | 0.012808 | 0.033070 | 0.043854 | 0.062644 | 0.079916 | 0.098910 | 0.106061 | 0.108613 | 0.113430 |
| AMENITY / neutral | 0 | 0.001593 | 0.182861 | 0.111375 | 0.114300 | 0.014408 | 0.041688 | 0.058423 | 0.087049 | 0.114300 | 0.145240 | 0.156324 | 0.160527 | 0.168278 |
| AMENITY / positive | 2 | 0.001264 | 0.089623 | 0.055879 | 0.057325 | 0.008470 | 0.022632 | 0.030552 | 0.044387 | 0.057325 | 0.071903 | 0.077437 | 0.079428 | 0.083245 |
| BRANDING / negative | 0 | 0.001460 | 0.142613 | 0.088482 | 0.090851 | 0.012594 | 0.034739 | 0.048301 | 0.070237 | 0.090851 | 0.113903 | 0.122723 | 0.125788 | 0.131280 |
| BRANDING / neutral | 0 | 0.000909 | 0.088226 | 0.053807 | 0.054881 | 0.007493 | 0.020751 | 0.028415 | 0.042035 | 0.054881 | 0.070009 | 0.075806 | 0.077838 | 0.081356 |
| BRANDING / positive | 0 | 0.002374 | 0.181718 | 0.117036 | 0.120621 | 0.018339 | 0.048812 | 0.066793 | 0.094560 | 0.120621 | 0.149222 | 0.159338 | 0.162863 | 0.169857 |
| EXPERIENCE / negative | 4 | 0.001250 | 0.127546 | 0.077188 | 0.078900 | 0.010732 | 0.029639 | 0.041413 | 0.060415 | 0.078900 | 0.100302 | 0.108129 | 0.111263 | 0.116525 |
| EXPERIENCE / neutral | 0 | 0.002067 | 0.227627 | 0.143300 | 0.147216 | 0.021422 | 0.057327 | 0.078818 | 0.114501 | 0.147216 | 0.184066 | 0.197395 | 0.202272 | 0.211118 |
| EXPERIENCE / positive | 30 | 0.002281 | 0.112188 | 0.073827 | 0.076010 | 0.014194 | 0.033550 | 0.044146 | 0.060901 | 0.076010 | 0.092589 | 0.098823 | 0.100958 | 0.105301 |
| FACILITY / negative | 3 | 0.001990 | 0.182642 | 0.113115 | 0.116515 | 0.016288 | 0.045031 | 0.061656 | 0.089313 | 0.116515 | 0.145536 | 0.156568 | 0.160633 | 0.168506 |
| FACILITY / neutral | 0 | 0.001015 | 0.081054 | 0.049981 | 0.051159 | 0.007644 | 0.019932 | 0.027208 | 0.039360 | 0.051159 | 0.064400 | 0.069784 | 0.071579 | 0.075079 |
| FACILITY / positive | 34 | 0.001717 | 0.088836 | 0.056385 | 0.057733 | 0.010129 | 0.024385 | 0.032567 | 0.045404 | 0.057733 | 0.071797 | 0.076898 | 0.078589 | 0.081992 |
| LOYALTY / negative | 0 | 0.001531 | 0.157333 | 0.097237 | 0.099750 | 0.013353 | 0.037794 | 0.052363 | 0.076661 | 0.099750 | 0.126100 | 0.135507 | 0.138752 | 0.144956 |
| LOYALTY / neutral | 0 | 0.001985 | 0.196712 | 0.123348 | 0.126810 | 0.017518 | 0.048764 | 0.067090 | 0.098075 | 0.126810 | 0.158844 | 0.170804 | 0.175067 | 0.182681 |
| LOYALTY / positive | 11 | 0.001093 | 0.125874 | 0.076332 | 0.078235 | 0.009928 | 0.029276 | 0.040440 | 0.059669 | 0.078235 | 0.099205 | 0.107281 | 0.110167 | 0.115831 |
| SERVICE / negative | 1 | 0.001087 | 0.107455 | 0.065941 | 0.067603 | 0.009107 | 0.025165 | 0.035023 | 0.051647 | 0.067603 | 0.085668 | 0.092466 | 0.094976 | 0.099285 |
| SERVICE / neutral | 0 | 0.002064 | 0.115472 | 0.073464 | 0.075655 | 0.012797 | 0.031788 | 0.042326 | 0.059491 | 0.075655 | 0.092993 | 0.099586 | 0.101992 | 0.106632 |
| SERVICE / positive | 12 | 0.001683 | 0.186269 | 0.114954 | 0.117907 | 0.015376 | 0.043689 | 0.060851 | 0.090528 | 0.117907 | 0.148599 | 0.160540 | 0.164692 | 0.172870 |

Positive/negative distributions have broad support overlap: their observed ranges intersect from 0.002701 to 0.162138. The common 100-bin [0,1] histogram overlap coefficient is 0.6998 (1 means identical binned class-conditional distributions). Positive cells have lower mean probability (0.059448) than negative cells (0.087264).

## Exact global threshold sweep

The sweep evaluated all 33,968 unique observed probability cut points plus a threshold immediately above the maximum for the zero-output case. A cut at a unique score includes every cell tied at that score. This is exact for all distinct prediction sets reachable by a global `>=` threshold; no development examples were individually tuned and no per-label thresholds were used.

| Sweep range | Threshold | Precision | Recall | Micro-F1 | TP | FP | FN | Predicted cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Global best | 0.0027010208 | 0.002877 | 1.000000 | 0.005737 | 98 | 33970 | 0 | 34,068 |
| Best in [0.00001, 0.10] | 0.0027010208 | 0.002877 | 1.000000 | 0.005737 | 98 | 33970 | 0 | 34,068 |
| Best > 0.10 | 0.1024719626 | 0.000754 | 0.081633 | 0.001493 | 8 | 10609 | 90 | 10,617 |

At the global optimum, exact-set accuracy is **0.000528**; predicted labels per segment: mean 17.9778, median 18, distribution `{"0": 1, "12": 1, "14": 1, "16": 1, "18": 1890, "6": 1}`.

Representative thresholds show the trade-off:

| Threshold (`>=`) | Precision | Recall | Micro-F1 | TP / FP / FN | Mean predicted labels / segment |
|---:|---:|---:|---:|---:|---:|
| 0.00000 | 0.002873 | 1.000000 | 0.005730 | 98 / 34012 / 0 | 18.0000 |
| 0.00001 | 0.002873 | 1.000000 | 0.005730 | 98 / 34012 / 0 | 18.0000 |
| 0.00010 | 0.002873 | 1.000000 | 0.005730 | 98 / 34012 / 0 | 18.0000 |
| 0.00050 | 0.002873 | 1.000000 | 0.005730 | 98 / 34012 / 0 | 18.0000 |
| 0.00100 | 0.002873 | 1.000000 | 0.005730 | 98 / 34011 / 0 | 17.9995 |
| 0.00200 | 0.002875 | 1.000000 | 0.005733 | 98 / 33990 / 0 | 17.9884 |
| 0.00500 | 0.002823 | 0.979592 | 0.005629 | 96 / 33914 / 2 | 17.9472 |
| 0.01000 | 0.002777 | 0.959184 | 0.005537 | 94 / 33760 / 4 | 17.8649 |
| 0.02000 | 0.002610 | 0.887755 | 0.005204 | 87 / 33249 / 11 | 17.5916 |
| 0.03000 | 0.002502 | 0.826531 | 0.004990 | 81 / 32288 / 17 | 17.0813 |
| 0.04000 | 0.002370 | 0.744898 | 0.004726 | 73 / 30724 / 25 | 16.2517 |
| 0.05000 | 0.002160 | 0.622449 | 0.004304 | 61 / 28185 / 37 | 14.9055 |
| 0.07500 | 0.001347 | 0.265306 | 0.002680 | 26 / 19282 / 72 | 10.1889 |
| 0.10000 | 0.000705 | 0.081633 | 0.001397 | 8 / 11346 / 90 | 5.9916 |
| 0.15000 | 0.000313 | 0.010204 | 0.000607 | 1 / 3196 / 97 | 1.6871 |
| 0.20000 | 0.000000 | 0.000000 | 0.000000 | 0 / 141 / 98 | 0.0744 |
| 0.30000 | 0.000000 | 0.000000 | 0.000000 | 0 / 0 / 98 | 0.0000 |
| 0.50000 | 0.000000 | 0.000000 | 0.000000 | 0 / 0 / 98 | 0.0000 |
| 0.75000 | 0.000000 | 0.000000 | 0.000000 | 0 / 0 / 98 | 0.0000 |
| 0.90000 | 0.000000 | 0.000000 | 0.000000 | 0 / 0 / 98 | 0.0000 |

The exact best improves on the saved zero threshold only from F1 0.005730 to 0.005737. The very low threshold therefore does not reveal a useful operating point: it still emits 34,068 of 34,110 possible label cells, with 33,970 false positives and almost every segment receiving all 18 labels.

## Ranking metrics

Ranking metrics do not depend on a single threshold. ROC-AUC measures pairwise ranking of positive over negative cells; average precision summarizes the precision-recall ranking curve. The global positive-cell prevalence is 98 / 34,110 = 0.002873, a useful AP reference.

- Global ROC-AUC: **0.294861** (0.5 is chance ranking).
- Global average precision: **0.001843** (below the 0.002873 prevalence reference).
- Histogram overlap coefficient: **0.6998**.

Per-label scores are computed only when the development label has at least one positive and one negative; zero-support labels are marked N/A.

| Output label | Positive support | Negative support | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|
| AMENITY / negative | 1 | 1894 | 0.065470 | 0.000565 |
| AMENITY / neutral | 0 | 1895 | N/A | N/A |
| AMENITY / positive | 2 | 1893 | 0.374274 | 0.001534 |
| BRANDING / negative | 0 | 1895 | N/A | N/A |
| BRANDING / neutral | 0 | 1895 | N/A | N/A |
| BRANDING / positive | 0 | 1895 | N/A | N/A |
| EXPERIENCE / negative | 4 | 1891 | 0.546140 | 0.003287 |
| EXPERIENCE / neutral | 0 | 1895 | N/A | N/A |
| EXPERIENCE / positive | 30 | 1865 | 0.392779 | 0.013179 |
| FACILITY / negative | 3 | 1892 | 0.292988 | 0.001520 |
| FACILITY / neutral | 0 | 1895 | N/A | N/A |
| FACILITY / positive | 34 | 1861 | 0.299981 | 0.026546 |
| LOYALTY / negative | 0 | 1895 | N/A | N/A |
| LOYALTY / neutral | 0 | 1895 | N/A | N/A |
| LOYALTY / positive | 11 | 1884 | 0.341633 | 0.004444 |
| SERVICE / negative | 1 | 1894 | 0.387540 | 0.000861 |
| SERVICE / neutral | 0 | 1895 | N/A | N/A |
| SERVICE / positive | 12 | 1883 | 0.288458 | 0.004754 |

Only nine labels have positive dev support, and most have support below five. Their per-label ranking estimates are highly unstable; they are descriptive diagnostics, not robust class-level conclusions. The global ROC-AUC below 0.5 and AP below prevalence indicate poor aggregate ranking, not merely a bad threshold calibration.

## Training history and epoch selection

| Epoch | Train BCE | Dev BCE | Coarse-grid dev micro-F1 | Coarse-grid threshold |
|---:|---:|---:|---:|---:|
| 1 | 0.306170 | 0.101501 | 0.005730 | 0.00 |
| 2 | 0.064063 | 0.045235 | 0.005730 | 0.00 |
| 3 | 0.037192 | 0.033634 | 0.005730 | 0.00 |
| 4 | 0.029526 | 0.028563 | 0.005730 | 0.00 |
| 5 | 0.025617 | 0.025744 | 0.005730 | 0.00 |
| 6 | 0.023159 | 0.023918 | 0.005730 | 0.00 |
| 7 | 0.021630 | 0.022674 | 0.005730 | 0.00 |
| 8 | 0.020579 | 0.021756 | 0.005730 | 0.00 |
| 9 | 0.019781 | 0.021049 | 0.005730 | 0.00 |

BCE is a continuous probability loss over all 34,110 label cells. It can fall as logits/probabilities move toward the many negative targets even if the small positive subset is not ranked above negatives. At every logged epoch, the configured coarse grid selected threshold 0.00 with unchanged F1; that threshold predicts every cell positive for finite sigmoid outputs, so its score is fixed by the class prevalence. Thus declining BCE does not prove useful positive-class learning.

The saved history contains only the chosen threshold and aggregate metrics per epoch, not each epoch's full dev probability matrix or intermediate checkpoint. We recomputed the exact sweep for saved epoch 1 only. A finer sweep **could** change which epoch is best if epochs differ in score ranking near the low-probability region, but the retained logs are insufficient to determine whether it does; no other epoch checkpoint was available to audit without retraining.

## Decision and class weighting

The fine sweep finds a mathematically better threshold, but the gain is negligible and predictions remain nearly all-positive. Ranking is below chance globally, so lowering the threshold cannot recover useful separation. The frozen result should not be retained as evidence that this classifier is viable; it demonstrates a failed unweighted linear-head operating point. It does not by itself prove that fine-tuning XLM-R is inadequate.

Train-only positive weighting is a plausible controlled follow-up because the training targets are extremely sparse and unweighted BCE is dominated by negative cells. If approved, derive per-label positive weights only from train supports, cap weights, define zero-train-support handling (e.g., leave the label unweighted and disclose it, or exclude its loss term under an explicitly approved rule), and keep the dev threshold global. Weighting can improve positive ranking/recall but can also inflate false positives and damage calibration; evaluate using the same dev micro-F1 and ranking diagnostics. No weighted model was trained here.

## Integrity checks

- Checkpoint: epoch 1; checksum/config/artifacts were not modified by this diagnostic.
- Only official train/dev embeddings were accessed via the selective cache reader; development predictions used the dev rows. Test embedding values read: **false**.
- Only the official dev target dataset was instantiated for metric calculations; test labels/rows/predictions/metrics were not accessed.
- No training, threshold deployment, source-annotation edits, or per-label threshold tuning occurred.
- The temporary diagnostic script is not part of the project deliverables; this report is the only persistent output from the audit.
