# HAMOS V4 — Epoch 1 Controlled Training Diagnostic

## Run status and scope

Exactly one full training epoch completed, followed by development evaluation. **No second epoch was started. No HAMOS test split, test annotations, predictions, or metrics were loaded or examined.** Train/dev hotel disjointness and the audited target counts were checked by the V4 trainer before training.

The run used `configs/implicit_hybrid_v4.json`, the isolated output `runs/structured/challenger_v4/hybrid/`, microbatch 2, gradient accumulation 16, FP16 autocast with GradScaler, gradient checkpointing, unweighted losses, and the configured 20-epoch scheduler horizon. The only run limit was `--epoch-limit 1`, which stopped after the first epoch while retaining the configured epoch-1 learning-rate schedule. No architecture, loss, target, or optimizer settings were changed.

## Training and timing

| Measure | Result |
|---|---:|
| Training start (UTC) | 2026-11-19 04:04:57.495 |
| Training end (UTC) | 2026-11-19 04:21:33.983 |
| Full train epoch duration | 996.49 seconds (16 min 36.49 sec) |
| Training microbatches | 4,408 |
| Average training microbatch duration | 0.2261 sec |
| Development evaluation duration | 35.56 sec |
| Process time through dev evaluation and run metadata | 1,034.04 sec (17 min 14.04 sec) |
| Train total loss | 0.500819 |
| Train any-pair loss | 0.320238 |
| Train conditional loss | 0.180581 |
| Development loss | 0.280812 |

The development evaluation was performed by the trainer after the epoch. A later development-only checkpoint recheck was used to verify ranking metrics and correct a threshold-sweep bookkeeping defect described below; it did not run training again.

## GPU and numerical stability

GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU; reported total 6,441,926,656 bytes (6,144 MiB).

| Point | Free bytes (CUDA) | Allocated | Reserved |
|---|---:|---:|---:|
| Before epoch training, after model/optimizer setup | 4,233,101,312 | 1,115,020,800 | 1,170,210,816 |
| Peak during run | — | 5,586,321,920 | 5,809,111,040 |
| After dev evaluation, before process exit | 0 reported free | 3,359,461,376 | 5,809,111,040 |

After the process exited, `nvidia-smi` reported 532 MiB used and 5,470 MiB free. The in-process zero-free reading after evaluation is recorded as observed; it differed from the post-process device reading, while the peak allocator counters remained below the physical device total. Peak reserved memory was close to the limit, so longer inputs or concurrent GPU use could still cause OOM. **No CUDA OOM occurred.**

GradScaler detected three non-finite-gradient optimizer attempts and skipped them:

| Optimizer attempt | Scale before | Scale after | Result |
|---:|---:|---:|---|
| 44 | 65,536 | 32,768 | Skipped on overflow |
| 85 | 32,768 | 16,384 | Skipped on overflow |
| 149 | 16,384 | 8,192 | Skipped on overflow |

There were 276 optimizer-step attempts, 3 skips, and 273 successful updates. Loss values remained finite; no non-finite loss was encountered. The final scaler scale was 8,192. The scheduler advances only after a successful optimizer update.

## Threshold-selection correction

After training, a re-evaluation exposed a defect in the exact development threshold sweep: for tied probabilities, the sweep associated the F1/counts at the end of a tie group with the first score in the sorted array, instead of that group's score. Training metrics therefore showed a selected operating point whose saved threshold did not reproduce its reported counts.

The sweep bookkeeping was corrected and a regression test added. The saved epoch-1 model weights were not changed. Using the saved checkpoint's FP16 development predictions and `probability >= threshold`, the corrected global development threshold is **0.0153045654296875**. The original run metadata threshold was 0.01580810546875; only this threshold metadata and its development metric record were refreshed. The corrected value and reason are recorded in `runs/structured/challenger_v4/hybrid/threshold_correction.json`. The checkpoint now stores the corrected development threshold and the verified development metrics.

The corrected sweep selects the highest threshold among tied best-F1 operating points. Re-evaluation confirmed that its threshold, TP, FP, FN, and F1 agree with thresholded metrics. The corrected threshold is development-only and was not chosen from test information.

## Primary development results

The unit is a unique implicit category–sentiment pair per segment. There are 1,895 development segments and 98 gold positive segment-label pairs. At the corrected global threshold:

| Metric | Result |
|---|---:|
| Threshold (`>=`) | 0.0153045654 |
| Micro precision | 0.320000 |
| Micro recall | 0.326531 |
| **Unique-pair micro-F1** | **0.323232** |
| TP / FP / FN / TN | 32 / 68 / 66 / 33,944 |
| Global ROC-AUC | 0.964538 |
| Global average precision | 0.251870 |
| Exact-set accuracy | 0.949868 (1,800 / 1,895) |
| Mean predicted labels per segment | 0.05277 |
| Median predicted labels per segment | 0 |
| Cardinality MAE / signed bias | 0.05699 / +0.00106 |
| Gold all-zero segments | 1,818 |
| Predicted all-zero segments | 1,839 |
| Both gold and prediction all-zero | 1,796 |

Predicted-label count distribution: `0: 1,839; 1: 30; 2: 19; 3: 1; 4: 3; 5: 1; 6: 2`.

The exact-set accuracy is largely driven by the high number of all-zero segments and should not be read as strong overall performance. ROC-AUC and average precision are ranking metrics and do not depend on the selected threshold; precision/recall/F1, confusion counts, exact-set accuracy, and cardinality are thresholded metrics.

### Per-label results

Support is the number of positive development segment-label pairs. Nine labels have zero development support, so those per-label F1 values are not evidence of learned recall for those labels.

| Category / sentiment | Support | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| AMENITY / negative | 1 | 0 / 0 / 1 | 0.0000 | 0.0000 | 0.0000 |
| AMENITY / neutral | 0 | 0 / 0 / 0 | 0.0000 | 0.0000 | 0.0000 |
| AMENITY / positive | 2 | 0 / 2 / 2 | 0.0000 | 0.0000 | 0.0000 |
| BRANDING / negative | 0 | 0 / 0 / 0 | 0.0000 | 0.0000 | 0.0000 |
| BRANDING / neutral | 0 | 0 / 0 / 0 | 0.0000 | 0.0000 | 0.0000 |
| BRANDING / positive | 0 | 0 / 3 / 0 | 0.0000 | 0.0000 | 0.0000 |
| EXPERIENCE / negative | 4 | 0 / 0 / 4 | 0.0000 | 0.0000 | 0.0000 |
| EXPERIENCE / neutral | 0 | 0 / 0 / 0 | 0.0000 | 0.0000 | 0.0000 |
| EXPERIENCE / positive | 30 | 12 / 38 / 18 | 0.2400 | 0.4000 | 0.3000 |
| FACILITY / negative | 3 | 0 / 0 / 3 | 0.0000 | 0.0000 | 0.0000 |
| FACILITY / neutral | 0 | 0 / 0 / 0 | 0.0000 | 0.0000 | 0.0000 |
| FACILITY / positive | 34 | 19 / 13 / 15 | 0.5938 | 0.5588 | 0.5758 |
| LOYALTY / negative | 0 | 0 / 0 / 0 | 0.0000 | 0.0000 | 0.0000 |
| LOYALTY / neutral | 0 | 0 / 0 / 0 | 0.0000 | 0.0000 | 0.0000 |
| LOYALTY / positive | 11 | 0 / 6 / 11 | 0.0000 | 0.0000 | 0.0000 |
| SERVICE / negative | 1 | 0 / 0 / 1 | 0.0000 | 0.0000 | 0.0000 |
| SERVICE / neutral | 0 | 0 / 0 / 0 | 0.0000 | 0.0000 | 0.0000 |
| SERVICE / positive | 12 | 1 / 6 / 11 | 0.1429 | 0.0833 | 0.1053 |

### Auxiliary heads

The same selected combined-score threshold is used here as a descriptive diagnostic; it was not separately tuned for either head.

| Head / evaluation denominator | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---:|---:|---:|---:|
| `any_pair`, all 34,110 segment-candidate rows | 0.13346 | 0.97241 | 0.23470 | 2,432 / 15,791 / 69 / 15,818 |
| `implicit_given_pair`, 2,501 valid present-pair rows only | 0.20888 | 0.81633 | 0.33264 | 80 / 303 / 18 / 2,100 |

The conditional head's macro-F1 over the fixed 18 labels is 0.13637; it is descriptive and includes labels with no valid rows or no positive support. The primary endpoint remains the combined score evaluated over all segment-label pairs.

## Comparison with V3-Frozen on the same development target

Both results use the 18-label unique implicit category–sentiment presence target and the same 1,895-segment development split. V3-Frozen used cached frozen XLM-R embeddings and an unweighted linear head; V4 fine-tuned XLM-R and adds the auxiliary any-pair objective. The comparison therefore does not isolate candidate conditioning alone.

| Dev metric | V3-Frozen | V4 epoch 1 |
|---|---:|---:|
| Unique-pair micro-F1 | 0.005730 | 0.323232 |
| Precision | 0.002873 | 0.320000 |
| Recall | 1.000000 | 0.326531 |
| ROC-AUC | 0.294861 | 0.964538 |
| Average precision | 0.001843 | 0.251870 |
| Exact-set accuracy | 0.000000 | 0.949868 |
| Threshold | 0.00 | 0.0153046 |

V3-Frozen predicted all 18 labels for every dev segment at its selected threshold. V4's first epoch predicts zero labels for most segments and shows much stronger development ranking. The high V4 exact-set accuracy mostly reflects empty gold sets. V4's result is promising, but it is a single epoch selected and measured on the same small development set; it does not establish generalization or final model quality.

## Learning assessment and next-epoch estimate

The model appears to be learning a useful development ranking signal after one epoch: primary F1 is 0.3232, global ROC-AUC 0.9645, and AP 0.2519, compared with V3-Frozen's near-degenerate operating point. The any-pair head has very high recall but many false positives; the conditional head improves precision among present pairs. Only 98 development positives support the primary result, and nine target labels have no positive dev support, so this is preliminary.

At observed throughput, another epoch would be approximately 996 seconds of training plus about 36 seconds of development evaluation, or roughly **17 minutes** under similar device/input conditions. This estimate is for runtime only. The completed process saved a model checkpoint but not optimizer/scheduler state, so an exact seamless continuation would require explicit resume support. No second epoch was started.

## Artifacts and final checks

- Checkpoint: `runs/structured/challenger_v4/hybrid/best_checkpoint.pt` (epoch 1; corrected dev threshold stored).
- Authoritative verified dev result: `runs/structured/challenger_v4/hybrid/epoch_001_dev_evaluation_final.json`.
- Epoch log and `epoch_001_dev_metrics.json` were synchronized to the corrected dev threshold and metrics.
- `threshold_correction.json` records the initial and corrected threshold and confirms weights were unchanged.
- Intermediate FP32/FP16 rechecks are retained with `diagnostic`, `pre_sweep_fix`, or `pre_checkpoint_threshold_fix` in their filenames; they are not the authoritative result.
- Regression suite: **11 V4 tests passed** after adding a tied-score threshold-selection regression.

**Final confirmation:** exactly one epoch completed; no second epoch started; no test data accessed; epoch-1 checkpoint saved successfully. No OOM occurred. Three optimizer steps were skipped by GradScaler following non-finite gradients. Continuing appears scientifically worth considering based on development ranking, but requires approval and should be interpreted cautiously because the development positive sample is small.
