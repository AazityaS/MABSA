# HAMOS V4 Training Progress Through Epoch 5

## Outcome

The controlled V4 hybrid continuation completed exactly five total epochs: the already authorized epoch 1 plus epochs 2–5. The trainer stopped after epoch 5 development evaluation. No epoch 6 was started, and no test split was loaded. Epoch 5 is the best development checkpoint, with unique-pair implicit micro-F1 **0.5933** at the development-selected global threshold **0.14856**.

The final checkpoint is `runs/structured/challenger_v4/hybrid/best_checkpoint.pt`; the per-epoch checkpoints are `epoch_001_checkpoint.pt` through `epoch_005_checkpoint.pt` in the same run directory.

## Resume method and reproducibility caveat

The epoch-1 checkpoint stored model weights and development results, but not optimizer moments, scheduler state, scaler state, or RNG/sampler state. Its model weights were strictly restored before continuation and the checkpoint was preserved as `epoch_001_checkpoint.pt`. A post-run SHA-256 check confirmed the preserved file matches the original epoch-1 checkpoint hash (`5c20dcd137d2a2632c484f63201fc966c3d26f3761a5661f382e5a9918b57491`).

This was therefore **not a seamless optimizer continuation**. AdamW was restarted. The configured scheduler was rebased at the epoch-1 count of 273 successful updates, within its configured 5,520-update horizon; the continuation began at approximately `9.8913e-6` learning rate. GradScaler was reinitialized at 8192. Seed 1729 was retained; because the prior RNG/sampler state was unavailable, continuation shuffling used deterministic epoch seed 1730. These choices and their limitation are recorded in the run's continuation metadata.

## Epoch results

All prediction metrics below use the unique category–sentiment pair endpoint on the 1,895-segment development split. Thresholds were selected on development data only using the corrected global sweep, `probability >= threshold`, with the highest threshold winning ties. ROC-AUC and average precision are ranking metrics; precision, recall, F1, exact-set accuracy, and cardinality are thresholded metrics. Development support was 98 positive implicit pairs, so small changes in correct/missed pairs have a noticeable effect on F1.

| Epoch | Train / any / conditional loss | Dev loss | Threshold | P / R / F1 (TP / FP / FN) | ROC-AUC / AP | Exact set accuracy | Predicted labels: mean / median; count distribution | Train / dev eval duration | GradScaler start → end; updates successful / skipped |
|---:|---:|---:|---:|---|---:|---:|---|---:|---|
| 1 | 0.50082 / 0.32024 / 0.18058 | 0.28081 | 0.01530 | 0.3200 / 0.3265 / 0.3232 (32 / 68 / 66) | 0.9645 / 0.2519 | 0.9499 | 0.0528 / 0; `{0:1839, 1:30, 2:19, 3:1, 4:3, 5:1, 6:2}` | 996.49 s / 35.56 s | 65536 → 8192; 273 / 3 |
| 2 | 0.23027 / 0.15355 / 0.07672 | 0.16658 | 0.17432 | 0.4828 / 0.5714 / 0.5234 (56 / 60 / 42) | 0.9831 / 0.4359 | 0.9562 | 0.0612 / 0; `{0:1825, 1:32, 2:31, 3:6, 4:1}` | 699.78 s / 29.93 s | 8192 → 4096; 275 / 1 |
| 3 | 0.16570 / 0.10395 / 0.06175 | 0.13944 | 0.35889 | 0.6923 / 0.4592 / 0.5521 (45 / 20 / 53) | 0.9901 / 0.4685 | 0.9646 | 0.0343 / 0; `{0:1848, 1:29, 2:18}` | 813.35 s / 21.87 s | 4096 → 4096; 276 / 0 |
| 4 | 0.12716 / 0.08483 / 0.04233 | 0.12822 | 0.36230 | 0.6341 / 0.5306 / 0.5778 (52 / 30 / 46) | 0.9883 / 0.4895 | 0.9652 | 0.0433 / 0; `{0:1843, 1:26, 2:23, 3:2, 4:1}` | 709.52 s / 24.72 s | 4096 → 4096; 276 / 0 |
| 5 | 0.10165 / 0.07063 / 0.03102 | 0.12883 | 0.14856 | 0.5586 / 0.6327 / 0.5933 (62 / 49 / 36) | 0.9747 / 0.5114 | 0.9652 | 0.0586 / 0; `{0:1842, 1:15, 2:20, 3:16, 4:2}` | 786.04 s / 15.85 s | 4096 → 4096; 276 / 0 |

The cardinality distribution is the number of emitted implicit category–sentiment labels per segment. Epochs 1–5 emit zero labels for 1,839, 1,825, 1,848, 1,843, and 1,842 development segments respectively; the median is zero at every epoch because most development segments have no gold implicit pair.

The development losses decrease through epoch 4 and rise slightly at epoch 5. Thresholded primary F1 rises across the continuation and is highest at epoch 5; epoch 5 also has the strongest recall and average precision, while epoch 4 has higher precision and slightly higher exact-set accuracy. The checkpoint selection rule uses development primary micro-F1, so epoch 5 is selected.

## Runtime, GPU, and numerical stability

- Active training time across the five epochs: **4,005.18 seconds** (66 minutes 45 seconds).
- Development evaluation time total: **127.92 seconds** (2 minutes 8 seconds).
- Recorded process elapsed time: **4,178.99 seconds** (69 minutes 39 seconds); wall-clock UTC span from first training start to final completion was 04:04:57–06:12:46 on 2026-11-19, including the pause between the initial epoch and continuation.
- Average training microbatch time by epoch: 0.2261, 0.1588, 0.1845, 0.1610, and 0.1783 seconds.
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, approximately 6 GiB.
- Peak allocated VRAM: **5,586,321,920 bytes** (about 5.20 GiB); peak reserved VRAM: **5,809,111,040 bytes** (about 5.41 GiB).
- Optimizer update attempts: **1,380**; successful updates: **1,376**; skipped updates: **4**.
- The four GradScaler skips occurred in epochs 1 (3) and 2 (1); scale decreased from 65,536 to 8,192 in epoch 1 and to 4,096 in epoch 2. Epochs 3–5 had no skipped updates and the scale remained 4,096.
- No non-finite loss was recorded, no CUDA OOM occurred, and no sustained development collapse was observed.

## Validation and stopping condition

Post-run checks confirmed metric rows for epochs `[1, 2, 3, 4, 5]`, five corresponding per-epoch checkpoints, finite numeric metric values, `best_checkpoint.pt` at epoch 5 with threshold 0.1485595703125, a preserved epoch-1 checkpoint whose SHA-256 matches the original, no epoch-6 checkpoint, and run metadata `test_split_loaded=false`.

**Final state:** exactly five epochs completed; best development checkpoint is `runs/structured/challenger_v4/hybrid/best_checkpoint.pt` (epoch 5, F1 0.5933); no epoch 6 was launched; no test data was accessed. Further training or final test evaluation requires separate approval.
