# HAMOS Primary Training Results

**Run date:** 2026-11-19 (Asia/Kolkata, UTC+05:30)  
**Status:** Both approved primary training runs completed successfully. No test-set inference/evaluation or image-only training was run.

## Pre-training checks

- Python was invoked explicitly through `.venv-hamos-cuda\Scripts\python.exe`.
- CUDA was available (`torch 2.11.0+cu128`, CUDA 12.8) on the NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- Cache `cache/structured_features/9072ce0b71cd3470/` existed. Its cache key and XLM-R/CLIP checkpoint IDs and revisions matched `configs/structured_baseline.json`; feature dimensions were 768 and 512 and the manifest device was CUDA.
- `runs/structured` did not exist before training, so neither run could overwrite an earlier checkpoint.
- Text-only and multimodal eligibility IDs were compared before the runs:

| Split | Text-only segments | Multimodal segments | Identical IDs |
|---|---:|---:|---|
| Train | 8,813 | 8,813 | Yes |
| Dev | 1,895 | 1,895 | Yes |

- The regression suite passed **21/21 tests** before training, including the direct keyed-cache path regression test.

## Run summary

| Mode | Start | End | Duration | Epochs | Best epoch | Best dev primary F1 | Selected objectness threshold | Stop reason | Peak allocated GPU memory |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| Text-only | 2026-11-19 00:50:51 +05:30 | 2026-11-19 01:02:40 +05:30 | 708.619 s (11m 48.6s) | 8 | 4 | 0.114478 | 0.70 | Patience 4 | 24.42 MiB |
| Multimodal | 2026-11-19 01:03:05 +05:30 | 2026-11-19 01:14:48 +05:30 | 702.672 s (11m 42.7s) | 7 | 3 | 0.132956 | 0.65 | Patience 4 | 27.73 MiB |

The trainer used the unchanged 20-epoch maximum, four-epoch early-stopping patience, saved seed/configuration, and development primary implicit category–sentiment micro-F1 for checkpoint selection. Threshold selection remained development-only. The listed GPU values are PyTorch peak allocated memory per epoch, not total device use; `nvidia-smi` showed approximately 1.4–1.7 GiB in use during the runs.

## Text-only epoch log

| Epoch | Training loss | Development loss | Dev primary implicit category–sentiment micro-F1 | Dev threshold | Epoch duration (s) | Peak allocated GPU memory (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.134992 | 0.522809 | 0.075949 | 0.80 | 91.733 | 24.42 |
| 2 | 0.486880 | 0.474281 | 0.104575 | 0.85 | 86.781 | 24.42 |
| 3 | 0.452561 | 0.458250 | 0.088799 | 0.80 | 89.942 | 24.42 |
| 4 | 0.427519 | 0.420483 | **0.114478** | 0.70 | 88.593 | 24.42 |
| 5 | 0.401620 | 0.418399 | 0.092345 | 0.75 | 82.692 | 24.42 |
| 6 | 0.393077 | 0.398147 | 0.060294 | 0.75 | 83.413 | 24.42 |
| 7 | 0.381258 | 0.385827 | 0.081862 | 0.70 | 86.481 | 24.42 |
| 8 | 0.374610 | 0.379783 | 0.079638 | 0.70 | 84.834 | 24.42 |

**Best checkpoint:** [text/best_checkpoint.pt](../runs/structured/text/best_checkpoint.pt), epoch 4. Training stopped after epochs 5–8 failed to improve the best development primary F1. The checkpoint stores the selected threshold 0.70.

## Multimodal epoch log

| Epoch | Training loss | Development loss | Dev primary implicit category–sentiment micro-F1 | Dev threshold | Epoch duration (s) | Peak allocated GPU memory (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.261163 | 0.569003 | 0.031250 | 0.95 | 94.338 | 27.73 |
| 2 | 0.507306 | 0.480761 | 0.113250 | 0.80 | 92.904 | 27.73 |
| 3 | 0.449499 | 0.445925 | **0.132956** | 0.65 | 98.289 | 27.73 |
| 4 | 0.422860 | 0.473739 | 0.098098 | 0.75 | 106.878 | 27.73 |
| 5 | 0.402560 | 0.468662 | 0.124208 | 0.70 | 102.477 | 27.73 |
| 6 | 0.382985 | 0.424140 | 0.085969 | 0.70 | 95.842 | 27.73 |
| 7 | 0.376458 | 0.419270 | 0.086811 | 0.65 | 98.696 | 27.73 |

**Best checkpoint:** [multimodal/best_checkpoint.pt](../runs/structured/multimodal/best_checkpoint.pt), epoch 3. Training stopped after epochs 4–7 failed to improve the best development primary F1. The checkpoint stores the selected threshold 0.65.

## Logs, warnings, and errors

Both processes exited with code 0. Each produced one JSONL record per completed epoch and a `run_summary.json`; summaries report `warnings: []` and `errors: []`. Checkpoints were verified to exist, use the corresponding mode, point to cache key `9072ce0b71cd3470`, and record the expected best epoch and development-selected threshold.

To capture the requested development loss, epoch duration, and per-epoch CUDA peak, `src/structured_train.py` was extended with reporting-only instrumentation. The loss function, target schema, threshold grid, checkpoint selection metric, optimizer/hyperparameters, early-stopping rule, data eligibility, and evaluation definitions were not changed. The direct cache-directory path supplied for these runs also exposed that the trainer previously always appended the cache key; `src/structured_features.py` and the trainer now accept either a cache root or its exact keyed directory. Tests cover both forms.

The first regression run after adding the cache-path tests hit a sandbox temp-directory permission error. Test fixtures were moved into the project `tmp` directory; the full suite then passed **21/21**. This did not affect either training run.

## Scope and next step

The two primary runs used the same cache, config, target schema, eligible examples, loss, and development evaluation protocol, and they ran sequentially. No image-only training was run. No test-set predictions or metrics were generated. Both best checkpoints are ready for review; test evaluation remains a separate next step and has not been started.
