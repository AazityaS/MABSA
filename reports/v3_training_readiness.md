# Challenger V3 training readiness

**Status:** implementation and smoke checks are complete. Full training has **not** started. No HAMOS test split examples, labels, predictions, or metrics were loaded or evaluated.

## Implemented files

- `configs/implicit_presence_v3.json` — isolated V3 config; uses pinned XLM-R revision, unweighted BCE, seed 1729, micro-batch 2, accumulation 16, max 20 epochs, patience 4, FP16, and activation checkpointing.
- `src/v3_presence_dataset.py` — train/dev-only dataset, fixed 18-label order, multi-hot construction from aspect-implicit quads only, duplicate-collapse counters, language metadata join, target validation, and hotel-disjoint assertion.
- `src/v3_presence_model.py` — fine-tuned XLM-R encoder and 18-logit dropout/linear head.
- `src/v3_presence_metrics.py` — global threshold sweep, unique-pair micro metrics, per-label support/PRF, fixed 18-label macro-F1 with unsupported-class warning, exact-set accuracy, and predicted label-count/cardinality diagnostics.
- `src/v3_presence_train.py` — CUDA-only trainer, unweighted baseline, optional separate train-support-based class weighting, dev threshold/checkpoint selection, early stopping, and saved config/log/support/metadata.
- `src/v3_presence_evaluate.py` — train/dev-only evaluator. Its CLI choices intentionally exclude test.
- `src/v3_presence_smoke.py` — small train-only pretrained GPU smoke test.
- `tests/test_v3_presence.py` — target, duplicate, all-zero, order, split, weighting, metric, and metadata-boundary tests.
- `reports/v3_implementation_specification.md` — approved task, implementation plan, and training gate.
- Four new smoke result JSON files under `runs/structured/challenger_v3/`; no prior artifact was overwritten.

Existing structured modules, configs, checkpoints, caches, annotations, split files, and historical results were not modified.

## Data checks

The V3 dataset reads only the official train and dev split rows. Train and dev contain 8,816 and 1,895 segment records, respectively. Actual dataset validation confirmed the target shape is 18, all values are binary, 370 train and 77 dev segments have at least one positive label, positive label-cell totals are 451 and 98, and train/dev hotel sets are disjoint. Therefore 8,446 train and 1,818 dev examples have all-zero targets. Language metadata was joined for train/dev IDs only: train has 7,182 English, 1,378 Vietnamese, and 256 other records; dev has 1,542 English, 298 Vietnamese, and 55 other records. The metadata lookup decodes full rows only for requested IDs; language and identifiers remain outside model tensors.

The implementation specification and dataset behavior retain these audit findings:

- 37 normalized text groups are shared across train/dev, with no conflicting target vectors.
- 96 train and 21 dev implicit quad occurrences collapse into repeated-pair presence labels.
- Four labels have zero development support: BRANDING / positive, BRANDING / neutral, LOYALTY / neutral, and SERVICE / neutral.
- Language stays in metadata and is not fed to this text-only model. It does not resolve target ambiguity in these records.

## Environment and checkpoint loading

| Item | Observed value |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA build | 12.8 |
| Transformers | 5.23.0 |
| CUDA available | Yes |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| NVIDIA driver | 591.86 |
| `nvidia-smi` total memory | 6,144 MiB |
| `nvidia-smi` free after smoke | 5,391 MiB |
| PyTorch free memory before model load | 5,403,312,128 bytes |
| PyTorch total memory | 6,441,926,656 bytes |

`AutoTokenizer` and `AutoModel` loaded `FacebookAI/xlm-roberta-base` at revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` from the local cache. No packages or model files were installed/downloaded during this implementation. Transformers reported the checkpoint’s `lm_head` keys as unexpected when loading the encoder-only `XLMRobertaModel`; this is expected because the pretraining head is not part of the encoder classifier. A Windows distributed-redirect note was informational.

## GPU smoke test

The smoke used two actual train-split texts selected around the middle of the train text-length distribution. Tokenization produced shape `[2, 12]`; targets and logits had shape `[2, 18]`. It used FP16 autocast and `GradScaler`, then performed optimizer updates on both the classifier head and XLM-R encoder.

| Check | Result |
|---|---|
| Exact pinned XLM-R/tokenizer loaded | Pass |
| Finite logits and BCE loss | Pass |
| Finite gradients | Pass (199 gradient tensors) |
| Backward pass and optimizer update | Pass |
| Encoder changed | Yes; max absolute parameter update `6.03e-05` |
| Head changed | Yes; max absolute parameter update `6.01e-05` |
| Three warmed batch-2 step times | 1.602 s, 1.166 s, 1.282 s |
| Mean of warmed steps (steps 2–3) | 1.224 s per batch of two |
| Activation checkpointing | Enabled through Transformers `activation_checkpointing_enable` |
| Test split loaded | No |

### Memory counter discrepancy

The smoke completed without an out-of-memory error, but PyTorch reported peak allocated memory of **6,765,336,064 bytes** and peak reserved memory of **7,832,862,720 bytes**, both above its reported 6,441,926,656-byte total device memory. This repeated in the uncheckpointed smoke and in the checkpointed smoke. The later `nvidia-smi` snapshot showed 5,391 MiB free after the process exited. These figures do not agree; this audit cannot determine whether Windows WDDM/shared-memory behavior explains the allocator counters. Checkpointing did not materially lower the reported peak on the short-sequence smoke. Treat this as an unresolved GPU memory-accounting and full-run risk; do not infer that max-length training batches are safe from this short smoke.

The pinned checkpoint also emits a `use_cache=True`/activation-checkpointing compatibility warning; Transformers automatically switches cache off. This did not prevent forward/backward or updates.

## Tests

The environment does not include `pytest`; the attempted `python -m pytest` command reported `No module named pytest`. The repository tests are `unittest` based and were run with the project Python executable:

- V3 tests: 13 passed.
- Existing label-mapping tests: 2 passed.
- Existing structured protocol tests: 16 passed.
- Existing cardinality-loss tests: 3 passed.
- Existing feature-cache path tests: 2 passed.
- Python compilation of all V3 modules and tests: passed.
- **Total: 36 tests passed.**

`tests/test_structured_dataset.py` was not run because its class setup opens train, dev, **and test** datasets. Running it would violate the no-test-access restriction. The V3 integration test independently instantiated only train/dev datasets and verified their actual counts, targets, and hotel separation.

The V3 evaluation CLI accepts only `train` or `dev`; test is excluded by both argparse and the evaluator function. Training/evaluation command help was checked without running either full workflow.

## Training-duration estimate and decision

Using the observed warmed throughput of 1.224 seconds for two very short (12-token padded) examples, 8,816 train segments at micro-batch 2 imply approximately 4,408 train batches per epoch, or about **90 minutes of train batches per epoch**. A dev pass of 1,895 records is approximately another 19 minutes at that same short-sequence rate. A 20-epoch run would therefore take about **36 hours** before checkpointing and other overhead if it reached the cap. If early stopping ends around epochs 5–8, the same extrapolation is roughly **9–15 hours**. Longer batches can increase these estimates; the smoke sample is not a max-length benchmark.

**Do not start full training yet.** The code path, tests, and mathematical GPU update passed, but the allocator peak exceeds the reported physical device total and the smoke did not test long sequences or a complete accumulation window. The current config enables checkpointing and remains unweighted. Review this report and explicitly approve full training only after accepting the memory-accounting risk or requesting a further train-only memory diagnostic. No full training, full feature extraction, or test evaluation has been performed.

After explicit approval, the planned command is:

```powershell
.\.venv-hamos-cuda\Scripts\python.exe -m src.v3_presence_train --root . --config configs/implicit_presence_v3.json --output-dir runs/structured/challenger_v3/text --device cuda
```
