# HAMOS V4 Hybrid Implementation and Readiness

## Status

V4 implementation, unit tests, and a real pinned-XLM-R GPU smoke test are complete. **Full training has not started.** No HAMOS test data was accessed. No prior model, checkpoint, config, annotation, split, or feature cache was modified. This is a new unique category–sentiment pair-presence formulation; its scores are not directly comparable to the original duplicate-sensitive structured multiset F1.

## Architecture

V4 predicts implicit pair presence by evaluating all 18 fixed category/sentiment candidates from one shared segment representation. Candidate category and sentiment embeddings enter the scorer and interact multiplicatively with the encoded segment representation, so the outputs are genuinely candidate-conditioned.

```mermaid
flowchart LR
    T[Original segment text] --> X[Pinned XLM-R-base encoder]
    X --> H[Segment representation h]
    C[Fixed category candidate] --> CE[Learned category embedding]
    S[Fixed sentiment candidate] --> SE[Learned sentiment embedding]
    H --> I[Candidate interactions and concatenation]
    CE --> I
    SE --> I
    I --> SC[Shared candidate scorer]
    SC --> A[any_pair logit]
    SC --> Q[implicit_given_pair logit]
    A --> P[Sigmoid probabilities]
    Q --> P
    P --> M[Product: P(any_pair) × P(implicit_given_pair)]
    M --> O[18 implicit pair-presence scores]
```

The scorer concatenates `h`, category and sentiment embeddings, `h × projected_category`, and `h × projected_sentiment`, then applies an MLP producing two logits per candidate. The implicit target score is the product of the two sigmoid probabilities; there is no hard gate. The model uses XLM-R `<s>` representation, revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, and no image or language features.

## Target construction and verified counts

Each segment produces 18 candidate rows in fixed order: `AMENITY`, `BRANDING`, `EXPERIENCE`, `FACILITY`, `LOYALTY`, `SERVICE`, each crossed with `negative`, `neutral`, `positive`.

- `any_pair = 1` when at least one quad of any aspect-implicitness status has the pair.
- `implicit_given_pair = 1` when at least one quad with the pair has an implicit aspect. For an explicit-only pair it is 0.
- The conditional loss mask is true only when `any_pair = 1`; absent pairs receive no conditional loss.
- If a pair occurs in both explicit and implicit quads, its conditional target is positive.
- Repeated occurrences collapse at `(segment_id, category, sentiment)` set-presence level. Source quads and records are not modified or collapsed.
- Model features contain only original segment text. IDs and hotel IDs remain metadata. Gold terms and opinions are not passed to the model.

Counts reconstructed from the official train/dev files matched the previously audited values exactly:

| Target rows | Train | Dev |
|---|---:|---:|
| Segments | 8,816 | 1,895 |
| Candidate rows | 158,688 | 34,110 |
| Any-pair positive | 11,663 | 2,501 |
| Any-pair negative | 147,025 | 31,609 |
| Conditional valid rows | 11,663 | 2,501 |
| Conditional implicit-positive | 451 | 98 |
| Conditional explicit-only negative | 11,212 | 2,403 |

The dataset constructor accepts only `train` and `dev`. Hotel-disjointness and segment-ID separation were verified. A train+dev text audit using casefolding and whitespace collapse while preserving punctuation and token content found 10,504 normalized groups, 115 repeated groups containing 322 records, and **zero** repeated groups with conflicting `(any_pair, implicit_given_pair)` targets. These rows remain separate; no text-group aggregation is used for training.

## Loss, optimization, and evaluation implementation

The baseline objective is unweighted BCE:

`L = mean(BCE(any_logits, any_targets)) + lambda × mean(BCE(conditional_logits, conditional_targets) over valid pairs)`

`conditional_loss_lambda` is configurable and currently set to `1.0`. A batch with no valid conditional pair returns a differentiable zero conditional loss. The conditional term is normalized by the count of valid candidate rows. No class weights, focal loss, or logit adjustment are used.

The training entry point uses AdamW, seed 1729, dynamic padding, FP16 autocast with GradScaler, gradient clipping, gradient accumulation, early stopping, and optional gradient checkpointing. The config starts with microbatch size 2 and accumulation 16. The model's checkpoint-selection metric is development unique-pair micro-F1. The V4 threshold selector performs an exact sweep over unique development scores, uses `score >= threshold`, and breaks metric ties by choosing the higher threshold. That threshold is saved with the checkpoint. The evaluator only loads the dev split.

Reported metrics include combined-score primary micro precision/recall/F1 and TP/FP/FN/TN; per-label support and metrics; exact-set accuracy; predicted-label cardinality; global ROC-AUC and average precision; all-zero segment counts; plus any-pair and masked conditional-head metrics. Unsupported labels are retained in output with zero support. Any/conditional head metrics use the selected primary threshold for descriptive diagnostics; their separate operating thresholds are not optimized.

## Tests

Command:

```powershell
.\.venv-hamos-cuda\Scripts\python.exe -m unittest discover -s tests -p 'test_v4_hybrid.py' -v
```

Result: **9 tests passed**. Coverage includes target dimensions and label ordering; explicit-only, implicit, absent, duplicate, and empty target cases; conditional masking and zero-valid batches; combined probabilities; gradients to encoder/category/sentiment embeddings; threshold selection and metrics; official train/dev counts; hotel/segment separation; input leakage safeguards; and rejection of a `test` split.

All V4 Python files also passed `py_compile` under the project's Python 3.12 environment.

## GPU smoke results

The smoke test loaded the exact local XLM-R checkpoint and ran two real training texts selected at the train token-length P90 (33 tokens). It computed both losses, backpropagated, and performed one confirmed optimizer update. Candidate output shape was `[2, 18]`.

| Measurement | Result |
|---|---:|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Reported device total | 6,441,926,656 bytes |
| Peak allocated | 5,586,323,456 bytes (about 5.59 GB) |
| Peak reserved | 5,737,807,872 bytes (about 5.74 GB) |
| Total smoke duration | 2.466 seconds |
| Final finite loss | 1.05141 (`any=0.64193`, `conditional=0.40948`) |
| Gradient flow | Encoder, category embeddings, and sentiment embeddings all nonzero |
| Optimizer update | Confirmed |

The initial GradScaler scale 65,536 produced non-finite gradients; scale 32,768 also overflowed; scale 16,384 produced finite gradients and the successful optimizer update. GradScaler therefore handled the observed instability by backing off. The smoke artifact records all three attempts and is saved at `runs/structured/challenger_v4/hybrid_smoke.json`.

**Memory is the main readiness risk:** peak allocated memory was about 87% of the reported physical total and reserved memory about 89%, leaving limited headroom for allocator fragmentation, other GPU processes, or longer batches. The smoke used batch 2 with P90-length sequences, not maximum-length samples. The maximum train sequence is longer than P90, so OOM remains possible. Gradient checkpointing was enabled as configured.

## Runtime estimate

The measured 2.466 seconds includes three forward/backward attempts because the scaler backed off twice before a successful step. Extrapolating that full observed cost to 4,408 two-example microbatches gives about **10,872 seconds (3.02 hours) for train batches in one epoch**, excluding dev evaluation and data-loading overhead. Dividing evenly across the three attempts would imply roughly one hour per train epoch after scale stabilization, but that is only an inference from one smoke step—not a measured steady-state throughput benchmark. A practical estimate is therefore **about 1–3 hours per epoch**, plus development evaluation. Five to eight epochs would be roughly **5–24 hours**; the 20-epoch cap could take **20–60 hours** before dev overhead.

This estimate is too uncertain to promise a deadline, and the conservative range is incompatible with a six-hour remaining window. A timed partial training run would be needed before making a stronger estimate, but that would already be training and is outside this approved readiness stage. No full training was launched.

## Readiness and remaining risks

The implementation is functionally ready for review and a future approved training run, but execution readiness is **cautious rather than comfortable**: the smoke fit in 6 GB, yet memory headroom is small and FP16 required scale reduction. The next run should preserve GradScaler and record skipped optimizer steps. If OOM occurs, stop and report it; do not silently change batch size, accumulation, or architecture.

Scientific risks remain: only 451 train and 98 dev unique implicit pair positives; four labels have no train positive support and nine have no dev positives; language transfer is untested; the auxiliary any-pair objective may not improve implicitness; and duplicate quad multiplicity is intentionally outside this V4 target.

## Exact command for a future approved training run

This command has **not** been run:

```powershell
& '.\.venv-hamos-cuda\Scripts\python.exe' -m src.v4_hybrid_train `
  --root . `
  --config configs/implicit_hybrid_v4.json `
  --output-dir runs/structured/challenger_v4/hybrid
```

The trainer refuses to overwrite an existing output directory. Full training requires your explicit approval after reviewing this readiness report.

## Added files

- `configs/implicit_hybrid_v4.json`
- `src/v4_hybrid_dataset.py`
- `src/v4_hybrid_model.py`
- `src/v4_hybrid_loss.py`
- `src/v4_hybrid_metrics.py`
- `src/v4_hybrid_train.py`
- `src/v4_hybrid_evaluate.py`
- `src/v4_hybrid_smoke.py`
- `tests/test_v4_hybrid.py`
- `runs/structured/challenger_v4/hybrid_smoke.json`
- `reports/v4_hybrid_implementation_and_readiness.md`

All prior experiment artifacts remain untouched. Only the new V4 implementation, V4 regression test, isolated smoke result, and this readiness report were added.
