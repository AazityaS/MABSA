# Challenger V3 implementation specification

**Approval:** The V3 task and architecture are approved as stated in the user instruction. This specification is written before implementation. It does not authorize full training.

## Task and output

V3 predicts the **presence** of each implicit broad category–sentiment pair in one segment. A label is positive when at least one gold quad in that segment has `aspect_implicit == true` and the corresponding broad category and sentiment. The output is an 18-dimensional multi-hot vector in this fixed order:

1. AMENITY / negative
2. AMENITY / neutral
3. AMENITY / positive
4. BRANDING / negative
5. BRANDING / neutral
6. BRANDING / positive
7. EXPERIENCE / negative
8. EXPERIENCE / neutral
9. EXPERIENCE / positive
10. FACILITY / negative
11. FACILITY / neutral
12. FACILITY / positive
13. LOYALTY / negative
14. LOYALTY / neutral
15. LOYALTY / positive
16. SERVICE / negative
17. SERVICE / neutral
18. SERVICE / positive

Explicit quads do not set a V3 label. Segments without aspect-implicit quads receive an all-zero vector. Repeated identical category–sentiment pairs within a segment collapse to one positive label. This is a different target from the existing multiset task; V3's unique-pair metrics will not be compared directly with the historical multiset F1.

## Data and leakage boundaries

- Read only official train and dev split rows for V3 implementation, tests, and smoke testing. The V3 dataset interface will reject split names other than `train` and `dev`.
- Use the split records' inline quad annotations to make targets. Do not use gold terms, taxonomy codes, sentiment beyond its target mapping, implicitness beyond target construction, or quad-image fields as model inputs.
- Keep text as model input. Preserve segment ID, source review ID, hotel ID, language, original implicit-quad count, and duplicate-collapse count in metadata only. Language remains available for reporting and later inference integration; the initial model consumes text only.
- Derive hotel grouping from the source review association/official split metadata, assert no train/dev hotel overlap, and never feed identifiers to the model.
- Keep the official split files and existing pipeline unchanged. Test split is neither opened nor instantiated by V3 code paths.

The collision audit found 37 normalized text groups shared between train and dev and no conflicting target vectors in either split or across the overlap. It found 96 train and 21 dev implicit quad occurrences that collapse to repeated-pair presence labels. Four labels have zero dev support: BRANDING / positive, BRANDING / neutral, LOYALTY / neutral, and SERVICE / neutral. These limits will be reported with V3 results. Language metadata does not resolve target ambiguity in the audited records; it is retained as metadata and may support multilingual analysis.

## Model

- Fine-tune `FacebookAI/xlm-roberta-base` at revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, using the matching tokenizer revision.
- Pool the final hidden state at the `<s>` position, apply dropout, then a linear head with 18 logits. Train with sigmoid-compatible binary cross entropy with logits.
- No CLIP, image features, language embedding, IDs, review context, or gold annotation fields are model features in this text-only run.
- Preserve new V3 code/config/run paths. Do not modify the existing 23-slot structured model, evaluator, checkpoints, or result artifacts.

## Initial training protocol (not yet run)

Initial configuration is unweighted BCE. Planned reproducibility defaults are seed 1729, AdamW, learning rate `2e-5`, weight decay `0.01`, maximum sequence length 256, micro-batch 2, gradient accumulation 16 (effective batch 32), maximum 20 epochs, linear warmup/decay, and early-stopping patience 4. These fine-tuning values are explicit V3-only choices, not copied from the frozen-feature structured-model run.

Use CUDA mixed precision (FP16 with gradient scaling) when supported. Begin with a real pretrained GPU smoke test before any full training. The initial no-checkpointing smoke completed but reported peak allocation above the GPU's reported total, so the V3 config enables activation checkpointing; the checkpointed smoke also completed, though allocator peaks remained above the reported total and are documented as an unresolved memory-accounting risk. The full training command must require CUDA and fail instead of silently falling back to CPU. Save the exact config, seed/runtime metadata, label supports, logs, threshold, and best development checkpoint under a new V3 text run directory. Refuse to overwrite an existing run.

Optional class weighting is a separate variant and is not part of the initial unweighted baseline. If explicitly run later, derive per-label `pos_weight` from training multi-hot targets only, cap it at 20, and set weight to 1 for a label with zero train positives while recording it as unsupported. Such a label receives negative-only BCE supervision and cannot learn positive examples; do not infer support from dev/test. Do not mix weighted and unweighted results under one checkpoint/run directory.

## Threshold, checkpoint selection, and metrics

- Select one global label threshold on dev from `{0.00, 0.05, …, 1.00}` to maximize unique-pair micro-F1. If scores tie, select the highest threshold. Freeze the chosen threshold in the best checkpoint metadata. Never inspect test labels or select on test.
- Select the best checkpoint by dev unique-pair presence micro-F1. Stop after four consecutive epochs without improvement, up to 20 epochs.
- Primary metric: micro precision, recall, and F1 over all segment × 18 binary decisions at the selected threshold, reporting TP/FP/FN.
- Secondary metrics: per-label precision/recall/F1 and positive support; fixed-label macro-F1 over all 18 labels; exact-set accuracy; mean/median/distribution of predicted label count; and cardinality error against unique gold label count.
- For any label with zero gold development positives, report its support and per-label score as zero under the zero-denominator convention, and explicitly warn that the class is unsupported. Include all 18 labels in the fixed-label macro average; do not present it without the warning.
- Language-stratified metrics may use metadata after predictions, but language and identifiers remain outside model inputs. Language never serves as a collision repair mechanism.

## Tests and execution gate

Before training, add tests for target shape and ordering, aspect-implicit-only construction, all-zero targets, duplicate collapse, train/dev hotel separation, threshold selection/tie-breaking, train-only class weighting/zero support, and metric calculations including unsupported classes. Run these and existing relevant tests that do not open HAMOS test data.

Then run a small GPU smoke test on actual train text only: load the pinned tokenizer/model, run a forward and backward pass with unweighted BCE, check finite logits/loss/gradients, optimizer update, encoder parameter update, CUDA peak memory, and throughput. Do not extract full features, train epochs, or run evaluation on test. Produce `reports/v3_training_readiness.md` with tests, GPU details, smoke results, available memory, and a realistic full-training estimate. Stop for explicit user approval before full training.

## Remaining human approval gate

The architecture/task are approved. **Full training is still not authorized** by this specification. The requested implementation, tests, GPU smoke test, and readiness report must be complete first; then report estimated training duration and available memory and wait for explicit approval.
