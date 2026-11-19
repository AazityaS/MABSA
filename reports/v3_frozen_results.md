# HAMOS V3-Frozen Results

## Scope and test-data boundary

V3-Frozen is a separate, text-only segment-level **18-label unique category–sentiment presence** experiment. Each development segment is represented by a multi-hot vector over the six broad categories crossed with three sentiments; repeated identical pairs collapse to one positive bit. This differs from the original structured task, which predicts duplicate-sensitive quad multisets.

Only official training and development IDs, targets, and embeddings were loaded. No test split was instantiated, no test embedding values were read, and no test evaluation was run. Existing V3 fine-tuning plans, structured baselines, checkpoints, caches, configurations, and historical reports were not overwritten.

## Cache provenance and compatibility

The frozen text feature cache was `cache/structured_features/9072ce0b71cd3470/`. Its manifest resolves `FacebookAI/xlm-roberta-base` to the exact V3-pinned revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`. The extraction implementation pools `last_hidden_state[:, 0]`, the `<s>` token representation expected by the V3 design. Embeddings are 768-dimensional float32 vectors.

| Check | Result |
|---|---:|
| Official train IDs with one aligned embedding | 8,816 / 8,816 |
| Official dev IDs with one aligned embedding | 1,895 / 1,895 |
| Train/dev ID overlap | 0 |
| Train/dev hotel overlap | 0 |
| Missing selected embeddings | 0 |
| Non-finite selected values | 0 |
| Selected vector shape | 768, all rows |
| Selected feature payload read | 32,904,192 bytes, exactly 10,711 × 768 × 4 |
| Test embedding values read | No |

The cache is a PyTorch archive whose batched tensor views can share backing storage across row boundaries. To honor the train/dev-only constraint, the compatibility reader read the archive index and only byte ranges for selected train/dev rows. It touched 5,356 storage members, including one partial-storage read for a selected row sharing storage with an unselected row. No unselected feature values were loaded. The cache audit is recorded in the run's `cache_provenance.json`.

## Training setup and result

The implementation used a single linear layer from 768 to 18 logits, unweighted binary cross-entropy, seed 1729, and AdamW at learning rate 0.001. Development threshold selection used a single global threshold and unique-pair presence micro-F1 over the configured fixed grid. Early stopping monitored that development metric. No images, language features, IDs, class weights, or extra inputs were used.

| Epoch | Train BCE | Dev BCE | Dev micro-F1 | Selected threshold |
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

Best checkpoint: epoch 1 (ties retain the earliest best epoch). Training stopped after eight epochs without a development F1 improvement. The recorded training/evaluation loop elapsed time was 3.26 seconds; the full process, including startup and selective cache loading, took approximately 13 seconds. Peak CUDA allocated memory was 20,338,176 bytes on the NVIDIA GeForce RTX 3050 6GB Laptop GPU.

### Development metrics at the selected operating point

| Metric | Result |
|---|---:|
| Segments | 1,895 |
| Gold positive segment-label pairs | 98 |
| Precision | 0.002873 |
| Recall | 1.000000 |
| Unique-pair micro-F1 | 0.005730 |
| TP / FP / FN / TN | 98 / 34,012 / 0 / 0 |
| Exact-set accuracy | 0 / 1,895 = 0 |
| Macro-F1, fixed 18-label order | 0.005674 |
| Macro-F1 caveat | 9 labels have zero dev support; zero-F1 convention included |
| Mean gold labels per segment | 0.051715 |
| Mean / median predicted labels | 18 / 18 |
| Predicted-label-count distribution | 18 labels: 1,895 segments |
| Cardinality MAE / signed bias | 17.948285 / +17.948285 |

At threshold 0.00, the classifier emits every label for every development segment. This produces full recall by construction and near-zero precision. The threshold is the metric-selected result under the approved fixed grid, not a useful operating point. It must not be characterized as successful prediction.

### Per-label development results

Support is the number of positive segment-label pairs. Because all labels were predicted for all 1,895 segments, each label has TP equal to support, FN 0, and TN 0; FP is 1,895 minus support.

| Category / sentiment | Support | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| AMENITY / negative | 1 | 1 / 1,894 / 0 | 0.000528 | 1.000 | 0.001055 |
| AMENITY / neutral | 0 | 0 / 1,895 / 0 | 0 | 0 | 0 |
| AMENITY / positive | 2 | 2 / 1,893 / 0 | 0.001055 | 1.000 | 0.002109 |
| BRANDING / negative | 0 | 0 / 1,895 / 0 | 0 | 0 | 0 |
| BRANDING / neutral | 0 | 0 / 1,895 / 0 | 0 | 0 | 0 |
| BRANDING / positive | 0 | 0 / 1,895 / 0 | 0 | 0 | 0 |
| EXPERIENCE / negative | 4 | 4 / 1,891 / 0 | 0.002111 | 1.000 | 0.004213 |
| EXPERIENCE / neutral | 0 | 0 / 1,895 / 0 | 0 | 0 | 0 |
| EXPERIENCE / positive | 30 | 30 / 1,865 / 0 | 0.015831 | 1.000 | 0.031169 |
| FACILITY / negative | 3 | 3 / 1,892 / 0 | 0.001583 | 1.000 | 0.003161 |
| FACILITY / neutral | 0 | 0 / 1,895 / 0 | 0 | 0 | 0 |
| FACILITY / positive | 34 | 34 / 1,861 / 0 | 0.017942 | 1.000 | 0.035251 |
| LOYALTY / negative | 0 | 0 / 1,895 / 0 | 0 | 0 | 0 |
| LOYALTY / neutral | 0 | 0 / 1,895 / 0 | 0 | 0 | 0 |
| LOYALTY / positive | 11 | 11 / 1,884 / 0 | 0.005805 | 1.000 | 0.011542 |
| SERVICE / negative | 1 | 1 / 1,894 / 0 | 0.000528 | 1.000 | 0.001055 |
| SERVICE / neutral | 0 | 0 / 1,895 / 0 | 0 | 0 | 0 |
| SERVICE / positive | 12 | 12 / 1,883 / 0 | 0.006332 | 1.000 | 0.012585 |

## Train label supports

Counts below are positive **segments** per label, consistent with multi-hot unique-pair targets. The run had 8,816 train segments, of which 8,446 had all-zero targets; 1,895 dev segments, of which 1,818 were all-zero.

| Category | Negative | Neutral | Positive |
|---|---:|---:|---:|
| AMENITY | 0 | 1 | 8 |
| BRANDING | 2 | 0 | 6 |
| EXPERIENCE | 17 | 1 | 148 |
| FACILITY | 12 | 1 | 153 |
| LOYALTY | 3 | 0 | 55 |
| SERVICE | 4 | 0 | 40 |

The severe positive/negative imbalance (only 451 train segment-label positives across 18 labels versus 8,816 × 18 possible positions) and the high all-zero rate are relevant to interpreting unweighted BCE. Four labels have no training positive support; nine labels have no development support. A linear head over frozen sentence embeddings has little capacity to adapt representation to this sparse label task.

## Comparison and interpretation

The frozen unique-pair presence metric is **not directly comparable** to the original structured model's duplicate-sensitive multiset F1. They have different units, outputs, and matching semantics. For contextual reference only, prior development structured primary F1 values were: original text-only 0.1145, original multimodal 0.1330; Challenger V1 text-only 0.2727 and multimodal 0.1405; Challenger V2 text-only 0.1009 and multimodal 0.1954. These numbers cannot be ranked against the V3-Frozen 0.005730 unique-pair F1.

The approved full fine-tuned V3 design has not been trained, so there is no V3 fine-tuned result to compare. This frozen experiment demonstrates that the specified linear head plus unweighted BCE and current threshold grid fails to produce a useful dev operating point. It does not establish that fine-tuning XLM-R would fail, because fine-tuning can adapt representations; however, it also does not provide positive evidence that the planned full run will succeed. Given the known 9–36 hour estimate and the six-hour submission window, this result does **not** justify starting the original full fine-tuning run under the current deadline without a revised, approved plan.

## Artifacts

Configuration: [implicit_presence_v3_frozen.json](../configs/implicit_presence_v3_frozen.json)

Run directory: [V3-Frozen artifacts](../runs/structured/challenger_v3/frozen/)

Implementation and tests: `src/v3_frozen_*.py`, `tests/test_v3_frozen.py`.

Relevant checks before training: 36 tests passed across V3-Frozen cache/data/model/metric tests and existing V3/label-mapping/structured protocol tests. Tests instantiated only train and dev datasets. The train and dev split IDs and hotel groups remained separate; source data and annotations were not changed.

**Conclusion:** V3-Frozen completed successfully as an isolated experiment, but its development predictions are degenerate at the selected threshold and performance is inadequate. No test data were accessed. No further experiment is started by this report.

