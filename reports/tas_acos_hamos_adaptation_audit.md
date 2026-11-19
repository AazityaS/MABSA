# TAS / ACOS Adaptation Audit for HAMOS

**Scope:** architecture and data-formulation review only. No model was trained, no HAMOS test data was accessed, and no HAMOS source, configuration, annotation, checkpoint, cache, or split was changed. Third-party repositories were inspected at pinned commits in a temporary directory; those clones are removed after this report is verified.

## Executive recommendation

Use a **candidate-conditioned, two-stage status prediction approach** as the next research implementation: for each segment and each of the 18 predefined broad-category × sentiment pairs, predict (1) whether that pair occurs in any quad, and (2), conditional on its occurrence, whether at least one matching quad has an implicit aspect. The primary implicit-pair score is the product of the two probabilities, thresholded globally on development data. This is a HAMOS-specific hybrid inspired by TAS pair conditioning and ACOS-style explicit/implicit distinction; it is not a literal port of either system.

This formulation keeps the approved unique-pair implicit prediction target, uses explicit annotations as auxiliary supervision, and avoids serializing 18 XLM-R inputs per segment if a shared segment encoder and candidate-pair scoring head are used. It does **not** recover multiplicity: repeated identical pairs remain one presence label, as already specified for V3. It should be treated as a new experiment with its own config, checkpoint, metrics, and development-only selection. Do not compare its unique-pair metric directly to the original multiset task's F1.

The direct TAS-style 18-query classifier is the closest published analogue, but its all-pairs data expansion is large. The original ACOS extraction pipeline models terms, opinions, implicitness, and category-sentiment; this is unnecessarily broad for HAMOS's implicit broad-pair presence target and adds several error-propagation stages.

## Sources inspected

- TAS-BERT paper: [Wan et al., “Target-Aspect-Sentiment Joint Detection for Aspect-Based Sentiment Analysis,” AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6447), [paper PDF](https://cdn.aaai.org/ojs/6447/6447-13-9672-1-10-20200517.pdf).
- HOSSemEval implementation: [repository](https://github.com/dtseedx/HOSSemEval-EB23), inspected at commit [`873d1f8`](https://github.com/dtseedx/HOSSemEval-EB23/tree/873d1f8fd035ce740efae1224cf33609b87ee25a). Relevant files include `TAS/data/data_preprocessing_for_TAS.py`, `TAS/TAS-BERT/processor.py`, `TAS/TAS-BERT/TAS_BERT_joint.py`, `TAS/TAS-BERT/modeling.py`, and `TAS/TAS-BERT/evaluation_for_TSD_ASD_TASD.py`.
- ACOS paper: [Cai et al., “Aspect-Category-Opinion-Sentiment Quadruple Extraction with Implicit Aspects and Opinions,” ACL 2021](https://aclanthology.org/2021.acl-long.29/), [paper PDF](https://aclanthology.org/2021.acl-long.29.pdf).
- ACOS implementation: [NUSTM/ACOS](https://github.com/NUSTM/ACOS), inspected at commit [`45d179a`](https://github.com/NUSTM/ACOS/tree/45d179a3dcc6a3dedd848d81b16f2552454805fe). The repository README identifies `Extract-Classify-ACOS` as the available implementation; it does not provide runnable code for every approach discussed in the paper.
- HAMOS facts below were taken from the current V3 specification, label-collision audit, training-readiness and frozen-threshold reports, plus official **train/dev only** splits. No test split file or test labels were opened.

## HAMOS target and data facts

The primary HAMOS target under consideration is **presence of an implicit aspect's broad category–sentiment pair in a segment**. There are 6 broad categories × 3 sentiments = 18 candidate labels. Labels are multi-hot; repeated instances of the same pair within a segment collapse to one positive. Gold terms, opinion terms, sentiment/category labels, and identifiers are not model inputs.

| Measure | Train | Dev |
|---|---:|---:|
| Official segments | 8,816 | 1,895 |
| Implicit-aspect quad occurrences | 547 | 119 |
| Segments with ≥1 implicit pair | 370 | 77 |
| Unique implicit pair-presence positives after duplicate collapse | 451 | 98 |
| Candidate segment × pair rows (18 per segment) | 158,688 | 34,110 |
| Candidate rows positive for implicit-pair presence | 451 | 98 |
| Candidate rows negative for implicit-pair presence | 158,237 | 34,012 |
| Unique segment × pair positives from all quads (explicit or implicit) | 11,663 | 2,501 |

For a hybrid status model, the implicit-positive conditional examples are 451 train / 98 dev; candidate pairs present only explicitly are 11,212 / 2,403; candidate pairs absent from all quads are 147,025 / 31,609. Thus the conditional implicit-positive rate among pairs known to occur is 3.87% train and 3.92% dev, compared with the unconditional candidate-row rate of 0.284% and 0.287%. This improves the class balance of the conditional decision but it remains a rare label.

There are four labels with zero train positive support and nine labels with zero dev positive support. HAMOS is hotel-disjoint between train and dev. The earlier text-collision audit found 37 exact normalized text groups shared across train/dev and no conflicting implicit target vectors in those groups; 96 train and 21 dev implicit quad occurrences collapse into repeated-pair presence labels. These are known data properties, not a reason to leak IDs or alter official splits.

The existing frozen XLM-R linear-head diagnostic ranked dev positives poorly (global ROC-AUC 0.2949, average precision 0.00184 against 0.00287 positive-cell prevalence). An exact threshold sweep did not produce a useful operating point. This is evidence against relying on threshold adjustment alone; it does not establish that fine-tuned or candidate-conditioned XLM-R cannot learn the task.

## 1. TAS-BERT / TAS-Transformers

### What they predict

TAS divides target-aspect-sentiment detection into candidate-pair problems. Given sentence `x` and a candidate category/sentiment pair `(a,p)`, it predicts whether any target exists for that pair and extracts the target token span(s). The paper explicitly handles implicit targets: a positive pair prediction with no extracted target span is interpreted as a null target. Multiple explicit targets for a pair can be represented by multiple decoded spans. HOS code constructs rows for candidate pairs and tags target tokens; the BERT input puts sentence and the category-sentiment candidate in one sequence. The joint model has a binary yes/no head and a token tagging head, optionally CRF-based.

The HOS preprocessing enumerates category-sentiment candidates observed in training and emits positive/negative candidate rows. It merges multiple target spans into a tag sequence for a row. Null spans do not contribute token tags, so a positive pair with no tags denotes a null target; mixtures of implicit and explicit targets for one pair are not represented as a multiplicity-preserving list of null and span objects. The Transformers variant follows the same candidate-conditioned extraction idea with a Transformers implementation.

### Adaptation to HAMOS

For HAMOS, candidate pairs should be the fixed 18 broad-category × sentiment combinations, not just pairs present in a gold annotation. For each segment/pair, the binary target would be one if one or more aspect-implicit quads have that pair. In the HAMOS presence task, the target term itself is not required; the candidate pair is the query and the output is presence. This eliminates the token span head and its empty-span conventions.

The candidate category/sentiment is an inference-time query drawn from the fixed label inventory; it is not a gold label fed into a prediction of that same label. Candidate queries must be enumerated identically at train, dev, and inference. A direct adaptation can append candidate text to each segment and run XLM-R for each of 18 pairs. This is semantically clear, but costs up to 18 encoder passes per segment. A more feasible variant encodes each segment once and scores learned/embedded candidate-pair representations against the shared text representation; that is a new factorized candidate-conditioned head rather than a literal TAS-BERT port.

TAS naturally accommodates multiple different category-sentiment outputs because multiple candidate rows may be positive for one segment. It does not preserve repeated identical quads under binary presence semantics, which is consistent with V3's already approved duplicate collapse. If exact quad multiplicity were the target, a binary candidate classifier would be insufficient.

### Languages and splits

Replace BERT with the already pinned XLM-R-base encoder and its tokenizer, using the official HAMOS train/dev split. XLM-R supports multilingual input, but English/Vietnamese quality is an empirical question; language metadata should remain metadata unless explicitly approved as an inference feature. Hotel IDs, review IDs and segment IDs are grouping/audit metadata only. Split by the existing hotel-disjoint partition and never tune using test.

### Feasibility and source limitations

The HOS README specifies an old stack (Python 3.6.8, PyTorch 1.0.1, TensorFlow 1.13.1 for conversion, and old ancillary packages). It should not be installed into the project's Python 3.12 / PyTorch 2.11 CUDA environment. Port the small conceptual components only. The HOS README also shows test evaluation during training and says the selected epoch was chosen using test TASD performance; reproducing that protocol would leak test information and is explicitly inappropriate for HAMOS. Its reported TAS score is not comparable to HAMOS implicit-only unique-pair F1.

## 2. ACOS / Extract-Classify

### What it predicts

ACOS aims to extract aspect term, category, opinion term, and sentiment as a structured quadruple, including null aspect and/or opinion terms. The inspected Extract-Classify implementation has two stages:

1. A token-level aspect/opinion tagger with separate aspect-implicit and opinion-implicit flags.
2. A category-sentiment classifier conditioned on candidate aspect/opinion representations.

The stage-one code ORs span tags and implicit flags across a sentence's quads. Stage two uses multi-label category-sentiment logits for a candidate aspect/opinion pair. The paper evaluates complete quadruple matches and reports implicit subsets. This formulation is useful when term extraction and opinion structure are part of the scientific task.

### Adaptation to HAMOS

HAMOS's proposed V3 target does not ask for aspect/opinion terms. It asks which implicit broad category-sentiment pairs are present. An ACOS-style first stage would therefore add unneeded span extraction and opinion predictions. If gold spans were used to condition the second stage, that would be target leakage at inference; predicted spans instead would introduce an additional source of errors and combinatorial candidate pairing. The stage-one OR construction also loses alignment between individual quads where multiple aspects/opinions occur in one segment.

The ACOS idea worth retaining is an explicit implicitness distinction. It can be adapted without aspect/opinion spans: for each of the fixed 18 candidate pairs, classify its status as absent, present only with explicit aspects, or present with at least one implicit aspect. The latter class is the V3 target. A two-head formulation (any-pair presence and conditional implicit presence) has a direct decomposition and permits auxiliary supervision from all quads.

### Feasibility and source limitations

The ACOS README targets Python 3.7 / PyTorch 1.8 / `pytorch-crf` 0.7.2. Do not install these legacy dependencies in the verified environment. Reimplementing the small relevant status head is safer than porting the full CRF-and-two-stage pipeline. The inspected training scripts choose checkpoints on development F1 but also invoke test evaluation when development improves; that behavior must not be copied.

## 3. Candidate approaches

### A. Candidate-conditioned implicit-pair presence (TAS-inspired)

**Input:** original segment plus one fixed candidate category-sentiment pair. **Output:** probability the candidate pair occurs among aspect-implicit quads.

*Strengths:* directly matches the unique-pair primary target; handles multiple distinct pairs through multiple positive queries; candidate identity is explicit, so identical text can have different answers for different candidate pairs; null aspects require no aspect-term span output.

*Weaknesses:* direct text-pair serialization multiplies encoder work by 18; unconditional positive prevalence is below 0.3%; four labels have no train positives; it discards repeat counts by design.

### B. ACOS-style term extraction plus classification

**Input:** segment, predicted aspect/opinion spans and implicitness. **Output:** category-sentiment classes for candidate aspect/opinion pairs, with null flags.

*Strengths:* richer output; explicitly represents null aspects/opinions; potentially useful if the research question expands to term/opinion extraction or full ACOS tuples.

*Weaknesses:* does not align with current primary endpoint; HAMOS opinion-span supervision has known annotation/span warnings and opinion-implicit F1 has been extremely weak in the existing slot model; stage errors compound; candidate spans and their pairings add complexity. Gold terms cannot be used at test inference. It is not the minimal defensible next experiment for implicit category-sentiment presence.

### C. Hybrid pair-status prediction (recommended)

For each segment and each of the 18 candidate pairs, predict:

1. `any_pair`: whether at least one quad of any aspect-implicitness status has that broad category-sentiment pair.
2. `implicit_given_pair`: among pairs with any quad, whether at least one aspect-implicit quad has that pair.

The final inference score for the approved target is `P(any_pair) × P(implicit_given_pair)`. Do not hard-gate stage two at an arbitrary first-stage threshold. Select one global threshold for this combined score using development unique-pair micro-F1, then freeze it. When a pair has both explicit and implicit quads, its conditional target is positive. When it has explicit-only quads, its conditional target is negative. For absent pairs, train the first head as negative and omit the conditional loss. This uses all aspect category-sentiment annotations as training supervision while leaving the target at inference as implicit pair presence.

*Strengths:* directly matches V3's unique-pair endpoint; increases implicit-positive frequency within the conditional subset from 0.284% to 3.87% in train; explicit-only and absent examples help distinguish the concepts; a shared XLM-R encoder can process each segment once and score 18 candidate vectors with low extra cost; implicit aspects need no term string because the target is defined by the pair.

*Weaknesses:* conditional positives remain rare; factorized probability calibration can be difficult; auxiliary any-pair performance can dominate optimization unless the two losses and primary dev metric are monitored; there are four zero-support train labels, so positive prediction for them cannot be learned from HAMOS train annotations. This remains a unique-pair presence task and does not predict quads, terms, or multiplicity.

## 4. Quantitative formulation sketch

Rows are made from official train/dev segments × the 18 fixed candidates. All labels below are derived from annotations solely as targets:

| Candidate row status | Train | Dev | Interpretation |
|---|---:|---:|---|
| `any_pair = 1` | 11,663 | 2,501 | Pair occurs in at least one quad, explicit or implicit |
| `any_pair = 0` | 147,025 | 31,609 | Pair is absent |
| Conditional implicit positive within `any_pair=1` | 451 | 98 | At least one aspect-implicit quad has the pair |
| Conditional implicit negative within `any_pair=1` | 11,212 | 2,403 | Pair occurs, but only with explicit aspects |

Target construction should aggregate by `(segment_id, candidate_pair)` using set presence. Gold term/opinion text, language labels (unless explicitly authorized as inputs), and all IDs remain outside the feature tensor. Preserve original source annotations and keep IDs only to align rows, group by hotel, and audit outputs. The existing official split is retained without resampling or record movement.

## 5. Ranked assessment

| Rank | Approach | Expected predictive potential | HAMOS compatibility | Implementation difficulty | Runtime / RTX 3050 6 GB |
|---:|---|---|---|---|---|
| 1 | C. Shared-encoder pair-status hybrid | Highest of these options: conditional supervision uses explicit-only evidence and a less extreme positive rate; still uncertain due to small implicit support | High; primary implicit unique-pair output retained | Medium | Similar encoder-pass count to one XLM-R pass per segment per epoch; likely viable with microbatch 2–4, accumulation, FP16 and checkpointing as needed |
| 2 | A. Direct TAS-style pair presence | Moderate; explicit candidate query may improve label-specific learning, but severe rare positives persist | Very high for unique pair presence; no term needed for null aspect | Medium-high if reusing the segment encoder; high if 18 separate serialized passes | Up to 18× sequence encoding with literal TAS input; factorized shared encoder reduces compute but is no longer a literal port |
| 3 | B. ACOS extraction/classification | Potentially high for a full ACOS task, uncertain/low for this narrower endpoint due to propagated span errors | Low-medium; substantially broader than target and not required for pair presence | High | CRF tagger plus candidate aspect/opinion classification; memory and runtime greater than necessary |

Current 23-slot structured prediction retains duplicate-sensitive set semantics and has shown severe overproduction. It is still the most annotation-faithful representation if the goal is full quad multiset recovery, but it is not the same target as unique-pair V3. V3-Frozen's weak ranking suggests frozen embeddings plus a linear head are inadequate; it does not establish that fine-tuning or candidate-conditioned supervision will fail.

## 6. Minimal implementation recommendation

Implement **C only** as a new isolated experiment. Reuse the approved V3 train/dev target-building and XLM-R pinned revision; encode each segment once. Add an 18-candidate scorer with two prediction heads, using a candidate-pair embedding or a small factorized scorer over the segment representation. Train the any-pair head on all 18 candidates and train the conditional implicitness head only for candidates with any annotated pair. Define the primary combined score as the product described above. Begin with unweighted binary cross entropy for both heads; class weighting can be a later train-only, separately configured experiment, not bundled into this architecture test.

Evaluation must retain the unique-pair target semantics: micro precision/recall/F1 from thresholded 18-label sets, per-label support/metrics with unsupported labels clearly marked, exact-set accuracy, and predicted-label cardinality. Select the global combined-score threshold on dev only; early-stop on dev primary unique-pair micro-F1. Keep language and IDs as metadata, use the existing hotel-disjoint train/dev split, and do not access test until separately authorized.

Do not report ACOS or HOS leaderboard numbers as comparable. Do not call the hybrid “quad extraction”: it predicts pair presence, not aspect terms, opinion terms, or duplicate multiplicity.

## 7. Resource estimate and risks

**Measured context:** on the RTX 3050 Laptop GPU (6 GB), a prior warmed XLM-R smoke measurement took about 1.224 seconds for a batch of two with short 12-token inputs. Train text token lengths were mean 18.24, median 15, P90 33, P95 42, max 182; dev lengths were mean 18.41, median 15, P90 32, P95 41, max 130. These are not a full-epoch benchmark. The readiness estimate for full XLM-R fine-tuning was approximately 90 minutes train plus 19 minutes dev per epoch at the measured short-input rate, with roughly 9–15 hours for 5–8 epochs and up to 36 hours for 20 epochs. Allocator-reported peak memory exceeded the physical total despite a successful smoke test, so memory accounting and stable batch settings remain an operational uncertainty.

For the recommended shared-encoder C model, compute remains approximately one encoder pass per segment, plus inexpensive scoring over 18 candidates. **Estimate:** about 1.8–2.5 hours per train+dev epoch using the prior rates; a 5–8 epoch run could take roughly 9–20 hours. This may exceed a six-hour deadline and must be timed using the actual new training loop before committing to full training. A GPU microbatch of 2, gradient accumulation, mixed precision, and gradient checkpointing only if needed are practical safeguards, but may change throughput. If 18 candidates are naively re-encoded, runtime can approach an order of magnitude higher and is not recommended.

Principal scientific risks: only 547/119 implicit quad occurrences; zero training positives for four labels; a small dev positive set; language-dependent generalization; auxiliary explicit-pair objective may improve pair semantics while not improving implicitness; threshold tuning can overfit dev; product scores can under-rank rare positives; and unique-pair prediction loses duplicate count and individual quad associations.

## 8. Approval boundary

This report recommends the hybrid candidate-status model but implements nothing. Before implementation, approve whether the auxiliary any-pair task is acceptable while keeping implicit-pair presence as the sole primary endpoint. Any future training should be a new named run with a new configuration and should not overwrite the frozen baseline, challengers, feature cache, or historical evaluation artifacts.
