# Challenger V3 architecture feasibility

**Status:** architecture investigation only. No HAMOS source, configuration, checkpoint, cache, or annotation was modified, and no model was trained.

## Executive recommendation

For the next architecture, the most direct experiment is an **XLM-R segment encoder with an 18-label sigmoid head**, one output for each of the six broad aspect categories crossed with three sentiments. Train its targets from aspect-implicit annotations only. This directly asks whether each implicit category–sentiment pair is present in the segment and removes the 23-slot cardinality problem.

This is a deliberate **presence prediction** formulation. It collapses repeated identical category–sentiment quads into one positive label. It therefore changes the current multiset prediction target and cannot be presented as a like-for-like replacement under the current duplicate-sensitive score. Before implementation, approve a V3 evaluation statement that reports unique-pair presence results separately and also describes the duplicate limitation. If preserving duplicate counts is mandatory, retain the current set predictor as the compatible formulation; neither the HOS TAS preprocessing nor a plain 18-label head preserves multiplicity.

The HOS TAS family is the strongest conceptual reference for aspect-conditioned extraction, but its original target-term tagging and one-query-per-category/sentiment construction are more work than HAMOS needs. GAS and ASQP are useful structured-generation references, but their term-heavy sequence targets and decoding are unnecessarily complex for HAMOS’s category-level implicit endpoint and less attractive on the RTX 3050 6 GB.

## Evidence and scope

The upstream [HOSSemEval-EB23 repository](https://github.com/dtseedx/HOSSemEval-EB23) README describes six hotel-review categories—Amenity, Branding, Experience, Facility, Loyalty, and Service—and a task that predicts aspect category, aspect term, and sentiment. Its code was inspected without installing its requirements. Relevant source paths are `TAS/data/data_preprocessing_for_TAS.py`, `TAS/TAS-BERT/modeling.py`, `TAS/TAS-BERT/TAS_BERT_joint.py`, the corresponding `TAS/TAS-Transformers` files, `GAS/{data_utils.py,main.py,eval_utils.py}`, and `ASQP/{data_utils.py,main.py,eval_utils.py}`. The repository README lists Python 3.6.8, PyTorch 1.0.1, a TensorFlow 1.13.1 checkpoint-conversion dependency for TAS-BERT, and legacy versions of CRF/NLP packages. These requirements should not be installed into the HAMOS environment.

The original TAS-BERT paper describes joint target, aspect, and polarity detection, including implicit targets ([AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/6447)). GAS casts ABSA outputs into generated sequences ([ACL paper](https://aclanthology.org/2021.acl-short.64/)); ASQP generates category, aspect term, opinion term, and sentiment quads ([EMNLP paper](https://aclanthology.org/2021.emnlp-main.726/)). The repository’s reported TAS-BERT Medium F1 of 79.74% is user-provided and is **not comparable** to HAMOS’s implicit-only results: task scope, split construction, target units, and metrics differ.

The HAMOS design reviewed here is segment-level prediction from original text, language metadata, and (for multimodal) review-derived candidate photos; all quads remain annotation targets. It has six primary broad categories, three sentiments, two implicitness flags, and up to 23 quads in one training segment. There are 16,784 training quads in the matched image-eligible view; 547 are aspect-implicit. Development has 119 implicit quads. The user-provided test support is 116, but this audit did not inspect HAMOS test data. HAMOS training and development hotel splits are disjoint. The existing model uses a 23-slot set representation and preserves duplicate-valued items.

## HOSSemEval model findings

| Model | What the inspected implementation predicts | Implicit aspects and multiple targets | Fit for HAMOS |
|---|---|---|---|
| **TAS-BERT** | For each segment and candidate aspect–sentiment pair, jointly predicts a binary yes/no label and a token-level target-term tag sequence (TO/BIO, optionally CRF). The pair is supplied as the second input sequence; the label answers whether that pair occurs, and token tags locate terms. | A null/implicit aspect has a positive pair label but no target tokens. The HOS preprocessing emits rows over the category–polarity combinations, marks presence with `yes_no`, and uses all-O tags for a target with no explicit span. Distinct pairs can be found over multiple queries. Preprocessing stores present pair keys in a set and merges target tags for repeated pair keys, so repeated identical pairs are collapsed and separate same-pair quad counts are not retained. | Best architectural reference for interpretable category-conditioned detection, but the term extraction head and 18 repeated queries are not needed for implicit category–sentiment presence. |
| **TAS-Transformers** | Same pair-conditioned yes/no plus token tagging design, with transformer encoder variants in place of the repository’s original BERT implementation. | Same query and target-span limitations as TAS-BERT. The README examples use a ConvBERT checkpoint and BERT-style token alignment; this is not a ready multilingual XLM-R integration. | More adaptable than the original BERT code, but still requires tokenizer/tag alignment and a multilingual encoder port. |
| **GAS** | T5 sequence-to-sequence generation for tasks including TASD, with extraction or annotation-style output strings. The HOS TASD targets serialize one or more `(aspect term, category, sentiment)` predictions. | The code represents a null aspect as `it` in its paraphrase target (and the extraction/annotation target utilities recognize NULL forms). Multiple outputs are concatenated with a separator. Repeated tuples could be repeated in generated text, but there is no explicit count constraint; order and generation errors complicate exact recovery. It does not directly predict HAMOS’s implicitness flags. | Flexible structured generation, but term generation is surplus for HAMOS’s primary endpoint. Use a multilingual T5 family and rewrite target construction for HAMOS. More memory and decoding cost. |
| **ASQP** | T5 generates full `(category, aspect term, opinion term, sentiment)` quads. The repository has HOS quad data utilities and span-based parsing/scoring. | Multiple quads are linearized in one target sequence. The code replaces a NULL aspect term with `it`; opinion terms can also be serialized. Repetitions are representable in principle, but sequence order and decoding make duplicate fidelity fragile. Implicitness is encoded indirectly through null-term conventions rather than explicit HAMOS flags. | The broadest output schema, but over-scoped: HAMOS V3’s question does not require generating aspect/opinion terms, and output parsing adds failure modes. |

### Metrics and comparability

TAS reports target/aspect/sentiment task variants, including settings that include or ignore NULL targets, as well as aspect and sentiment subtasks. Its token-span and category/polarity matching is not HAMOS’s implicit-only segment-level endpoint. GAS and ASQP decode serialized targets and compute tuple/quad precision, recall, and F1; the inspected ASQP scoring includes term-oriented matching logic and removes NULL quads from its denominators in the active scoring path. None of these definitions should be copied as HAMOS’s primary metric. HAMOS’s approved matching, primary implicit endpoint, development threshold selection, and hotel-clustered evaluation remain the authority for the existing models.

### Implicitness, multiplicity, languages, and split protocol

- **Null aspect terms:** TAS can represent an implicit aspect as a positive category–sentiment query with no target span. GAS/ASQP serialize a null marker or substitute a placeholder. HAMOS should preserve an explicit binary target internally; placeholder words such as `it` should not be placed into model input.
- **Multiple quads:** TAS can produce several different category–sentiment pairs by asking multiple queries. GAS/ASQP produce a list in one sequence. The current 23-slot model supports multiple items directly and preserves duplicate multiplicity. TAS preprocessing collapses repeated pair keys; generation may repeat a tuple but does not guarantee count-faithful decoding.
- **Multilingual English/Vietnamese:** the HOS TAS-BERT code is based on uncased BERT and word-split/tag alignment. It would need XLM-R or another multilingual encoder, subword-to-token alignment rebuilt for XLM-R, and language-aware validation. GAS/ASQP default to `t5-base`, which should not be assumed to provide adequate Vietnamese coverage. A multilingual T5 checkpoint would need evaluation and may be larger. XLM-R is the lowest-risk encoder for HAMOS’s English/Vietnamese mix.
- **Hotel-disjoint splits:** none of the architectures inherently requires hotel IDs as input. Port only the HAMOS train/dev split files; do not use HOS splits or randomly split quads. Build every training target from HAMOS training annotations and use only HAMOS dev for model selection.

## Direct 18-label multi-label XLM-R classifier

### Definition

Encode one segment once with XLM-R. A linear or small MLP head emits 18 logits, indexed by `(broad_category, sentiment)`. For each label, the training target is 1 if at least one aspect-implicit quad with that pair occurs in the segment, otherwise 0. Explicit-only quads do not turn on the corresponding implicit label. In this formulation, aspect implicitness is built into the prediction target; a separate aspect-implicitness head is not required for the primary task. Opinion implicitness can remain a secondary head if it is an explicit research objective, but it should not affect the primary pair labels.

### Fit and trade-offs

- **Research fit:** closest direct match to “does the model identify implicit broad category–sentiment pairs?” It has no 23-slot existence decision, Hungarian assignment, or variable output cardinality.
- **Duplicate behavior:** targets are multi-hot, not counts. Two identical `(category, sentiment)` quads in a segment become one positive. The model cannot predict multiplicity. This is a task change from the current multiset formulation. Results need unique-pair presence metrics or a clearly labeled compatibility score with duplicate limitations; do not silently compare it as if it predicted the full multiset.
- **Rare labels:** calculate support from training segments after converting the implicit quads to 18 binary targets. Start with a transparent unweighted BCE baseline and publish the 18 training supports. If weighting is needed, derive `pos_weight` from training supports only, cap extreme weights, and state how zero-support labels are handled. Never calculate weights from dev/test labels. Because only 547 train implicit quads exist and several category/sentiment combinations may be extremely rare, class-weighted results need careful dev-only assessment.
- **Multilingual:** XLM-R is a good fit for English and Vietnamese relative to the inspected English-oriented HOS TAS setup. The small `other` group remains too heterogeneous for broad claims.
- **RTX 3050 6 GB:** full XLM-R-base fine-tuning is plausible but not guaranteed at maximum sequence length 256 and large batches. Use micro-batches around 2–4, gradient accumulation, mixed precision, and gradient checkpointing if needed; benchmark a real training batch before the full run. This is more memory-intensive than the current frozen cached-feature model, but far less expensive than decoding 18 separate T5 generations per sample. No training smoke test was run in this audit.
- **Multimodal later:** retain the same 18 outputs and add a projection of masked-mean CLIP candidate-image features with late concatenation/gated fusion. Compare text-only and multimodal on exactly the same eligible segment IDs. That extension is technically straightforward but needs a separate approved experiment; do not infer visual benefit from this feasibility review.

## Ranked architecture comparison

Ranks are qualitative estimates for HAMOS, not measured performance. Predictive potential means potential on the intended implicit category–sentiment task, not guaranteed score improvement.

| Rank | Approach | Predictive potential | Annotation compatibility | Implementation difficulty | Training time | RTX 3050 memory | Implicit handling |
|---:|---|---|---|---|---|---|---|
| 1 | **18-label multi-label XLM-R** | High for pair-presence detection; avoids the observed slot/cardinality failure mode. Rare labels are a serious limit. | High for the implicit presence question; **does not preserve duplicate quad multiplicity** or the full structured schema. | Low–medium. One encoder pass and one 18-logit head. | Medium; encoder fine-tuning over all segments. | Medium–high; likely needs small micro-batches, mixed precision, and possibly checkpointing. | Direct: every class is an implicit category–sentiment pair. Null aspect text is irrelevant to input. |
| 2 | **Adapted TAS-BERT-style query/tagger** | Medium–high for category-conditioned decisions; the pair query may help rare categories share parameters. | Medium. It can identify several distinct pairs, but the upstream preprocessing merges identical pair targets; term tags are unnecessary and would need redesign. | High. Rewrite for XLM-R, multilingual alignment, and HAMOS implicit-only pair labels; remove/rework span head. | High if run as 18 separate conditioned examples per segment; shared-encoder/vectorized implementation can reduce cost. | High due to repeated or expanded encoder inputs. | Strong for pair presence when a null target becomes pair-positive with no span. |
| 3 | **Current 23-slot set predictor** | Demonstrated low-to-moderate current dev performance; V1/V2 changes improved some outcomes but cardinality remains problematic. A better decoder/loss could still improve it. | Highest: supports all quads, flags, and duplicate multiset semantics. | Existing code is ready; substantive architectural changes are harder to diagnose quickly. | Low–medium using cached frozen features. | Low during cached-feature training; feature extraction already exists. | Directly predicts implicitness while retaining all structured outputs. |
| 4 | **GAS/ASQP generation** | Potentially expressive, but no evidence here it will beat XLM-R classification on HAMOS. | Medium for full terms/quads; less directly aligned with category-only implicit targets and duplicate scoring. | High. Retarget sequences, parser, multilingual checkpoint, and evaluation. | High; autoregressive training plus generation-based dev inference. | High; T5 fine-tuning may require micro-batch 1–2 and memory optimization. | Can serialize NULL aspect terms, but placeholder/token output errors can obscure implicitness. |

## Proposed V3 design and files to change after approval

**Proposed model:** fine-tuned XLM-R-base, one segment at a time, outputting 18 independent logits for implicit category–sentiment presence. Use one global decision threshold selected on HAMOS development data for the designated presence micro-F1, then freeze it. Train-only label supports determine any class weighting. Keep segment ID, hotel ID, review ID, and source quad/image links in metadata only. The first V3 should be text-only to isolate the architectural change; a visual branch should be a later paired extension. Evaluate a second variant only if approval permits a different presence-based target and metric. Retain the original models/artifacts unchanged.

If approved, likely files/components:

- New `src/implicit_multilabel_dataset.py` or a separate view in `src/structured_dataset.py` that constructs multi-hot labels from only aspect-implicit quads, preserving official splits and separating metadata from inputs.
- New `src/implicit_multilabel_model.py` for XLM-R plus an 18-logit head; use pinned tokenizer/model revisions and saved seed/config.
- New `src/implicit_multilabel_train.py` for train-only support calculation, optional capped BCE weights, dev-only threshold selection, early stopping, and checkpoint/config saving.
- New `src/implicit_multilabel_metrics.py` or a carefully isolated extension to `src/structured_metrics.py` for duplicate-collapsed pair-presence precision/recall/F1 and per-language support. Do not alter the current structured evaluator’s semantics.
- New config such as `configs/implicit_multilabel_v3.json`, new run paths under `runs/structured/challenger_v3/`, and a new V3 design/results report. No existing config or checkpoint should be overwritten.
- New synthetic tests for multi-hot construction, duplicate collapse, zero-label segments, threshold selection, train-only weighting, multilingual batch handling, and hotel split separation.

## Feasibility and runtime estimate

These are planning estimates, not benchmark measurements. The 18-label text model should take about **2–4 hours** to implement, document, and validate. On an RTX 3050 Laptop GPU with 6 GB, plan **roughly 1–3 hours for one run** at batch 2–4, gradient accumulation, max length 256, and a modest epoch/patience schedule; actual throughput depends strongly on sequence length and GPU thermal/power limits. Budget **2–5 hours for sequential text-only and later multimodal runs**, plus a GPU smoke test and adjustment time. If memory or speed is poor, use gradient checkpointing and a shorter predeclared max length before considering external compute. Do not install the HOS repository’s legacy dependencies into the existing environment.

## Limitations and approval gates

1. Approve or reject the presence-formulation change. It discards repeated identical target multiplicity and is not the same prediction object as the current multiset model.
2. Approve the V3 primary metric. A unique-pair presence metric is the natural metric for this head. Scoring only by the existing multiset metric would penalize an output the model cannot represent.
3. Approve whether V3 is text-only first (recommended) or whether to include an image branch in the first implementation.
4. Approve the development-only threshold procedure and whether BCE class weighting is used. Any weights must come from train labels only.
5. Clarify the upstream repository inspection incident: during file discovery, a broad recursive source search also traversed the public HOSSemEval checkout’s bundled split text files and returned a matching line from its test file. No HAMOS test file, label, prediction, or metric was opened; the upstream example was not used in the analysis. The temporary checkout was removed. This should be considered when interpreting the “no test split access” restriction, although it did not expose HAMOS test data.

## References

- [HOSSemEval-EB23 repository and README](https://github.com/dtseedx/HOSSemEval-EB23)
- [TAS-BERT, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6447)
- [GAS, ACL 2021](https://aclanthology.org/2021.acl-short.64/)
- [ASQP, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.726/)
