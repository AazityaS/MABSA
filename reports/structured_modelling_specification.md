# HAMOS structured modelling specification

**Status:** research and implementation specification only. No model, weights, source data, annotations, or official splits were modified; no model was trained.

## 1. Decision summary

The primary task should be **segment-level structured prediction of an unordered set of aspect–sentiment outputs**. For each segment, predict zero or more items containing the six-way broad aspect category, three-way sentiment, aspect implicitness, and opinion implicitness. Use the original segment text and language plus the review’s independently available candidate photo set. Do not create one classifier row per gold quad: the input may not identify which of several conflicting quads it represents.

The **primary experiment** compares text-only with multimodal segment-level structured prediction on the same eligible examples, using the same output schema and evaluation protocol. The primary scientific comparison is multimodal versus text-only performance on **implicit-aspect outputs**. Image-only is a separately reported exploratory weak baseline, not a directly comparable primary modality arm: identical review-level candidate photo sets recur across segments with different gold target sets. Given only 116 implicit test quads, report the primary implicit result as exploratory with uncertainty, not as a definitive general claim. See the [image-only feasibility audit](image_only_feasibility_audit.md).

**Recommended initial model design:** a modest fixed-slot set predictor with permutation-invariant matching, a multilingual text encoder, a visual encoder for each candidate photo, and late fusion of text and pooled candidate-image features. Use no more output slots than the observed training maximum of 23, with a no-output class for unused slots. Keep encoders frozen initially and cache their features if practical. This is a design recommendation, not an implementation decision already carried out.

## 2. Research question and scope

The proposal, *ImplicitMABSA: Multimodal Implicit Aspect-Based Sentiment Analysis Using Text and Images*, asks: “How effectively can the combination of textual and visual information improve the detection of implicit aspects and their associated sentiment in multimodal content?” It proposes text-only, image-only, and multimodal comparisons and identifies visual objects, scenes, attributes, and context as possible cues.

For HAMOS, scope this to multilingual hotel reviews and the category-level task its annotations can directly support:

> **On HAMOS’s hotel-disjoint test split, does independently available review-photo context improve prediction of implicit broad aspect categories and their associated sentiment over text alone?**

The model predicts an aspect category even when `aspect_term` is null; `aspect_implicit` marks that output as implicit. This is category-level implicit aspect prediction, not recovery of a missing phrase or free-form aspect-term extraction. Opinion-term extraction is also outside this initial task; only the annotated opinion implicitness flag is predicted.

## 3. Confirmed data facts and constraints

The verified files report 8,816/1,895/1,890 segments and 16,789/3,600/3,606 quads in train/dev/test. There are 782 implicit-aspect quads overall: 547/119/116 by split. The test implicit subset comprises 44 English, 61 Vietnamese, and 11 `other` cases; it contains only 2 neutral cases, 2 BRANDING cases, and 1 AMENITY case. Overall implicit aspects cover 516 reviews and 471 hotels. Category/sentiment and language distributions are strongly imbalanced.

The training split has 4,274 multi-quad segments, dev 911, and test 923. Segment cardinality ranges up to 23 quads in train, 15 in dev, and 13 in test; two training segments have more than 15 quads. A fixed-slot implementation must therefore preserve this long tail and must not silently truncate outputs. The long tail also makes cardinality evaluation and loss balancing important.

Candidate images are derived without annotations by joining `metadata/images.jsonl` to segments/reviews on `source_review_id`. The official multi-image candidate lists matched this metadata-derived pool on all 12,601 segment rows. There are 8,412 one-image reviews and 384 multi-image reviews (350 with two, 29 with three, five with four candidate images). There are 9,216 locally readable images and three inaccessible images, all in train and each the only candidate for its review. Those three reviews contain three segments and five quads in total. Dev and test candidate images are all locally available.

The `original_caption` field is present for 6,472/9,219 images, but its provenance is undocumented, it is incomplete, and observed captions include hotel-identifying content. Exclude captions from primary experiments. Consider a caption ablation only if provenance and deployment-time availability are established, and report it separately from visual-only evidence.

Official hotel overlap is zero across train/dev/test. Keep these splits intact; no random quad-level split or cross-split tuning is acceptable. Refer to the detailed counts and distributions in [dataset suitability audit](dataset_suitability_audit.md), [quad dataset summary](quad_dataset_summary.json), [ambiguity audit](model_input_ambiguity_audit.md), and [per-quad design](per_quad_modelling_design.md).

## 4. Prediction unit, schema, and leakage controls

### 4.1 Unit and inputs

One example is one **segment**, grouped with all of its gold quads. The inference input contains:

- `segment_text`: original segment text, unchanged.
- `language`: segment language metadata (`en`, `vi`, or the dataset’s `other` grouping).
- `candidate_images`: the available photos for that segment’s source review, constructed from image metadata and current inference inputs.

No image is copied from another review. For a review with no retrievable candidate photos, represent the candidate set as empty and mark it; the primary matched text-only and multimodal comparison excludes its segment(s) from both conditions under the eligibility rule below. The full text-only set may be reported separately. For image-only, use the separately documented exploratory empty-pool policy in Section 5 and the [image-only feasibility audit](image_only_feasibility_audit.md).

Do **not** read gold `primary_image_id` or `supporting_image_ids` to select model inputs. The gold quad-image links can be used for data validation and post-hoc analysis only. Do not provide aspect/opinion terms, spans, broad categories, taxonomy codes, sentiment, implicitness flags, quad IDs, or other target-derived fields as inputs. IDs can remain in evaluation metadata for tracing but must not enter the feature tensor. Language is a permitted input because it is segment metadata available at inference, not a target.

### 4.2 Output set

Each predicted set item is:

```text
(aspect_category ∈ 6 broad categories,
 sentiment ∈ {negative, neutral, positive},
 aspect_implicit ∈ {false, true},
 opinion_implicit ∈ {false, true})
```

Each gold quad corresponds to one set item. Preserve duplicate-valued items as separate elements if the source contains them; this is a multiset for scoring, not a deduplicated list of label tuples. Keep the 31 fine-grained `taxonomy_code` values as an optional auxiliary target/head, never an input. Broad categories are the primary category outputs. The workbook-verified code-to-category mapping is already held separately by `taxonomy_code_to_category` in the label mappings.

`aspect_term`, `opinion_term`, and their spans remain available only for metadata/error analysis. In particular, implicit aspect prediction must not depend on an aspect term supplied as a query: by definition it may be null. This task predicts the broad category and implicitness value directly.

## 5. Modality conditions and fair comparison

The primary comparison has two conditions—text-only and multimodal—on an identical eligible segment set, with the same output schema, matching rule, loss family, split policy, and evaluation code. Image-only is an additional, separately reported exploratory weak baseline with the same segment-level target schema, not a third arm for the primary modality comparison. Its limitations and measured ambiguity are documented in the [image-only feasibility audit](image_only_feasibility_audit.md).

| Condition | Input features | Purpose |
|---|---|---|
| **Text-only** | Text encoder representation + language metadata | Establish the text baseline. |
| **Image-only (exploratory)** | Review candidate-photo representations only | Weak diagnostic baseline; intrinsically ambiguous at segment level. |
| **Multimodal** | Text/language representation + review candidate-photo representations | Test whether visual context adds value over text. |

For the primary matched text-only versus multimodal comparison, exclude the three training segments whose reviews have no accessible candidate image from **both conditions**; dev and test require no exclusion. This leaves 8,813/1,895/1,890 segments for the primary train/dev/test runs (12,598 total). Keep a separate full text-only result on all 8,816/1,895/1,890 segments as a coverage result, not as the paired vision comparison. Labels remain unchanged in either view. Never filter all quads of a segment merely because a different segment or quad elsewhere in its review is unavailable. For image-only, either run an image-present-only exploratory variant on the same eligible examples and report that coverage, or retain all examples with an explicit empty-image-set representation and report the empty-pool coverage case separately; there are three such train segments and none in dev/test. Do not silently drop these examples.

The primary comparison must use the same review candidate-image set for all outputs from a segment, assembled without gold grounding. Do not use the `single_image` split’s gold/selected `primary_image_ids` as if known at inference. The `multi_image` metadata list can be cross-checked, but build the input pool from image metadata/review association so that inference inputs are not annotation-derived. Since 95.6% of reviews have one candidate, gains cannot by themselves establish robust photo-selection performance.

## 6. Recommended predictor and alternatives

### 6.1 Initial recommendation: fixed-slot set prediction

Use a shared segment encoder and a decoder or prediction head that produces up to 23 unordered output slots. Each slot predicts an existence/no-output decision and the four labels. Match predicted slots one-to-one with gold set items using minimum-cost bipartite matching during training. This makes output order irrelevant and permits multiple items with identical inputs or identical label tuples.

The maximum slot count of 23 is dictated by the observed training maximum; do not truncate. If implementation cost or convergence is problematic, a variable-length alternative may be investigated only after documenting a lossless representation of the two high-cardinality outliers. Any smaller slot cap must be rejected unless a deterministic, lossless policy is justified and the annotations remain recoverable.

### 6.2 Encoders and fusion

A feasible first configuration is a multilingual Transformer such as XLM-R base for text and a compact pretrained vision encoder such as CLIP ViT-B/32 for photos. Initially freeze both encoders; cache text and image embeddings where licensing, storage, and reproducibility permit. Add a learned language embedding or language ID feature to the text representation, with a documented `other` fallback. For image-only, omit text and language entirely.

For multiple candidate photos, encode each independently and aggregate the resulting vectors with a masked mean or learned attention pooling over **all available candidates**. The aggregation must not use gold primary/supporting IDs or target labels. Late fusion (concatenate or gated-combine pooled text and image vectors, then feed the set predictor) is the recommended first fusion baseline: it is easier to fit and diagnose than token-to-patch cross-attention, and it cleanly supports text-only/image-only ablations. Cross-modal attention over text tokens and image patches is a later experiment if the first model shows reliable signal and compute allows it.

For the primary text-only condition, use the same text encoder and set head without visual features. For the exploratory image-only condition, use candidate-photo aggregation and the same target/set schema and head capacity where feasible. It must receive **no segment text, language, segment IDs, gold primary/supporting image associations, or other target-derived information**. Repeated exact review photo sets can correspond to different segment-level target sets; this is an intrinsic input/target ambiguity, not a problem to solve by adding hidden identifiers or labels. A separate learned empty-set representation may be used if retaining empty-pool examples, but it must be explicitly marked as no available photos; if those examples are excluded from an image-present-only run, report coverage separately. This image-only output remains exploratory and is not evidence directly comparable to the primary text-only/multimodal contrast. See the [image-only feasibility audit](image_only_feasibility_audit.md).

### 6.3 Why not the other initial formulations?

- **Independent per-quad classification** is not defensible as the main task. The ambiguity audit found identical text-only inputs with conflicting primary targets for 42.65–44.11% of split instances; even text-plus-gold-primary-image inputs had conflicts for 41.35–43.06%. Per-quad rows also assume the correct quad/image association is available at inference.
- **Aspect-conditioned classification** can be useful for sentiment given a known aspect, but gold terms/categories would leak targets. A predicted aspect query makes this a pipeline with upstream aspect discovery and error propagation. A gold term is unavailable for implicit aspects. It is a later diagnostic, not the initial end-to-end test.
- **Set prediction** matches the actual one-segment-to-many-quads annotation and can emit category-bearing outputs even when terms are null. It supports count/multiplicity under a shared target schema. The image-only condition still has unresolved input/target ambiguity because its review-level visual input is repeated across segments; see the [image-only feasibility audit](image_only_feasibility_audit.md).

## 7. Training objective (for the later approved implementation)

For each segment, match the `K` gold items to distinct output slots. A matching cost can combine negative log probability for broad category and sentiment with binary cross-entropy costs for aspect and opinion implicitness; include taxonomy cost only in the optional taxonomy experiment. Apply the Hungarian algorithm (or an equivalent exact assignment solver) to obtain permutation-invariant matching. Train matched slots with categorical cross-entropy for category and sentiment and binary cross-entropy for each implicitness flag. Train unmatched slots as `no-output` with a lower/background-weighted loss so that the many empty slots do not dominate. Use the same matching and loss in all modality conditions.

The maximum slot count must be at least 23 for this split. One-to-one matching preserves repeated identical items as separate targets. The set objective should penalize missed gold items, spurious outputs, and wrong labels. Report predicted versus gold cardinality and exact-set performance, not only per-label accuracy. If fine-grained taxonomy is trained, use a separate optional 31-way head and check predicted taxonomy/category consistency at evaluation; do not feed the code into the category head.

Begin with an unweighted baseline. If class weighting or focal loss is tested for the rare implicit, neutral, AMENITY, BRANDING, or taxonomy classes, tune it on train/dev only and report it as a controlled sensitivity analysis. Do not rebalance test or use test feedback.

## 8. Evaluation protocol

### Primary endpoint

The primary endpoint is **implicit-aspect set performance** on the hotel-disjoint test set: compare predicted and gold items where the gold aspect is implicit, requiring the correct broad category and sentiment for a matched item. Report precision, recall, and F1 (micro and macro where defined), with implicit-aspect category macro-F1 and implicit sentiment results shown separately. Explicit-aspect outputs and the full-set task are secondary. A model must also predict the implicitness flag; report implicitness precision/recall/F1 and false positives on explicit cases.

For the complete structured task, report duplicate-aware set precision/recall/F1, exact-set accuracy, broad category macro-F1, sentiment macro-F1, both implicitness F1s, and cardinality error. Define matching/scoring rules before seeing test outcomes; count duplicate target tuples with multiset semantics. Also report per-class support and a confusion matrix where meaningful.

### Comparisons and uncertainty

- Train and select on train/dev only; use dev for early stopping and all hyperparameter choices. Evaluate the held-out test once after the protocol is frozen.
- Keep the official hotel-disjoint split files unchanged. Verify zero hotel overlap and retain all quads inside their original segment/review.
- **Primary comparison:** compare text-only versus multimodal on the identical 8,813/1,895/1,890 matched segment sets. This eligibility rule and the output/evaluation protocol are unchanged. Report the full text-only coverage run separately. **Image-only:** report as a separately labeled exploratory weak baseline, never as a directly comparable primary modality arm. The [image-only feasibility audit](image_only_feasibility_audit.md) found conflicting target sets among repeated photo-pool inputs for 3,609 train segments (40.94%), 782 dev segments (41.27%), and 765 test segments (40.48%).
- Report aggregate plus English and Vietnamese implicit results; mark `other` as descriptive only. State the small subgroup counts and do not make category-specific claims for test categories with one or two examples.
- Compute confidence intervals using resampling clustered by hotel (not independent quad bootstrap); also show raw counts. Only 70 test hotels contain implicit aspects, so intervals may remain wide.
- Report both the usual all-output score and the conditional implicit-output score, making explicit whether implicitness is gold-conditioned for metric selection. The model itself must never receive gold implicitness during prediction.

A positive aggregate difference alone does not prove vision caused it if modality-specific data differ. Keep the primary text-only/multimodal matched subset, training budget, stopping rule, output matching, and seeds comparable; report variability across multiple seeds if compute permits. Do not overinterpret small score changes relative to hotel-clustered uncertainty. The image-only baseline has an additional intrinsic ambiguity: the same review-level candidate image set may correspond to different segment-level target sets. Its scores are descriptive diagnostics, not a matched modality-effect estimate; report the ambiguity counts and image coverage alongside them, as detailed in the [image-only feasibility audit](image_only_feasibility_audit.md).

## 9. Experimental sequence

1. **Protocol lock and audit:** freeze split hashes, label mapping, candidate-pool builder, eligibility rule, target schema, and metric definitions. Verify no target-derived input fields and no hotel leakage.
2. **Text-only set baseline:** establish cardinality and multi-label/set behavior first; use all segments only as a clearly separate coverage result.
3. **Exploratory image-only baseline:** use only the independently available review candidate photos, with no text, language, segment IDs, gold image associations, or target-derived features. Label and interpret it as a weak, input-ambiguous diagnostic, not as a direct comparison arm. Report the audit’s conflict counts (3,609/40.94% train; 782/41.27% dev; 765/40.48% test) and state the empty-pool handling/coverage. See the [image-only feasibility audit](image_only_feasibility_audit.md).
4. **Matched multimodal baseline:** late-fuse text and aggregated candidate-photo features under the same objective and the exact primary eligible segment set used by text-only.
5. **Implicit-focused analysis:** inspect implicit false negatives/positives and category/sentiment performance with hotel-clustered uncertainty.
6. **Ablations:** compare image aggregation (masked mean vs learned attention), fusion (late concat vs gate), and optional fine-grained taxonomy supervision. Treat captions as a separate, conditional experiment only after provenance is established. Do not use gold primary image IDs in the main comparisons.
7. **Optional cross-attention:** evaluate only if simpler fusion is stable and resources permit; it is not required to answer the initial question.

No architecture or training code should be implemented until the specification is approved.

## 10. Compute and reproducibility plan

No weights have been downloaded and no runtime benchmark has been run. The following are planning estimates, not measured requirements. On a 6 GB GPU, start with frozen encoders, 224-pixel photos, text length capped at 256, mixed precision where supported, and a small batch (approximately 2–4 segments if both encoders run online). Use gradient accumulation to reach a stable effective batch. Feature caching may allow larger batches for the small set head and avoids recomputing frozen encoders. On a 16 GB Colab-class GPU, online frozen encoding should be less constrained, but measure before increasing sequence or image size. Avoid full encoder fine-tuning for the first comparison; it is likely to exceed comfortable memory or iteration budgets on 6 GB, especially with multiple photos and 23 output slots.

Record dependency and checkpoint identifiers, tokenizer/preprocessing versions, candidate image list and availability, seed, optimizer schedule, selected checkpoint rule, source fingerprints, and exact eligible segment IDs. Keep downloaded third-party images outside distributed artifacts and honor source-image rights. The current pipeline’s XLM-R setting and CLIP normalization are useful preparation defaults, not proof that a particular pretrained checkpoint is optimal.

## 11. Risks and interpretation limits

1. **Small implicit test set:** 116 quads across 70 hotels; neutral and some category groups have only one or two examples. Claims about category- or sentiment-specific improvements will be unstable.
2. **Severe imbalance:** implicit aspects are 3.26% of all quads and sentiment is predominantly positive. A high overall score can hide poor implicit recall; prioritize macro and implicit-specific measures.
3. **Visual availability and rights:** three training reviews have no accessible candidate image; test images are available in this snapshot but can disappear or be restricted. Report the availability date/snapshot and use only eligible matched cases.
4. **Candidate selection:** most reviews have one photo, limiting evidence about selecting useful images among alternatives. Gold quad grounding is unavailable at inference and must not guide the primary model.
5. **Set cardinality:** two training segments contain 20 and 23 quads. A 23-slot decoder is feasible in principle but presents class-imbalance and optimization challenges; measure cardinality recall and do not silently truncate.
6. **Language scope:** English and Vietnamese dominate. `other` aggregates languages; per-language performance, especially for the 11 implicit test cases in `other`, is descriptive only.
7. **Annotation quality:** prior diagnostics identified source records with explicit terms but empty spans and term/text mismatches. Terms/spans are not initial targets, but these inconsistencies caution against span-based claims. Do not repair or filter annotations silently.
8. **Visual grounding semantics:** the human-assigned primary image expresses a quad-level association, whereas deployment receives a review-level pool. The research comparison tests added review-photo context, not whether model selection reproduces human gold grounding.
9. **Caption leakage:** captions can be missing and may reveal hotel identity; exclude them from primary inputs.

## 12. Proposed implementation stages after approval

- **Stage 1 — Separate structured view:** add a segment-level `StructuredDataset` (or equivalent) beside the current segment and `QuadDataset` implementations. It returns the original segment text, language, metadata-derived candidate photos, and the complete ordered list of target records. Preserve original IDs and source files read-only.
- **Stage 2 — Batch and invariants:** collate variable-length target sets with padding/masks; batch multiple candidate photos per segment. Assert no terms/labels/image-association annotation fields enter model inputs, cardinality is not truncated, and all official segment/quad totals reconcile.
- **Stage 3 — Loss and metrics:** implement assignment, no-output handling, duplicate-aware set metrics, implicit-specific metrics, and hotel-clustered intervals; test on synthetic examples including duplicate labels and cardinality 23.
- **Stage 4 — Feature extraction and modality heads:** add cached/frozen text and visual encoders; preserve the shared output schema. Implement the matched text-only and multimodal primary conditions first. If image-only is implemented, keep it separately reported and enforce its input restrictions and coverage reporting using the [image-only feasibility audit](image_only_feasibility_audit.md).
- **Stage 5 — Experiments and reporting:** run the preregistered sequence above and report actual compute, seeds, matched denominators, confidence intervals, and limitations.

This document does not authorize those later implementation/training stages. Obtain approval before implementation or model training.

## 13. Relevant project artifacts

- `PJ_05.pdf` — proposal and original research question.
- `README.md` — HAMOS data structure, image metadata, annotations, and official split policy.
- `reports/dataset_suitability_audit.md` — implicit sample sizes and candidate-image feasibility.
- `reports/quad_dataset_summary.json` — actual quad counts, distributions, hashes, and hotel-overlap check.
- `reports/model_input_ambiguity_audit.md` — duplicate-input/conflicting-target diagnostics and examples.
- `reports/image_only_feasibility_audit.md` — repeated review-photo input groups, conflicting segment target sets, and image-only recommendation.
- `reports/per_quad_modelling_design.md` — current quad-level input/target design and availability policy.
- `src/data_loader.py`, `src/quad_dataset.py`, `src/label_mapping.py`, `src/config.py` — current data views, mappings, and preprocessing defaults. The proposed structured view is a separate addition; do not replace these representations.
