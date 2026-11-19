# Proposed evaluation and loss protocol

**Status:** proposal for approval before implementation. This document resolves the open choices in Sections 7–8 of `reports/structured_modelling_specification.md`; it does not change the dataset, annotations, splits, model architecture, or training configuration. No models were trained.

## 1. Fixed task contract

- **Example unit:** one segment with its complete gold quad set.
- **Per-item outputs:** one of six broad aspect categories; one of three sentiment classes; binary `aspect_implicit`; binary `opinion_implicit`.
- **Output capacity:** 23 slots, preserving the observed maximum. No gold item may be truncated.
- **Primary model comparison:** text-only versus multimodal on the exact same eligible segments, with the same output schema, objective, decoding, and evaluation protocol.
- **Image-only:** exploratory weak baseline only, separately reported because review image pools recur across segments with conflicting target sets. It does not determine the primary modality conclusion. See [image-only feasibility audit](image_only_feasibility_audit.md).
- The optional 31-way taxonomy code is excluded from this initial loss and primary evaluation. It can be added later as a separately approved auxiliary target without changing input features.

The notation below treats each target set as a **multiset**, so duplicate-valued quads remain separate items.

## 2. Prediction decoding and one-to-one matching

Each of 23 slots emits an objectness score and logits for category, sentiment, aspect implicitness, and opinion implicitness. Use one global objectness threshold, shared by all 23 slots. Select it on the development set by maximizing end-to-end implicit category–sentiment micro-F1 as defined in Section 3.1. Search the fixed grid `{0.00, 0.05, 0.10, …, 0.95, 1.00}`; include a slot when its objectness probability is greater than or equal to the candidate threshold. If thresholds tie on development micro-F1, choose the highest threshold (the more conservative output count). Freeze that threshold before test evaluation. Decode categorical outputs by argmax. Retain the fixed 0.5 sigmoid threshold for both binary implicitness flags; do not tune those flag thresholds in this procedure.

### 2.1 Evaluation assignment

For each segment, first decode its emitted slots. Match predictions and gold items one-to-one using a deterministic lexicographic assignment:

1. Maximize the number of **exact four-field target matches** `(category, sentiment, aspect_implicit, opinion_implicit)`.
2. Among assignments with the same number of exact matches, minimize the total Hamming distance across the four fields for the remaining pairs.
3. Resolve any remaining tie by stable slot order and stable gold record order, solely for reproducibility.

The assignment can be solved as a bipartite minimum-cost/max-cardinality matching problem. A predicted or gold item can be assigned at most once. Extra predictions and gold items remain unmatched. Quad IDs and term/span annotations are not part of matching. Duplicate target tuples are separate nodes, so two identical gold quads require two matched predictions for full credit.

### 2.2 Complete structured-set scoring

For complete-set precision/recall/F1, define a true positive as an exact four-field match under the assignment. Each unmatched prediction is a false positive (FP); each unmatched gold item is a false negative (FN). A paired but non-exact item is not a TP: count one FP and one FN. This penalizes both an incorrect output and the missed correct output. Exact set match requires every item to pair exactly and no unmatched items.

## 3. Implicit-aspect endpoints

The main research question is whether visual context helps predict **implicit broad aspect categories and their sentiment**. The main endpoint should require both the category–sentiment pair and correct implicit-aspect detection.

### 3.1 Primary: end-to-end implicit category–sentiment set F1

Create the gold implicit set from all gold items with `aspect_implicit=true`. Create the predicted implicit set from emitted items whose **predicted** `aspect_implicit=true`. Represent an item for this endpoint by `(aspect_category, sentiment)`; opinion implicitness is not part of this key.

Within each segment, perform maximum-cardinality exact one-to-one matching between those two sets by `(category, sentiment)`, preserving duplicates. Sum counts over segments:

- TP: matched implicit category–sentiment pairs.
- FP: unmatched predicted implicit pairs, including spurious implicit detections on otherwise explicit targets when they cannot match a gold implicit item.
- FN: unmatched gold implicit pairs, including gold implicit items predicted with `aspect_implicit=false` or not emitted at all.

This is the **primary implicit endpoint**, reported as micro precision, recall, and F1 over the test split. A false-negative implicitness prediction removes that slot from the predicted implicit set, so it cannot be a TP and its gold pair is an FN. A false-positive implicitness prediction on an explicit target adds a candidate to the predicted implicit set; if no gold implicit pair remains to match it, it is an FP. If an explicit and implicit gold item share the same category–sentiment tuple, multiset matching resolves available multiplicity without using IDs.

### 3.2 Diagnostic: category–sentiment quality on gold implicit aspects

Also report a **gold-implicit-conditioned** diagnostic to separate category/sentiment recognition from implicitness detection. This diagnostic uses its own one-to-one assignment, independent of Section 2.1: for each segment, take all gold items with `aspect_implicit=true` and all emitted predictions, regardless of the predicted aspect-implicit flag. First maximize the number of exact `(aspect_category, sentiment)` matches. Among the remaining items, form maximum-cardinality one-to-one pairs and minimize total Hamming distance across category and sentiment. Resolve remaining ties deterministically by stable prediction-slot order and gold record order. Do not use predicted or gold aspect implicitness or opinion implicitness in this assignment. Preserve duplicate items as separate nodes (multiset semantics); a prediction or gold item can be assigned at most once. Unmatched gold implicit items count as misses. Unmatched predictions are ignored by this gold-conditioned diagnostic because they are evaluated as detections by the primary endpoint.

This diagnostic answers, “Given each gold implicit target, were its category and sentiment recovered?” It does **not** require a correct predicted implicitness flag and must not be called end-to-end detection. A prediction with a false aspect-implicit flag can still score correctly here, but it fails the primary endpoint above. For each gold implicit item, a missing assigned prediction is a miss; a wrong category/sentiment is a classification error (one FP for the predicted class/pair and one FN for the gold class/pair); an exact category–sentiment match is correct. Report category and sentiment confusion matrices, per-class F1, macro-F1, and the proportion of gold implicit items with an exact category–sentiment match.

For these diagnostic confusion matrices, rows are gold labels and columns are predicted labels, followed by a `MISS` column for unmatched gold items. Unmatched predicted items are omitted, as specified above. Every assigned item contributes once, including duplicate-valued items. Category and sentiment matrix labels use the integer order from the taxonomy-derived dataset mapping.

### 3.3 Implicitness and opinion flag treatment

`aspect_implicit` participates directly in the primary endpoint through the predicted-implicit gate and gold-implicit target subset. `opinion_implicit` does **not** participate in the primary endpoint: the research question concerns implicit aspects and associated sentiment, and requiring an opinion-implicitness match would conflate a distinct target.

Report both flags as secondary classification outcomes using the full-set one-to-one assignment from Section 2.1. For each matched item, compare the relevant predicted and gold flag; an unmatched predicted item is a false positive for its predicted binary class, and an unmatched gold item is a false negative for its gold binary class. In particular, report:

- Aspect-implicit false positives: predicted implicit on explicit gold among aligned items, plus unmatched emitted items predicted implicit.
- Aspect-implicit false negatives: gold implicit predicted explicit among aligned items, plus unmatched implicit gold items.
- Opinion-implicit false positives/negatives analogously.

Report flag confusion matrices and precision/recall/F1 for the positive (`implicit`) class, with explicit and implicit supports. This makes errors on explicit gold visible instead of hiding them in overall accuracy.

## 4. Metrics and reporting hierarchy

### 4.1 Definitions

For a specified class, set, or one-vs-rest event:

- `precision = TP / (TP + FP)` — of predicted positives, the proportion correct.
- `recall = TP / (TP + FN)` — of gold positives, the proportion recovered.
- `F1 = 2 × precision × recall / (precision + recall)`.
- **Micro-F1** pools TP, FP, and FN over all segments/classes first, then computes F1. It emphasizes frequent items/classes.
- **Macro-F1** computes F1 separately for each defined class and takes the unweighted mean. It gives rare classes equal weight. For broad category macro-F1, use the six fixed categories; for sentiment, the three fixed classes; for a binary flag, both classes. State supports alongside the score. For a class with no gold and no predicted examples in a bootstrap replicate, define F1 as zero for the fixed-label macro average and flag this convention.
- **Exact-set accuracy** is the fraction of segments whose predicted multiset of four-field items equals the gold multiset exactly.
- **Cardinality error** is the per-segment absolute difference between the number of emitted items and gold quads, averaged over segments: `mean(abs(predicted_count - gold_count))`. Also report signed count bias as a secondary diagnostic.

If a precision or recall denominator is zero, report the value as zero for aggregate F1 computation and disclose zero-support cases. An empty predicted set against a non-empty gold set has zero precision under this convention, zero recall, and zero F1; its gold items are all FNs. An empty prediction and empty gold set is an exact-set success, although HAMOS annotated segments normally have at least one quad.

### 4.2 Primary and secondary metrics

**Primary metric:** end-to-end implicit category–sentiment **micro-F1** from Section 3.1. Report its precision and recall as co-primary diagnostics. This metric requires predicted `aspect_implicit=true` and exact category–sentiment agreement.

**Key secondary metrics:**

1. Gold-implicit-conditioned category–sentiment micro-F1 and category/sentiment component scores from Section 3.2.
2. End-to-end implicit category macro-F1 (six fixed broad categories) and sentiment macro-F1 (three classes), with support shown; rare category estimates are descriptive. The primary category–sentiment-pair endpoint remains micro-F1.
3. Complete structured-set micro precision/recall/F1 and exact-set accuracy.
4. Broad-category, sentiment, aspect-implicit, and opinion-implicit macro-F1; positive-class flag precision/recall/F1 and confusion matrices.
5. Cardinality MAE and signed bias.

For the end-to-end component macro-F1 diagnostics, first apply the primary endpoint's `aspect_implicit=true` prediction gate and gold implicit subset. Then independently perform exact-first, maximum-cardinality one-to-one matching on the single component being scored (category for category macro-F1; sentiment for sentiment macro-F1), preserving duplicates. A matched component error contributes one FP to the predicted class and one FN to the gold class; unmatched implicit predictions contribute class FPs and unmatched gold implicit items contribute class FNs. Average F1 over all six broad categories or all three sentiment labels, respectively. These component metrics are secondary diagnostics and do not replace or alter the exact category–sentiment matching used by the primary pair micro-F1.

For added per-language reporting, group eligible examples by the official segment language metadata, without adding language or IDs to model features. Report segment support, gold-quad support, gold-implicit support, and the same protocol metrics within each language subset. For language-level macro-F1 summaries, report a macro only when every fixed class has at least five gold examples in that subset; otherwise report it as unavailable (`null`), retain the per-class scores and supports, and state that the support threshold was not met. This five-example minimum is a reporting safeguard for language slices only; aggregate test metrics continue to use the fixed-label zero-support convention in Section 4.1.

Do not use accuracy alone as the primary measure: explicit and positive-majority predictions can produce deceptively high accuracy. Report counts and category/sentiment supports. The main text-only versus multimodal comparison uses identical eligible examples and the same metric implementation. Image-only remains separately labeled exploratory; its input ambiguity is quantified in [image-only feasibility audit](image_only_feasibility_audit.md) and its score is not a directly comparable primary modality effect.

## 5. Training assignment and objective proposal

### 5.1 Matching cost

For gold item `k` and predicted slot `j`, propose this assignment cost, using model probabilities before decoding:

`C(j,k) = 1.0 × CE_cat(j,k) + 1.0 × CE_sent(j,k) + 0.5 × BCE_aspect_implicit(j,k) + 0.5 × BCE_opinion_implicit(j,k)`

where categorical cross-entropy is negative log probability of the gold broad category/sentiment and binary cross-entropy is evaluated against the gold flag. Run Hungarian assignment between all `K` gold items and the 23 slots; the assignment is one-to-one and minimizes total cost. Objectness is trained separately and is not used to match a gold item to a slot. All text-only and multimodal runs use exactly this cost, weights, and assignment rule.

### 5.2 Matched-item and no-output losses

For matched slots, use the same four component losses and weights as the matching cost. Define:

- `L_matched = mean over gold items of [CE_cat + CE_sent + 0.5·BCE_aspect + 0.5·BCE_opinion]`.
- `L_present = mean BCE(objectness, 1)` over matched slots.
- `L_empty = mean BCE(objectness, 0)` over unmatched slots; set to zero when there are no unmatched slots.
- `L_total = L_matched + 1.0·L_present + 0.1·L_empty`.

The `0.1` coefficient downweights the many unused slots so they cannot dominate the matched-target learning signal. Averaging separately within matched and unmatched groups prevents the raw count of empty slots (up to 22 per segment) from scaling the loss. The proposal’s initial `no-output` loss is therefore an objectness negative loss on unmatched slots, not a 24th aspect-category class. If an implementation instead uses a categorical no-output class, it must preserve the same positive/negative balancing and be approved before coding.

### 5.3 Class imbalance

The initial baseline is **unweighted**: no class weights, focal loss, oversampling, or test-informed thresholds. This is the clean reference point and avoids unstable weights from very small neutral/implicit categories. Address imbalance in analysis by emphasizing the primary implicit endpoint, macro-F1, per-class support, confusion matrices, and uncertainty rather than by changing the test distribution.

If a sensitivity experiment is later justified, estimate any class weights using training labels only, cap/normalize them to avoid extreme rare-class weights, and select among the predeclared alternatives on dev only. Keep test untouched. The same selected loss configuration must be used for text-only and multimodal runs. Document whether a sensitivity run changes only the matched-target classification loss or also sampling; do not conflate these changes.

## 6. Synthetic examples

The examples use abbreviated tuples `(category, sentiment, aspect_implicit, opinion_implicit)` and assume one segment per row. `×2` means two distinct duplicate-valued items.

### A. Duplicate target values require duplicate predictions

Gold: `{(FACILITY, positive, false, false) × 2}`.

- Prediction with two identical items: 2 exact TPs; exact-set correct.
- Prediction with one item: 1 TP and 1 FN; exact-set incorrect; cardinality error 1.
- Prediction with three identical items: 2 TPs and 1 FP; exact-set incorrect; cardinality error 1.

The scoring is multiset-based, so duplicate quads are never collapsed into one label.

### B. Category/sentiment correct, implicitness wrong

Gold: `{(FACILITY, negative, true, false)}`.

Prediction: `{(FACILITY, negative, false, false)}`.

The gold-implicit-conditioned category–sentiment diagnostic scores the category and sentiment correctly because it ignores the predicted aspect flag. The primary end-to-end implicit set has no predicted implicit item: TP=0, FN=1, recall=0. Aspect-implicit classification records one FN (gold implicit predicted explicit). Opinion implicitness is correct and does not alter the primary endpoint.

### C. Unmatched extra prediction

Gold: `{(AMENITY, positive, false, false)}`.

Prediction: `{(AMENITY, positive, false, false), (SERVICE, negative, true, false)}`.

For complete-set scoring, the first item is TP and the extra item is FP; exact-set accuracy is false and cardinality error is 1. For the end-to-end implicit endpoint, the unmatched predicted `(SERVICE, negative)` is an implicit FP (no gold implicit target exists).

### D. Empty predicted set

Gold: `{(EXPERIENCE, positive, true, false)}`.

Prediction: `{}`.

There are no TPs or FPs and one FN. Precision is zero by the stated zero-denominator convention, recall and F1 are zero, exact-set accuracy is false, and cardinality error is 1. The implicit gold pair is a primary-endpoint false negative.

## 7. Confidence intervals and modality comparison

Use a **paired hotel-cluster bootstrap** for the primary text-only versus multimodal comparison:

1. Keep every test hotel as a cluster, including hotels with no implicit targets. Each cluster includes all its reviews, segments, and their complete gold/predicted sets.
2. For each replicate, sample the test hotel IDs with replacement, drawing the same number of hotels as in the original test set. Include all segments for each sampled hotel; a hotel drawn twice contributes its records twice.
3. Use the same sampled hotel list for the text-only and multimodal predictions, then recompute aggregate primary and secondary metrics and their difference (`multimodal − text-only`). This paired resampling preserves within-hotel dependence and estimates uncertainty in the performance difference.
4. Propose 2,000 replicates and percentile 95% intervals (2.5th and 97.5th percentiles), with the bootstrap seed recorded. Do not resample individual quads/segments independently.
5. Report raw test counts and supports beside intervals. The implicit subset has 116 quads across 70 hotels; intervals for rare categories and macro-F1 can be unstable. Mark unestimable/zero-support class behavior under the convention in Section 4.1 rather than presenting false precision.

For the same paired replicates, report the count and proportion where multimodal primary F1 is greater than text-only, the count of ties, and the count where text-only is greater. These are descriptive summaries of the existing paired replicate differences; they do not change the resampling or interval procedure and do not by themselves establish statistical significance.

Use the official hotel-disjoint test partition once after all choices are frozen. For any development-based threshold selection or loss sensitivity, use train/dev only. The bootstrap assesses test-sample uncertainty; it does not replace multiple-seed training variability, which should be reported separately if multiple runs are feasible.

## 8. Decisions requiring human approval

The following are proposed defaults, not approved implementation settings:

1. **Endpoint hierarchy:** approve end-to-end implicit category–sentiment micro-F1 as primary; gold-implicit-conditioned category/sentiment as a diagnostic; opinion implicitness and complete-set results as secondary.
2. **Evaluation assignment:** approve exact-match-first, then minimum-Hamming one-to-one pairing for full-set/component scores, and exact `(category, sentiment)` multiset matching for the end-to-end implicit endpoint.
3. **Decoding:** approve one global objectness threshold selected by development end-to-end implicit category–sentiment micro-F1 on the fixed 0.00–1.00 grid in 0.05 increments, with ties resolved to the highest threshold; keep both binary flag thresholds at 0.5.
4. **Initial loss weights:** approve category 1.0, sentiment 1.0, each implicitness flag 0.5, objectness positive 1.0, and unmatched-slot/no-output 0.1.
5. **Imbalance policy:** approve an unweighted baseline first and require dev-only evidence before any weighting or sampling sensitivity.
6. **Bootstrap:** approve 2,000 paired hotel-cluster replicates with percentile 95% intervals and the zero-support convention for macro-F1.
7. **Taxonomy:** approve excluding the optional 31-way target from the initial objective and primary metrics.

Regardless of these choices, the primary text-only and multimodal conditions must share the same 23-slot capacity, eligible examples, loss, decoding rule, and evaluation. Image-only remains exploratory and separate. No source, dataset, annotation, split, architecture, or training configuration change is proposed here.
