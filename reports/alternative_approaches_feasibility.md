# HAMOS Alternative Implicit Prediction Approaches: Feasibility Review

## Executive summary

V3-Frozen's failure is primarily a **ranking/separability failure**, not just a threshold-grid problem. Its epoch-1 linear head over frozen 768-dimensional XLM-R `<s>` features scored global ROC-AUC 0.2949 and average precision 0.00184 on dev, below the 0.5 ROC-AUC reference and 0.00287 positive-cell prevalence. An exact score sweep moved micro-F1 only from 0.005730 to 0.005737 and still emitted almost every possible label. Reweighting the loss may change the fitted linear decision surfaces, but cannot make the frozen representation richer. A threshold change alone is not a credible remedy.

The three approaches answer different questions. **Partial XLM-R fine-tuning (C)** has the strongest technical rationale because it can adapt the representation that currently ranks positives poorly. **Hierarchical prediction (A)** is well matched to the many all-zero segments and the distinction between “any implicit aspect” and “which pairs,” but the conditional training set is small and hard gating can lose all downstream recall. **Positive-weighted frozen classification (B)** is the fastest, lowest-risk diagnostic, and should be the first small experiment; it is not the approach with the strongest expected ceiling.

The most useful immediate experiment is a single separately saved, train-only-weighted frozen-head run. It takes minutes rather than hours, directly tests whether positive loss weighting helps under the same embeddings, and can determine whether a more costly partial fine-tune is worth attempting. Do not treat a favorable dev result as independent test evidence.

## Evidence base and task boundary

V3 predicts presence of each of the 18 unique broad-category × sentiment pairs among aspect-implicit quads. A segment has a multi-hot target; identical pairs repeated in the segment collapse to one bit. This is intentionally different from the original duplicate-sensitive quad/set prediction. Its micro-F1 must not be compared directly with the historic structured-model F1.

Measured training/development facts from the existing V3 audits and artifacts:

| Quantity | Train | Dev |
|---|---:|---:|
| Segments | 8,816 | 1,895 |
| Segments with at least one implicit aspect | 370 | 77 |
| All-zero target segments | 8,446 | 1,818 |
| Implicit quad occurrences | 547 | 119 |
| Positive unique segment-label cells | 451 | 98 |
| Repeated pair occurrences collapsed | 96 | 21 |

There are 34,110 possible dev segment-label cells and 98 positives: prevalence 0.287%. Four labels have no positive train examples: AMENITY / negative, BRANDING / neutral, LOYALTY / neutral, SERVICE / neutral. In dev, nine labels have zero positive support. Some supported labels have only one to four positive dev segments. The label-collision audit found no conflicting vectors in normalized repeated-text groups; 37 normalized text groups occur in both train and dev with matching target vectors. Train/dev are hotel-disjoint. These facts favor preserving the multi-label unique-pair target, but make estimates for rare labels and dev model selection noisy.

The failed frozen model used the pinned XLM-R-base checkpoint at revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, cached 768-dimensional last-layer `<s>` vectors, one `Linear(768,18)` head, unweighted `BCEWithLogitsLoss`, AdamW learning rate 0.001, and the configured global coarse threshold grid. The best epoch was 1. The exact dev sweep's best threshold was 0.002701, with precision 0.002877, recall 1.0 and F1 0.005737; it still emitted 34,068/34,110 label cells. The broader probability audit showed positive-cell mean 0.0594 versus negative-cell mean 0.0873, 100-bin class-conditional histogram overlap 0.6998, and poor ranking. Thus lowering the threshold cannot rescue the current scores.

## Comparison at a glance

| Approach | Main weakness addressed | Fit to target/data | Expected potential | Implementation and GPU burden | Main risk / time judgment |
|---|---|---|---|---|---|
| **A. Hierarchical presence → conditional pair prediction** | All-zero dominance and the fact that pair identity is meaningful only when any implicit aspect exists | High: binary any-presence head plus 18-label multi-label conditional head preserves multi-aspect and unique-pair outputs | Medium to high if presence is learnable; representation remains a bottleneck unless encoder is tuned | Medium to high; encoder can be shared; GPU resembles fine-tuning if end-to-end | Hard stage-one gate can make pair recall collapse; stage two sees only 370 train segments. Not a safe six-hour full run. |
| **B. Positive-weighted frozen XLM-R + linear head** | BCE's low positive contribution under 0.287% dev cell prevalence | High: same 18 unique-pair multi-hot target; repeated pairs still collapse | Low to medium; cheap test of loss imbalance, but current ranking is below chance | Low; cache/head already exist; GPU use and training duration are negligible compared with XLM-R fine-tuning | Weighting may only shift score intercepts or create false positives; cannot enrich frozen features. Best immediate diagnostic. |
| **C. Partially fine-tuned XLM-R** | Frozen `<s>` representation is not separating positive from negative label cells | High: same text inputs, target vector, pair semantics, and evaluation | Highest technical chance of the three because it can adapt the upper representation | Medium; freeze lower layers, train upper layers + head. RTX 3050 6 GB needs small micro-batches and memory checks | Overfit risk with 370 positive segments; throughput and WDDM allocator uncertainty; likely too long for a short deadline. |

## Approach A — Hierarchical prediction

### Formulation and data suitability

Stage 1 predicts `contains_any_implicit_aspect` for every segment. This is derived from whether the existing 18-dimensional target has any positive bit: 370/8,816 train segments (4.20%) and 77/1,895 dev segments (4.06%) are positive. This is much less sparse than 451 positive cells among 158,688 train segment-label positions (0.284%). A binary head can learn the presence decision using all official training segments.

Stage 2 should remain **18-label multi-label classification**, trained on the positive train segments, with each label denoting a unique category–sentiment pair. A single softmax over 18 classes is inappropriate because a segment can contain multiple implicit aspects with different categories and/or sentiments. A one-label-per-segment classifier would silently change the target. Repeated identical pairs remain collapsed as in approved V3; this hierarchy cannot restore multiplicity and should continue to be evaluated only with unique-pair presence metrics.

Conditioning on positive segments changes the Stage 2 training set to only 370 examples. It reduces the all-zero burden but does not remove rare-label scarcity: some category–sentiment labels have one or a few examples, and four labels have no positive train support at all. No stage-two architecture can learn a positive decision for a label with zero train positives without changing the data or importing unsupported assumptions.

### Training and inference

An implementable variant is a shared XLM-R encoder with a binary presence head and an 18-logit pair head. Train the presence head on every segment; train the pair head only on segments with at least one implicit target, with a multi-label BCE objective. Alternatively, independent classifiers allow different training schedules but duplicate encoder cost. A single shared encoder is more practical.

At inference, **do not use gold presence**. A hard stage-one gate (`if presence < threshold: return no pairs`) makes every stage-one false negative an unrecoverable pair false negative. A safer composition is to emit joint pair scores such as `P(any) × P(pair | any)` and select one global threshold on development unique-pair micro-F1. This preserves the existing unique-pair output and metric while allowing uncertain presence to attenuate, rather than categorically erase, pair scores. A second option is hard gating, but its gate threshold must be chosen jointly with end-to-end pair F1 on dev and its recall cost reported. The final procedure must be predeclared; never use teacher-forced presence at evaluation.

### Imbalance, complexity, and risks

Stage 1 can use unweighted BCE as a baseline and report its PR curve; a train-only positive weight may be tested separately. Stage 2 still has sparse positives, so train-only per-label `pos_weight` can be used with explicit caps, while unsupported classes receive no invented positive examples. Loss weights between stages need an explicit predeclared choice. A strong Stage 1 with a weak Stage 2 merely transfers the error; a weak Stage 1 gates away correct pair predictions. Joint scoring can also be miscalibrated because Stage 2 trains only on positive segments.

**Existing components reusable:** `ImplicitPresenceDataset`, 18-label construction and label ordering in `src/v3_presence_dataset.py`; XLM-R model/pooling in `src/v3_presence_model.py`; train-only class-weight helper, optimizer, mixed precision and accumulation in `src/v3_presence_train.py`; `evaluate_presence` and global threshold selection in `src/v3_presence_metrics.py`. New heads, conditional loss masking, joint-score decoder/threshold evaluation, tests, config and isolated run directory are required. Do not modify the frozen model/checkpoint.

**GPU/time estimate:** If the shared encoder is fine-tuned, its per-epoch cost will be close to full V3; the second head is negligible. If both heads use cached embeddings, the experiment is fast but retains the same frozen-representation limitation. No hierarchical model was benchmarked or trained. The all-XLM-R estimate below applies as a planning bound, not a measured hierarchy runtime.

## Approach B — Positive-weighted frozen classifier

### What weighting addresses—and what it cannot

Unweighted elementwise BCE averages over 18 cells per segment while only a small fraction are positive. The learned head can reduce loss substantially by assigning low probability to most cells, even if the few positives are not ranked well. A positive weight multiplies the positive term of BCE and makes missed positives more costly. It can change the learned linear coefficients, not merely the inference threshold.

However, the diagnostic found global ROC-AUC 0.2949 and AP 0.00184, below chance/prevalence references. This points to poor score ordering as well as poor calibration. Reweighting may change the ordering after refitting the head, but a post-hoc threshold or prior shift cannot repair fundamentally non-separable features. For a well-specified logistic model, class weighting can act largely like an intercept/prior shift, so a threshold adjustment can approximate some of its effect; finite samples, regularization, and shared multi-label optimization may alter this, but it is not guaranteed.

### Weight options and calibration

The existing full V3 trainer already implements per-label `pos_weight = min((N - positives) / positives, cap)` from training targets only, with cap 20; zero-train-support labels receive weight 1 and are explicitly marked as negative-only. The frozen trainer does not currently apply this option, but it can reuse the existing selective cache reader, aligned target dataset, linear head and metrics in a separate variant.

An important consequence of the actual supports: all 14 labels with any positive train support have fewer than 420 positive segments, so the ratio is above 20 and the current cap makes their weights **all 20**. Four labels have zero train support. Therefore, with cap 20, per-label weights and a global `pos_weight=20` produce the same positive-term multiplier for every supported label; the unsupported labels have no positive terms either way. A more elaborate per-label weighting comparison is not informative under this support distribution unless a different, justified cap/formula is predeclared.

Focal loss is not the first choice: it adds a focusing parameter and can over-emphasize mislabeled or intrinsically ambiguous rare positives, while making calibration harder. Logit/prior adjustment is also not a ranking fix; estimates must come from train only, and it can push false positives sharply upward. First measure whether a simple capped positive weight improves dev ranking and global micro-F1.

Keep one **global** threshold to retain the approved V3 evaluation formulation and avoid unstable per-label thresholds on supports as low as one or zero. The previous coarse grid missed the low-score region; a separately approved weighted run should predeclare a denser global grid covering near-zero probabilities, or use an exact sweep as a diagnostic and clearly label its threshold-selection optimism. Do not choose one threshold per label: rare/unsupported dev labels make those estimates particularly unstable.

**Existing components reusable:** cached 768-dimensional features and their provenance; `FrozenPresenceDataset`; `FrozenXLMRPresenceHead`; V3 label supports; `compute_train_pos_weight` in `src/v3_presence_train.py` as the source of train-only support logic; `evaluate_presence` with unique-pair semantics. New separate weighting config/run and tests are needed. No encoder or data changes are needed.

**GPU/time estimate:** Measured V3-Frozen loop duration was 3.26 seconds for nine epochs, with 20.3 MB CUDA allocated; full process was about 13 seconds including cache/setup. One weighted-head run should remain seconds to a few minutes including process startup, but this is an estimate, not a measured weighted run. It is comfortably the most feasible approach before a near deadline.

## Approach C — Partially fine-tuned XLM-R

### Proposed scope

Freeze embeddings and the lower eight of XLM-R-base's 12 transformer blocks; train the upper four blocks and the 18-logit head. If GPU memory or speed is unacceptable, a separately specified two-upper-block variant is a more conservative fallback, but do not mix both in one run. Keep the same multi-hot unique-pair targets, text-only inputs, seed policy, and dev-only model selection. No IDs, hotel/language metadata, aspect terms, opinions, images, or gold-derived inputs are needed.

This is the approach that directly addresses V3-Frozen's below-chance ranking: the representation can adapt to the implicit category–sentiment distinction instead of fitting only a linear separator over static `<s>` vectors. Freeze behavior must be verified: lower parameters have `requires_grad=False`, frozen blocks are kept in eval mode so dropout does not drift, gradients exist for the upper blocks/head, and optimizer parameter groups contain only trainable parameters. Use a lower learning rate for the upper encoder than for the classifier head; this is a proposed configuration, not an observed optimal value.

### GPU and time evidence

Measured environment: NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB VRAM. The readiness smoke loaded the exact XLM-R revision, used FP16/GradScaler, and measured 1.224 seconds per warmed batch of two on **12-token padded** examples. Extrapolating 4,408 micro-batches gives approximately 90 minutes of train batches per epoch, plus about 19 minutes for a dev pass at that same short-sequence rate. The readiness report estimates 9–15 hours for 5–8 epochs and up to about 36 hours for 20 epochs. The smoke's reported PyTorch allocator peaks exceeded reported physical device capacity despite completing; this unresolved Windows/WDDM memory accounting means the max-length run is not proven safe.

Additional read-only tokenization using the locally cached pinned tokenizer measured train length mean 18.24 tokens, median 15, P90 33, P95 42, max 182; dev mean 18.41, median 15, P90 32, P95 41, max 130. No train/dev text exceeded the configured 256-token truncation limit. These lengths are still above the 12-token smoke input, and batch padding, activation checkpointing, mixed precision, and actual optimizer throughput affect runtime. The length summary is measured; any extrapolation from it is an estimate.

Freezing lower blocks should reduce backward activation storage and backward compute, but it does not eliminate the forward pass through all 12 blocks. With 8 blocks frozen and only 4 trainable, a rough compute model suggests around 55–75% of full fine-tune train-step cost, but this ratio is **not benchmarked** and ignores checkpointing overhead. A cautious planning range is about 1–3 hours per epoch including dev, with substantial uncertainty; 4–8 epochs may take roughly 4–24 hours. Even the low end may leave little time for repeated development, report generation, and any troubleshooting. A real representative-length smoke (e.g. batches with padding near P90, not max length alone) is mandatory before committing.

On 6 GB, start with micro-batch 1–2, gradient accumulation to the desired effective batch, FP16 with GradScaler, and gradient checkpointing on the trainable upper blocks if required. Sequence length 256 is configured but observed maximum train length is 182, so dynamic padding will avoid 256-token work for typical batches. A smaller multilingual encoder could be faster, but no alternative checkpoint is approved or benchmarked here; switching tokenizers/checkpoints adds provenance and comparability risk and is not a good deadline shortcut.

**Existing components reusable:** approved train/dev dataset and 18-label target encoding; pinned model and tokenizer; current mixed precision, GradScaler, gradient accumulation, gradient clipping, AdamW, threshold selection, and metric code. New partial-freeze model/trainer configuration, trainable-layer assertions/tests, and a fresh isolated run/checkpoint directory are required. Existing full V3 trainer currently fine-tunes the entire encoder; its `gradient_checkpointing` option is not a layer-freezing option.

## Ranked recommendation

### By expected technical benefit

1. **C — partial XLM-R fine-tuning.** Best chance of improving ranking because it can change the representation; still high uncertainty due tiny positive support and overfitting.
2. **A — hierarchical prediction.** Directly models presence versus pair identity and removes many all-zero examples from Stage 2, but pair support remains small and hard gating can suppress recall.
3. **B — weighted frozen head.** Cheapest, directly tests loss imbalance, but cannot adapt the representation and current scores are below-chance ranked.

### By deadline feasibility

1. **B — weighted frozen head.** Reuses the cache and takes negligible GPU time; best first experiment.
2. **A with frozen features.** Computation is modest, but adds architecture/loss complexity without addressing the measured poor frozen representation. A fine-tuned shared-encoder version moves to the slow category.
3. **C or A with a trainable shared encoder.** Highest technical upside but uncertain memory and multi-hour epochs; not a reliable near-deadline commitment.

## Minimal next experiment: isolated weighted frozen head

If approved, create one new config and run directory; do not touch the original frozen run or fine-tuned V3 plan.

1. Reuse only the validated cache, 8,816 official train targets, and 1,895 official dev targets; assert hotel-disjointness and feature/target ID alignment.
2. Keep the 768→18 linear architecture, seed 1729, optimizer settings, train/dev split and unique-pair target unchanged. Change only the loss to `BCEWithLogitsLoss(pos_weight=20)` (equivalent to the existing cap-20 per-label rule for all supported train labels; record four train-unsupported labels and their negative-only supervision).
3. Run a tiny finite-loss/gradient smoke and one separately saved head-only training run. No test rows or features are opened.
4. Select one global threshold using development unique-pair micro-F1 with a predeclared dense global grid that includes low probabilities. Report the exact-threshold sweep as a diagnostic, plus global ROC-AUC/AP, precision/recall/F1, TP/FP/FN, exact-set accuracy, predicted-label-count distribution, and all-zero/support caveats.
5. Compare only against V3-Frozen on the same dev split and target definition. Treat a dev gain as a reason to consider C, not as test evidence or proof of generalization.

This experiment should take minutes rather than hours; implementation, isolation checks, tests, execution and reporting likely fit within roughly 1–2 hours. That is a planning estimate. If positive weighting does not improve ranking (not just thresholded F1), do not spend the remaining deadline on more frozen-head weighting variants; make a deliberate decision whether the partial fine-tune's uncertain runtime is acceptable.

## Direct answers

1. **Best chance to improve the current result:** Approach C, because the evidence points to poor frozen-feature ranking rather than a threshold-only issue. It remains unproven and carries overfit/runtime risk.
2. **Most feasible within the available time:** Approach B. It reuses the cached embeddings and linear head and can be trained/evaluated in minutes.
3. **Implement first:** Approach B as the minimal evidence-gathering run. If it improves dev ranking and F1, that supports proceeding to a more expensive representation-adaptation experiment; if not, weighting is unlikely to be a useful deadline fix.
4. **Is 50–60% F1 plausible?** It is not impossible as a long-term research result, but it is an **unsupported expectation** for this dataset/run/deadline. The 0.287% dev cell prevalence, sparse/unsupported labels, current below-chance ranking, and no successful unique-pair model provide no evidence for that target. Historical multiset F1 values do not establish it because the tasks differ.
5. **Smallest useful experiment:** One train-only cap-20 weighted frozen-head run on the existing 768-d cache, with an isolated output directory, followed by global dev threshold selection and ranking metrics. This tests imbalance handling in minutes without committing to XLM-R fine-tuning.

## Assumptions and unresolved uncertainties

- The near-deadline context is treated as a short remaining window; the exact remaining wall-clock time was not available from the attached request.
- Partial fine-tuning throughput is estimated, not benchmarked. GPU memory accounting in the earlier smoke is unresolved, and maximum-length training was not tested.
- The reported token lengths use the locally cached pinned tokenizer on official train/dev texts only; they do not predict padded batch length or throughput exactly.
- A positive-weighted frozen head may change its ranking after refitting, but no such variant has been run. No expected F1 improvement is asserted.
- The primary V3 unique-pair presence target deliberately collapses repeated identical category–sentiment pairs; none of these approaches restores quad multiplicity.
- Four labels have no train positives, and nine have no dev positives; per-label and macro metrics are limited by support.
- Train/dev normalized duplicate texts and hotel-disjointness facts are taken from the existing collision audit; official split files remain unchanged.

## Read-only verification

This review inspected only named documentation, V3-Frozen/V3 source and configuration files, cached tokenizer metadata, and official train/dev text lengths. It did not open the test split or read test labels, predictions, embeddings, or metrics. It did not run training/evaluation, download models or data, install packages, or write changes to existing artifacts. The only project file created for this task is this report.
