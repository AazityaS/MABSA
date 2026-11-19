# Evaluation Reporting Gap Fixes

**Status:** Reporting implementation updated and synthetic regression checks passed. No test-set evaluation was run.

## Original gaps and implementations

### 1. End-to-end implicit category and sentiment macro-F1

Added `end_to_end_implicit_components.category` and `.sentiment` to `evaluate_structured`.

- Both use only emitted predictions with predicted `aspect_implicit=true` and gold quads with `aspect_implicit=true`, preserving the primary endpoint's implicitness gate.
- Category and sentiment are evaluated independently as multisets, using exact-first maximum-cardinality one-to-one assignment on the component being scored. Duplicate labels remain separate items.
- A wrong assigned component label adds one FP to its predicted class and one FN to its gold class. Unmatched predicted implicit items add an FP for their component class; unmatched gold implicit items add an FN for their class.
- Macro-F1 averages over the fixed six aspect categories or three sentiment labels. The primary category–sentiment pair micro-F1 implementation and counts are unchanged.
- Per-class gold/predicted supports and class ordering accompany each macro score.

### 2. Gold-implicit-conditioned diagnostic confusion matrices

Added category and sentiment matrices under `gold_implicit_conditioned.confusion_matrices`.

- The matrices use the existing diagnostic assignment over **all** emitted predictions and gold implicit quads; predicted aspect implicitness and opinion implicitness do not affect matching.
- Rows are gold labels. Columns are predicted labels followed by `MISS` for unmatched gold items.
- An assigned wrong label increments its gold-row/predicted-column cell. Every unmatched gold item increments its row's `MISS` cell. Unmatched predictions are ignored, as required by Section 3.2.
- Duplicate items increment cells independently.
- The evaluator builds category and sentiment display order from the actual mappings supplied by `StructuredDataset.label_mappings`; no new label ordering is introduced.

### 3. Per-language results and support

`structured_evaluate.evaluate_checkpoint` now collects each eligible segment's language from its existing dataset metadata and passes that language to reporting only. It does not add language, hotel ID, review ID, or segment ID to model inputs. The two modes must also agree on language order as well as segment IDs, gold targets, and hotels.

Each language result includes:

- Eligible segment count and total/gold-implicit quad counts.
- Primary implicit category–sentiment TP, FP, FN, precision, recall, and F1, plus explicit predicted/gold set denominators.
- End-to-end implicit category and sentiment component metrics and supports.
- Gold-implicit-conditioned category/sentiment scores, exact-pair rate, and confusion matrices.
- Other protocol metrics returned by `evaluate_structured`, including complete-set, component/flag, and cardinality results.

**Support convention added for language slices:** a language-level fixed-label macro-F1 is withheld (`null`) unless every class in that fixed label set has at least **five gold examples**. Its per-class scores and supports remain in the output, with a status explaining why the macro is withheld. This is a reporting safeguard for language-stratified subsets only; aggregate macro-F1 retains the protocol's fixed-label, zero-support-as-zero convention. The five-example threshold is an explicit reporting convention for small slices, not a change to the primary metric.

### 4. Paired-bootstrap win/tie/loss summary

The existing paired hotel resampling, seed, replicate count, primary metric, intervals, and mean difference are unchanged. Each existing replicate now also increments one of:

- `multimodal_better_replicates` when multimodal primary F1 is higher;
- `ties` when primary F1 values are equal;
- `text_only_better_replicates` when text-only primary F1 is higher.

`proportion_multimodal_better` is wins divided by the number of replicates. The same sampled hotel IDs still contribute to both model scores within every replicate.

## Ambiguities identified and clarifications

The approved protocol specified end-to-end component macro-F1 but did not define whether component labels should share pair matching or be matched independently. The implementation uses independent one-to-one matching per component after the primary implicitness gate. This lets category macro-F1 give category credit independently of sentiment, and vice versa, while preserving multiplicity and penalizing unmatched items. It does not alter the primary exact category–sentiment pair metric.

The protocol requested diagnostic confusion matrices but did not prescribe an encoding for missed gold items. The implementation adds an explicit `MISS` prediction column and omits unmatched predictions, following the diagnostic's stated missed-gold/ignored-prediction treatment.

The protocol did not specify language-level aggregation or a small-support macro policy. Language metrics are calculated over the official eligible segment subset for each segment-language value. Class supports use gold labels in that subset; the five-gold-example-per-class rule above withholds only language-level macros when support is low. No language-specific threshold, matching rule, or model input is introduced.

The requested bootstrap win share is a descriptive proportion over the already specified paired replicates. It is not a significance test and does not itself support a claim of statistical significance.

These additions clarify reporting for secondary metrics. The primary endpoint, its threshold, matching, loss, class order, hotel-cluster bootstrap sampling, official splits, checkpoints, and training configuration are unchanged.

## Files updated

- `src/structured_metrics.py`
- `src/structured_evaluate.py`
- `tests/test_structured_protocol.py`
- `reports/evaluation_and_loss_protocol.md` (additive clarifications for confusion-matrix encoding, component macro-F1, language-slice support, and bootstrap win proportions)

## Tests

Using `.venv-hamos-cuda\Scripts\python.exe`:

- Structured protocol synthetic tests: **16 passed**.
- Label mapping regression tests: **2 passed**.
- Cache path regression tests: **2 passed**.
- `py_compile` for `src/structured_metrics.py` and `src/structured_evaluate.py`: passed.

The complete project test suite was **not** run: `tests/test_structured_dataset.py` instantiates the official test split and reads its quad annotations in `setUpClass`. Running that file would violate the explicit constraint against accessing test examples/labels while implementing these changes. The listed 20 tests are the complete safe synthetic/mapping/cache-path subset; no official split data was read by these tests.

## Data and experiment integrity

- No test examples, test labels, test predictions, or test metrics were accessed.
- The final evaluation CLI was not run.
- Checkpoints, checkpoint thresholds, training configuration, annotations, labels, and official split files were not modified.
- No training, test inference, or image-only evaluation was performed.
