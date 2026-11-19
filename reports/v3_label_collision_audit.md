# V3 multi-hot label collision audit

**Scope:** official `single_image` train and dev split records only. This audit did not open HAMOS test examples, labels, or metadata. It read the train/dev split JSONL files, whose rows contain their segment text and quads. No source files, annotations, splits, configs, checkpoints, or caches were changed; no model was trained.

## Method

For each segment record, construct an 18-dimensional binary target from quads with `aspect_implicit == true`. The label order is the six broad categories crossed with the three sentiments: AMENITY, BRANDING, EXPERIENCE, FACILITY, LOYALTY, SERVICE; each crossed with negative, neutral, positive. A cell is 1 if at least one implicit quad in that segment has the category–sentiment pair. Explicit quads do not activate labels.

Text normalization is deliberately conservative: decode the original split text, normalize Unicode to NFC, collapse each run of whitespace to one ASCII space, and trim leading/trailing whitespace. Preserve case, punctuation, token spelling, and word order. Group records by this normalized text alone; language and IDs are not part of the grouping key. The same normalization is used for the train/dev overlap count. A separate raw-string check produced the same overlap count.

“Conflicting group” means that at least two segment records in the text group have different 18-bit target vectors. Percentages are shown over all unique text groups and, separately, over groups with multiple records. Group-size distributions count unique text groups by number of segment records. A “collapsed quad” is an extra implicit quad beyond the first occurrence of the same category–sentiment pair **within one segment**; this is exactly the multiplicity discarded by a multi-hot target.

## Text-group collisions

| Split | Segment records | Unique normalized text groups | Groups with multiple records | Conflicting-vector groups | Conflicts / all groups | Conflicts / multi-record groups |
|---|---:|---:|---:|---:|---:|---:|
| Train | 8,816 | 8,676 | 87 | 0 | 0.00% | 0.00% |
| Dev | 1,895 | 1,888 | 6 | 0 | 0.00% | 0.00% |

No within-split repeated-text group has differing 18-label targets. Therefore there are no conflicting examples or segment IDs to list. This includes repeated groups whose records all have the all-zero implicit target vector.

### Group-size distribution

| Records in group | Train groups | Dev groups |
|---:|---:|---:|
| 1 | 8,589 | 1,882 |
| 2 | 58 | 5 |
| 3 | 19 | 1 |
| 4 | 6 | 0 |
| 5 | 1 | 0 |
| 6 | 1 | 0 |
| 9 | 1 | 0 |
| 10 | 1 | 0 |

Train has 87 multi-record groups (1.00% of its unique text groups); dev has 6 (0.32%). The largest normalized text group contains 10 train segment records.

## Multi-hot support by label

“Positive segments” is the number of segment records with at least one implicit quad. “Positive label cells” is the sum of the 18 per-label positive segment counts. Since each segment has one binary cell per label, repeated quads of the same pair do not add positive cells.

| Split | Positive segments | Positive label cells | Implicit quads before collapse | Quads collapsed by multi-hot encoding | Segments with a repeated pair |
|---|---:|---:|---:|---:|---:|
| Train | 370 | 451 | 547 | 96 | 84 |
| Dev | 77 | 98 | 119 | 21 | 17 |

| Implicit category–sentiment label | Train positive segments | Dev positive segments |
|---|---:|---:|
| AMENITY / negative | 0 | 1 |
| AMENITY / neutral | 1 | 0 |
| AMENITY / positive | 8 | 2 |
| BRANDING / negative | 2 | 0 |
| BRANDING / neutral | 0 | 0 |
| BRANDING / positive | 6 | 0 |
| EXPERIENCE / negative | 17 | 4 |
| EXPERIENCE / neutral | 1 | 0 |
| EXPERIENCE / positive | 148 | 30 |
| FACILITY / negative | 12 | 3 |
| FACILITY / neutral | 1 | 0 |
| FACILITY / positive | 153 | 34 |
| LOYALTY / negative | 3 | 0 |
| LOYALTY / neutral | 0 | 0 |
| LOYALTY / positive | 55 | 11 |
| SERVICE / negative | 4 | 1 |
| SERVICE / neutral | 0 | 0 |
| SERVICE / positive | 40 | 12 |
| **Total positive label cells** | **451** | **98** |

The label support is highly imbalanced: several train labels have only one to four positive segments, and four labels have no development positives (including BRANDING / positive). The support table describes segment-level presence, not quad counts. Training weights, if later approved, must be derived from train targets only; dev support is for evaluation and threshold/model selection, not weight calculation.

## Duplicate text across the hotel-disjoint boundary

There are **37 normalized text groups** present in both official train and dev. These groups account for 79 train records and 40 dev records. The raw-string exact comparison also found 37 groups with the same record counts. All 37 cross-split overlapping text groups have the same 18-label target vector on the train and dev sides; none show a target-vector conflict. No IDs or example texts are included here. The official splits remain untouched.

This is consistent with hotel-disjoint partitions: repeated text can occur across hotels even when hotels do not overlap. The overlap is small relative to both splits, but should be disclosed because exact repeated text may make development performance somewhat optimistic for recurring boilerplate. It is not evidence of hotel leakage, and this audit did not change split membership.

## Language metadata and ambiguity

Language metadata could help an encoder represent English and Vietnamese differently, and retaining the approved inference-time language feature remains reasonable. However, language would **not resolve an actual collision in these data**: no repeated normalized text group within train or dev has conflicting target vectors, and the train/dev overlapping text groups also agree on targets. More generally, language can disambiguate identical text only when the conflicting records differ in language; it cannot resolve conflicting annotations for the same text in the same language. This audit did not join language metadata because it was not needed to establish the target-vector collision counts.

## Conclusion and recommendation

- **Is segment-level multi-hot presence classification reasonable?** Yes, as a separate formulation for predicting whether each implicit category–sentiment pair is present. Repeated text does not create conflicting target vectors in train or dev.
- **Are conflicting targets common enough to undermine it?** No conflicts were observed in either split, so duplicate text does not undermine the formulation in the audited data. There are 37 exact normalized text groups crossing train/dev, all with matching target vectors; this is a small development-overlap caveat to disclose.
- **Does language metadata resolve ambiguity?** No resolution is needed for the observed collisions. Language is still useful as an input for multilingual representation, but it is not a general solution to contradictory labels attached to identical text.
- **Should V3 proceed?** Proceed with the proposed multi-hot model only as an explicitly approved **unique category–sentiment presence** task. The material limitation is multiplicity: 96 train and 21 dev implicit quads collapse because the same category–sentiment pair repeats within a segment. The head cannot reproduce those counts. Approve a matching unique-pair evaluation protocol before implementation; retain the existing multiset model and its results unchanged for the original task.

No model, code, or data changes were made. No training was run. Stop here pending approval.
