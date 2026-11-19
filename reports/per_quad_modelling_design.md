# Initial per-quad classification design

## Instance definition

Use one supervised example for each annotation quad in the official `single_image` split. A segment with three quads creates three instances with the same original segment text and language, and each instance keeps its own quad ID and primary/supporting image association. The source segment and annotation records are not rewritten.

## Inputs and targets

The model input surface is `inputs` only:

- `text`: original segment text, tokenized with XLM-R when configured.
- `image`: the primary image tensor for this quad in multimodal mode; absent in text mode.
- `language`: segment language metadata.

The target surface is separate:

- `aspect_category`: one of six broad category classes.
- `sentiment`: positive, negative, or neutral.
- `aspect_implicit` and `opinion_implicit`: binary classes.
- `taxonomy_code`: optional 31-class fine-grained target.

Target strings are retained in `target_labels` for reporting, while integer IDs are in `targets`. Aspect terms, opinion terms, categories, sentiments, and taxonomy codes are never placed in `inputs`; term strings are not model targets in this initial formulation. IDs and image association fields are metadata, not features.

## Missing primary images

The text view contains every quad instance. The multimodal view filters only instances whose own required primary image is unavailable and records their quad, segment, and image IDs in `excluded_instances`. This checkout has five such training quads, all pointing to the three unavailable image IDs. Dev and test have no missing primary images. If one quad in a multi-quad segment were missing while another had a valid image, only the unavailable quad would be excluded from the multimodal view.

## Verified instance counts

| Split | Text instances | Multimodal instances | Excluded for missing primary | Implicit aspect instances | Segments with multiple quads | Quad instances from those segments |
|---|---:|---:|---:|---:|---:|---:|
| Train | 16,789 | 16,784 | 5 | 547 | 4,274 | 12,247 |
| Dev | 3,600 | 3,600 | 0 | 119 | 911 | 2,616 |
| Test | 3,606 | 3,606 | 0 | 116 | 923 | 2,639 |

The exact label distributions are in `quad_dataset_summary.json`, separately for text and multimodal views and for each split. The summary also records source-file SHA-256 fingerprints.

## Limitations and evaluation notes

- The quad-level classifier predicts category, sentiment, and implicitness after an annotation quad is already given. It does not extract aspect/opinion spans or discover how many quads a segment contains.
- Several instances from one segment share identical text. They must remain grouped for any future resampling or analysis; do not randomly split quad instances across folds. The official hotel-disjoint splits remain the unit of train/dev/test separation.
- Positive sentiment and FACILITY are much more common than other labels; implicit aspects are rare. Report Macro-F1 and implicit/explicit performance separately, with per-language results where counts support them.
- The optional fine-grained target is hierarchical with the broad category. Validate predictions for consistency if both heads are used.
- Multimodal results use a slightly smaller training set because five quad associations lack primary images. Compare image and text results on the same eligible instances when attributing gains to vision, and report the full text-only result separately.
- The image is quad-specific. Reusing one segment image for all of its quads would discard the annotation's grounding distinctions.

No model has been trained.
