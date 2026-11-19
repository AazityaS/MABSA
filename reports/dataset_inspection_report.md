# HAMOS dataset inspection and preparation notes

## Schema verified in this checkout

- `metadata/reviews.jsonl`: one row per review with `source_review_id`, numeric `hotel_id`, `review_text`, `languages`, official `split`, and `emoji`.
- `metadata/segments.jsonl`: one row per segment with `segment_id`, `source_review_id`, `segment_text`, language, context flag, and character offsets.
- `annotations/quads.jsonl`: one row per quad with aspect/opinion terms and spans, broad `aspect_category`, fine-grained `taxonomy_code`, sentiment, implicitness flags, `primary_image_id`, and `supporting_image_ids`.
- `metadata/images.jsonl`: image ID, source review, URL, caption, order, hash, and local path.
- Official split rows contain `instance_id`, review and segment IDs, text, and a list of quad projections. Supporting image IDs are available in the master annotation file, so the loader joins annotations by segment ID and uses the master quads as the full record.

The primary image belongs to an individual quad. The dataset representation therefore remains segment → list of quads → each quad's primary/supporting IDs and paths. Encoded label IDs are added alongside unchanged source strings.

## Taxonomy verification

The supplied workbook has `README`, `TAXONOMY`, `DECISION_RULES`, and `ASPECT_SUMMARY` sheets. Its taxonomy sheet defines 31 fine-grained codes. Prefix mapping verified against the workbook: `FAC_` → FACILITY, `AM_` → AMENITY, `SER_` → SERVICE, `EXP_` → EXPERIENCE, `BRA_` → BRANDING, `LOY_` → LOYALTY. The broad category and fine-grained code mappings are separate and are saved in `label_mappings.json`.

## Verified dataset shape

| Item | Count |
|---|---:|
| Hotels / reviews / segments | 3,399 / 8,796 / 12,601 |
| Images / quads | 9,219 / 23,995 |
| Train segments / quads | 8,816 / 16,789 |
| Dev segments / quads | 1,895 / 3,600 |
| Test segments / quads | 1,890 / 3,606 |
| Train segments with multiple quads | 4,274 |

The validation run confirms every split segment appears exactly once, split rows preserve every master quad, and there is no hotel overlap. All 9,216 downloaded images exist and pass Pillow verification; three report-listed images are unavailable. Five quads across three training segments refer to those unavailable primary images. There are no corrupt downloaded files, unknown sentiment/taxonomy labels, broken image metadata references, or validation errors. Twenty-six explicit terms have empty token span fields in the source annotations (mostly emoji opinions and a few non-English aspect spans); the validator reports these as warnings and preserves the source values.

The validation JSON also records source split SHA-256 fingerprints to make later changes detectable. It does not modify the official split files.

## Missing image behavior

Missing image associations remain in the sample. Each quad records `primary_image_available=False`, retains the image ID and path, and has `None` in the aligned `quad_images` list. In collated batches, `quad_mask` marks real quads (including those with missing photos), while `image_mask` indicates which quad images are available. This distinguishes a missing image from padding and retains all annotation labels. A strict all-images-required experiment may filter explicitly and must record excluded quad and segment IDs.

## Output formulation options

1. **Multi-label category and sentiment prediction:** useful for a quick segment-level baseline, but it loses aspect-opinion pairing and cannot fully represent repeated categories with different opinions.
2. **Structured quad extraction:** predicts aspect/opinion spans or implicitness, broad category, fine-grained code, sentiment, and optionally quad-specific image grounding. This preserves the annotation unit and directly supports implicit-aspect evaluation, but requires a more involved structured decoder.
3. **Hierarchical categories:** predict one or more of six broad categories, then predict the 31-code taxonomy within each relevant category. This respects the taxonomy but still needs per-quad slots or extraction to represent multiple opinions and sentiments.

**Recommended first formulation:** retain one output record per quad, with parallel per-quad category, taxonomy code, sentiment, aspect implicitness, and opinion implicitness predictions. Start with a modest set-prediction or span-plus-label head; evaluate implicit and explicit aspect subsets separately. Keep the text-only baseline first, then use the same per-quad representation with quad-level image features. Avoid collapsing segment labels or requiring all quads in a segment to share one image.

## Runtime note

This development environment has Pillow but lacks PyTorch, torchvision, transformers, and Jupyter. The full image/reference validator and lazy schema-level dataset inspection were run locally. The notebook contains dependency installation and actual tokenizer/DataLoader tensor verification cells; those require the declared dependencies and a tokenizer download. No model training was started.
