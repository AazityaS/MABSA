# HAMOS Feature Extraction and Cache Validation

**Date:** 2026-11-19  
**Result:** Full CUDA feature extraction completed; all requested cache validations passed. No training or test-set model evaluation was started.

## Extraction run

The configured cache destination did not exist before extraction, so no prior cache was overwritten. The requested command ran from the HAMOS root using `.venv-hamos-cuda\Scripts\python.exe` and completed successfully:

```powershell
$py = '.\.venv-hamos-cuda\Scripts\python.exe'
& $py -m src.structured_features --root . --config configs/structured_baseline.json --output cache/structured_features --device cuda
```

The process reported `Caching 12601 segment features and 9216 unique candidate images on cuda.` It then loaded the pinned XLM-R and CLIP vision checkpoints and wrote:

`cache/structured_features/9072ce0b71cd3470/`

The cache contains `text_features.pt`, `image_features.pt`, and `manifest.json`. The serialized feature files are dictionaries keyed by segment ID and image ID, respectively; the manifest records the sorted ID lists, dimensions, cache key, device, and checkpoint revisions. Requested and resolved revisions match the baseline config. The cache reports `device: cuda` and `feature_inputs_only: true`.

## Feature inventory and numerical validation

| Check | Text features | Image features |
|---|---:|---:|
| Expected / actual feature count | 12,601 / 12,601 | 9,216 / 9,216 |
| Expected / actual vector dimension | 768 / 768 | 512 / 512 |
| Duplicate feature dictionary IDs | 0 | 0 |
| Duplicate IDs in manifest list | 0 | 0 |
| Missing expected IDs | 0 | 0 |
| Unexpected IDs | 0 | 0 |
| Non-finite feature vectors | 0 | 0 |
| Manifest ID list exactly matches feature keys | Yes | Yes |

The cache key `9072ce0b71cd3470` matches the configured checkpoint revisions, maximum text length, and image size.

## Independent candidate-image association validation

The validator reconstructed image pools independently from the official split files, `metadata/reviews.jsonl`, `metadata/images.jsonl`, and actual local image paths. It joined each split segment's `source_review_id` to image metadata, retained only image records whose resolved local paths exist, and compared each resulting per-segment candidate ID set with the candidate pool exposed by `StructuredDataset`.

- Official split segment IDs were unique: **yes**.
- Duplicate image IDs in image metadata: **0**.
- Segments checked: **12,601**.
- Segment-to-candidate-image associations checked: **13,854**.
- Per-segment candidate-pool mismatches: **0**.
- Missing local paths among the 9,216 expected available images: **0**.
- Reconstructed available image IDs absent from the feature cache: **0**.
- Gold `primary_image_id` and `supporting_image_ids` were not used to construct or validate model inputs.

The cache deliberately stores a segment-text feature map and a global image-feature map rather than copying candidate pools into the feature tensors. At training time, each segment's review-level candidate IDs are joined through metadata to the corresponding image-feature keys. The independent validation confirms that all 13,854 available segment/image associations resolve through that join.

## Primary mode eligibility check

The text-only and multimodal primary views were independently instantiated for each official split. Their eligible segment ID sets matched exactly, and all required features were present for both views.

| Split | Text-only eligible segments | Multimodal eligible segments | Same IDs | Missing text features for eligible IDs | Missing image features for eligible candidate IDs |
|---|---:|---:|---|---:|---:|
| Train | 8,813 | 8,813 | Yes | 0 | 0 |
| Dev | 1,895 | 1,895 | Yes | 0 | 0 |
| Test | 1,890 | 1,890 | Yes | 0 | 0 |

The three training segments with empty accessible candidate pools are present in the 12,601-item text feature map but excluded from both matched primary experiment views under the approved shared eligibility rule. This validation did not change that rule or any split.

## Integrity and scope

The validation used feature-key counts and exact set comparisons, tensor shape checks, per-vector `torch.isfinite` checks, metadata-to-filesystem reconstruction, and separate text/multimodal dataset views. Overall validation status: **PASS**.

No annotations, labels, split files, or evaluation definitions were modified. No full training, image-only training, or test-set evaluation was run. The workflow stops here for review before training.
