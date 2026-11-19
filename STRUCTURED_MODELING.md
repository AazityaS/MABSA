# HAMOS structured-model implementation

This implementation provides the two primary segment-level structured prediction runs: text-only and multimodal. Both use the same 23-slot output schema and the same train/dev/test segment eligibility. Image-only remains out of the immediate run plan because its repeated review-level image inputs map to conflicting segment target sets.

## Checkpoints and configuration

`configs/structured_baseline.json` pins:

- Text: `FacebookAI/xlm-roberta-base`, revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.
- Vision: `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.

The frozen encoders produce CLS text vectors and CLIP image vectors. Cache extraction processes text and images separately, records resolved revisions in `manifest.json`, and stores features by segment ID/image ID. It caches text features for all 12,601 segments, including the three image-unavailable training segments, so a future full-coverage text-only evaluation is possible. Primary training views still use only eligible examples. The model adds a language embedding to projected text. Multimodal mode masked-means every available candidate image vector for that review, projects it to the hidden size, concatenates it with the projected text/language representation, then applies a fusion projection, normalization, and activation before the learned 23-slot head. Text-only retains its original text pathway. No gold quad-image association is read into inputs. Taxonomy prediction is not implemented.

## Eligibility and data

`StructuredDataset` constructs candidate pools only by joining image metadata to `source_review_id`, keeps every quad target as a separate set element, and excludes a segment only when its review has no accessible candidate images. That exact rule applies to both primary modes: train 8,813 segments (three excluded), dev 1,895, test 1,890. The target counts are 16,784/3,600/3,606. Text-only is not enlarged for this immediate matched comparison. Metadata IDs stay outside model features.

For a future coverage-only text result, instantiate `StructuredDataset(..., mode="text", require_candidate_images=False)`; this retains the three image-unavailable training segments and all 16,789 training quads. This option is not used in the primary experiment commands above.

## Environment and commands

Install project requirements, then ensure PyTorch is a CUDA-enabled build supported by the installed NVIDIA driver. The feature extractor and trainer refuse an explicit CUDA request if `torch.cuda.is_available()` is false; they do not silently switch devices. Use small encoder batches (2 initially) on the 6 GB GPU and inspect `nvidia-smi` before increasing them. The model head trains from cached features with batch size 32 by default.

```powershell
# 1. Extract frozen XLM-R and CLIP features once (shared by both runs)
python -m src.structured_features --root . --config configs/structured_baseline.json --output cache/structured_features --device cuda

# 2. Train only after approval; these commands are provided for reproducibility
python -m src.structured_train --root . --config configs/structured_baseline.json --cache-dir cache/structured_features --output-dir runs/structured --mode text --device cuda
python -m src.structured_train --root . --config configs/structured_baseline.json --cache-dir cache/structured_features --output-dir runs/structured --mode multimodal --device cuda

# 3. Evaluate selected checkpoints once; paired 2,000-replicate hotel bootstrap
python -m src.structured_evaluate --root . --cache-dir cache/structured_features --text-checkpoint runs/structured/text/best_checkpoint.pt --multimodal-checkpoint runs/structured/multimodal/best_checkpoint.pt --output runs/structured/test_evaluation.json --device cuda
```

Threshold selection and early stopping use dev only. The checkpoint records the selected global objectness threshold. Binary flag decoding remains fixed at 0.5. The test evaluator loads those values and does not tune on test. Evaluation asserts that both model runs use identical test segment IDs and target sets.

## Validation commands

```powershell
python -m unittest discover -s tests -v
python -m src.structured_smoke
```

Before full extraction, the bounded real-checkpoint CUDA smoke and extraction-rate estimate can be run with the isolated CUDA environment:

```powershell
.\.venv-hamos-cuda\Scripts\python.exe -m src.structured_gpu_smoke --root . --config configs/structured_baseline.json --sample-size 8 --batch-size 2 --output reports/gpu_smoke_sample.json
```

This loads the pinned revisions, extracts only eight training texts and eight metadata-derived candidate images, checks associations, and performs one head optimizer update. It does not create the full feature cache, train a full model, or use test annotations.

The smoke command uses synthetic frozen-feature tensors and a randomly initialized head; it validates forward/backward/loss/checkpoint mechanics only, not the downloaded pretrained encoders. The dataset tests check counts, missing-image eligibility, metadata-derived candidate sets, hotel separation, target preservation, and official source fingerprints.

## Evaluation details

See `reports/evaluation_and_loss_protocol.md` for the exact assignment, loss weights, metrics, threshold grid, and bootstrap definition. See `reports/image_only_feasibility_audit.md` for why image-only is excluded from the primary comparison.
