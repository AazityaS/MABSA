# HAMOS Pre-training Readiness Audit

**Audit date:** 2026-11-18  
**Scope:** Static review of the structured-model pipeline, runtime/CUDA inspection, and data/configuration checks. No full training, test-set evaluation, pretrained feature extraction, source-data edit, or annotation/split edit was performed.

## Executive result

The structured implementation is broadly consistent with the approved design, and static checks show that all training targets fit within the 23 output slots. The current environment is **not ready for the requested GPU smoke test or full training**: the installed PyTorch is CPU-only even though an NVIDIA RTX 3050 Laptop GPU and driver are available. In accordance with the CUDA gate, pretrained checkpoints were not loaded and no feature extraction or GPU training batch was attempted. Establish an isolated supported Python/CUDA environment first, then rerun the GPU smoke tests before full training.

## Critical blockers

1. **CUDA-enabled PyTorch is absent.** `torch.cuda.is_available()` is false and `torch.version.cuda` is `None`; the structured feature and training commands explicitly request CUDA. Do not proceed to feature extraction or training in this environment.
2. **Real pretrained/GPU smoke tests remain unverified.** The exact XLM-R and CLIP checkpoints were not loaded. The local Hugging Face cache has XLM-R configuration/tokenizer artifacts but no model weights, and no CLIP checkpoint cache. Consequently, pretrained output dimensions/associations, peak GPU memory, elapsed time, and a real GPU optimizer update are unavailable. The existing synthetic CPU smoke test is not evidence for these properties.

No demonstrated source-code defect requiring a code change was found in this audit.

## CUDA environment

| Item | Observed value |
|---|---|
| Python | 3.15.2, 64-bit (MSC v.1951) |
| Python executable | `C:\Python315\python.exe` |
| PyTorch | `2.15.0+cpu` |
| PyTorch CUDA build | `None` |
| `torch.cuda.is_available()` | `False` |
| PyTorch CUDA device count | `0` |
| NVIDIA driver (`nvidia-smi`) | 591.86 |
| Driver-reported CUDA compatibility | 13.1 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| GPU memory | 6,144 MiB total; 1,365 MiB used; 4,637 MiB free at audit time |

The current Python 3.15 environment is outside the official Windows support range stated by the PyTorch install page, which currently lists Python 3.10–3.14. Use an isolated Python 3.12 environment. The official PyTorch selector currently lists a CUDA 12.8 Windows wheel; the NVIDIA driver is newer and the CUDA runtime is supplied by the PyTorch wheel. The command below follows the official Windows/Pip/CUDA 12.8 selector:

```powershell
py -3.12 -m venv .venv-hamos-cuda
.\.venv-hamos-cuda\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

The official selector notes the Windows support range and instructs users to select Windows, Pip, and the CUDA version suited to their machine; it also documents `torch.cuda.is_available()` as the verification check: [PyTorch Get Started Locally](https://pytorch.org/get-started/locally/).

After the environment check returns CUDA available, run the project synthetic smoke and unit suite, then a small real checkpoint/cache and GPU optimizer smoke before launching either full training run. No command in this report was executed to install packages.

## Pretrained checkpoints and feature extraction

The baseline config pins:

- XLM-R: `FacebookAI/xlm-roberta-base`, revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.
- CLIP: `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.

Both revisions are explicit and are passed through the Hugging Face loaders. **Load result: not attempted**, because the active PyTorch build fails the user-required CUDA gate. No pretrained feature cache was produced or dimension/segment association check performed in this audit. The existing cache-extraction implementation separates text features keyed by `segment_id` and vision features keyed by `image_id`; the feature manifest records revisions, dimensions, IDs, and device. Static inspection did not reveal an ID-key mismatch in the training view, which joins review candidate image IDs to their cached image vectors.

Static code inspection confirms both encoders are set to evaluation mode, have gradients disabled, and run under inference mode. This is a code-path finding, not a runtime confirmation that the pinned weights load successfully.

## Structured pipeline review

### Assignment, duplicates, and variable target counts

- The loss uses SciPy’s Hungarian assignment between emitted slots and gold items, with category, sentiment, aspect-implicitness, and opinion-implicitness costs using the configured weights `1.0`, `1.0`, `0.5`, and `0.5`.
- Assignment is one-to-one. It operates on item positions rather than deduplicated labels, so equal-valued gold items remain separate and can be assigned to separate slots.
- Rectangular assignment accommodates variable target counts; unmatched slots receive no-output/objectness supervision and unmatched gold items cannot be represented if the slot count is insufficient.
- The metric implementation separately applies the approved exact-first structured matching rule and preserves multiset behavior. The gold-implicit diagnostic uses its own category/sentiment assignment and does not condition on predicted implicitness or opinion implicitness.

### Loss normalization and no-output behavior

- Matched-item component losses are averaged across matched targets. Positive objectness loss is averaged over matched slots; unmatched-slot objectness loss is averaged over unmatched slots and multiplied by the configured `0.1` empty-slot weight. Thus the large number of unused slots does not simply accumulate as an unnormalized sum.
- Empty-gold examples have no matched-item term and use the empty-slot objectness term. The current HAMOS split has annotated target quads, so this is chiefly a defensive path.
- **Minor stability risk:** training uses CUDA autocast in FP16 but does not use `GradScaler`. This is not a demonstrated failure, but loss scaling is a useful safeguard against FP16 gradient underflow. Confirm finite losses/gradients in the required small GPU smoke; consider enabling scaling only if the smoke demonstrates instability, without changing the approved objective.

### Slot capacity and data representation

The training official split contains 8,816 segments; an independent count of the source training split found a maximum of **23 quads per segment**, and **zero** segments exceed 23. The model config also specifies 23 slots. Thus every training target is representable without truncation under the current split and schema. The data view retains each quad as a separate target item, including duplicate-valued items.

### Feature caching, candidate pooling, and associations

- Static inspection shows text cache rows are keyed by segment ID and image vectors by image ID; candidate metadata IDs are used to retrieve vectors, with segment association carried by the dataset view. No gold quad primary/supporting image IDs are used as model inputs.
- Candidate pooling applies a masked mean across all available candidate image embeddings. Multi-image reviews are therefore supported; candidate sets are not reduced to a single arbitrary image.
- The primary structured dataset eligibility rule excludes segments whose review has an empty accessible candidate pool. The model code can accept an all-false image mask and yields a zero pooled vector before the image projection, but an empty-pool segment is excluded by the primary view. Empty-pool behavior should be covered explicitly in the GPU/cache smoke if a future run opts into that view.
- The established matched primary comparison has identical eligible segment IDs and output/evaluation protocol for text-only and multimodal conditions. The five image-unavailable training quad instances are absent from the primary eligible view; the full-coverage text view remains a separate option documented in the implementation guide.

### Frozen encoders and transfers

Static code review confirms the encoder `.eval()`, `requires_grad_(False)`, and inference-mode paths. Feature vectors are moved to the selected device by the training batch path, and the structured head is trained from cached feature tensors. Mixed precision is scoped around head forward/loss computation. These transfers and autocast behavior were **not runtime-tested on CUDA** because CUDA is unavailable to the active PyTorch build.

### Metrics, threshold, and bootstrap

- The primary endpoint is the approved end-to-end implicit category–sentiment micro-F1. The separate gold-implicit-conditioned diagnostic uses its own assignment and counts a gold implicit item as a hit on exact category and sentiment irrespective of predicted aspect implicitness; this is kept distinct from the primary metric.
- A single global objectness threshold is selected from the configured fixed grid on development data only, maximizing the primary endpoint and resolving ties deterministically toward the higher threshold. Binary implicitness flags retain the fixed `0.5` threshold. Evaluation loads the selected threshold from the checkpoint rather than selecting it on test.
- Paired hotel-cluster bootstrap samples hotel IDs with replacement and applies the same sampled hotel sequence to the text-only and multimodal predictions. This preserves pairing and within-hotel clustering.
- The code paths are consistent with the approved protocol in `reports/evaluation_and_loss_protocol.md`; no test evaluation was run as part of this audit.

### Best checkpoint and reproducibility

- Training selects/records the best checkpoint using the development primary implicit endpoint and applies early stopping on development performance. The global objectness threshold is selected on development per evaluation epoch and saved with the best checkpoint.
- Python, NumPy, and PyTorch seeds are set from the saved config; the config contains the model dimensions, optimizer settings, epochs, thresholds, loss weights, and pinned checkpoint revisions. The feature manifest records the resolved checkpoint revisions and cache identifiers.
- **Minor reproducibility limitations:** deterministic-algorithm/CuDNN settings are not enabled; cache identity does not include a complete source-data, preprocessing-code, and library-version fingerprint; logs append and the same output path can overwrite a prior best checkpoint; and optimizer/scheduler state is not saved for exact mid-run resume. Preserve run artifacts and environment package versions, and use fresh output directories for each run.

## Data, split, and implementation checks

The existing dataset summary reports:

| Split | Text-only eligible quad instances | Multimodal eligible quad instances | Missing-image quad instances excluded from multimodal |
|---|---:|---:|---:|
| Train | 16,789 | 16,784 | 5 |
| Dev | 3,600 | 3,600 | 0 |
| Test | 3,606 | 3,606 | 0 |

The five unavailable-image quad instances are retained in the text-only dataset view; they are not silently deleted from source annotations. The summary records all original quads preserved in the text view, unchanged source fingerprints, and zero hotel overlap for every split pair. Static inspection confirms candidate images are selected from review-level metadata rather than gold image associations. No annotation, split, or source dataset file was changed in this audit.

## Smoke-test results

| Check | Result |
|---|---|
| Existing synthetic CPU smoke (prior project validation) | Passed for synthetic forward/loss/backward/checkpoint mechanics; does not load pretrained encoders |
| Existing unit/invariant suite (prior project validation) | 16/16 passed |
| New real pretrained feature extraction | Not run: active PyTorch has no CUDA build |
| Encoder feature dimensions and cached associations | Not runtime-verified in this audit |
| GPU training-batch loss/backward/optimizer update | Not run: `torch.cuda.is_available()` is false |
| Peak GPU memory and elapsed time | Not measured |
| Full training / test evaluation | Not run |

## Recommended commands for the two primary training runs

Once the isolated CUDA environment is installed and verified, the approved sequence is:

```powershell
# Extract pinned frozen encoder features once for both primary conditions.
python -m src.structured_features --root . --config configs/structured_baseline.json --output cache/structured_features --device cuda

# Primary run 1: text-only segment-level structured prediction.
python -m src.structured_train --root . --config configs/structured_baseline.json --cache-dir cache/structured_features --output-dir runs/structured --mode text --device cuda

# Primary run 2: multimodal segment-level structured prediction, same eligible examples/protocol.
python -m src.structured_train --root . --config configs/structured_baseline.json --cache-dir cache/structured_features --output-dir runs/structured --mode multimodal --device cuda
```

These commands are **recommendations only and were not run**. Run the small real pretrained/cache and one-batch GPU optimizer checks before starting these full training jobs. Image-only remains an exploratory weak baseline and is not part of this matched primary pair.

## Readiness decision

**Not yet ready for full training in the current environment.** The static pipeline review found no demonstrated critical logic bug, and all training targets fit the 23-slot representation. The required gate is a supported isolated Python environment with CUDA-enabled PyTorch, followed by successful real pretrained loading, feature-cache association/dimension verification, and a small GPU forward/backward/optimizer update. At that point, recheck memory use on the 6 GB GPU before starting the approved primary runs.
