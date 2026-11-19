# HAMOS / ImplicitMABSA Research Pipeline

This repository contains an ongoing research implementation for multilingual hotel aspect-based sentiment analysis with a focus on implicit aspects. It includes dataset preparation and validation, several evolving modeling formulations, experiments, and their reports. Results from different formulations are not necessarily directly comparable; consult each report's task definition and split before interpreting metrics.

## HAMOS v4 dataset

The official HAMOS v4 dataset is published on [Mendeley Data](https://data.mendeley.com/datasets/x46f4y4fpr/4), DOI [10.17632/x46f4y4fpr.4](https://doi.org/10.17632/x46f4y4fpr.4). The record identifies version 4 and lists the dataset license as **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Cite the dataset and its authors when using it, follow the license attribution requirements, and check the release page for any version-specific notices.

Download the v4 archive from the official Mendeley Data record and extract the dataset files into the project root, preserving these paths:

```text
HAMOS/
├── annotations/
│   └── quads.jsonl
├── metadata/
│   ├── reviews.jsonl
│   ├── segments.jsonl
│   └── images.jsonl
├── splits/
│   ├── single_image/{train,dev,test}.jsonl
│   └── multi_image/{train,dev,test}.jsonl
├── taxonomy/
│   └── hotel_absa_taxonomy.xlsx
└── images/                         # optional local image files
```

The code reads these directories directly beneath the project root; there is no `data/` parent directory. `annotations/`, `metadata/`, `splits/`, and `taxonomy/` are required for dataset loading and validation. `images/` is optional for text-only work and required only when a selected experiment needs locally available photographs. Do not change the official split files.

The repository does **not** include the HAMOS dataset, downloaded images, cached features, model checkpoints, or pretrained model weights. Those files are excluded from Git. The public dataset package does not redistribute the original third-party photographs. `metadata/images.jsonl` contains image metadata and source URLs; any retrieval is subject to the original host's access terms and the rights of the photograph owners. The dataset's CC BY license does not grant rights to third-party photographs.

## Environment setup

Python **3.12** is the tested environment. Install project dependencies in an isolated environment:

```powershell
py -3.12 -m venv .venv-hamos
.\.venv-hamos\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include PyTorch, Transformers, NumPy, SciPy, Pillow, pandas, matplotlib, Jupyter, and Requests (used by the optional image retrieval utility). The dependency file currently specifies lower bounds rather than a complete lockfile; the exact verified GPU environment is recorded in the relevant readiness reports.

### Reproducing V4 on NVIDIA CUDA

V4 uses the pinned XLM-R base checkpoint `FacebookAI/xlm-roberta-base` at revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`. It is a text-based candidate-conditioned implicit category–sentiment presence model; V4 does not use CLIP image features. The local V4 training environment was verified with Python 3.12.10, PyTorch 2.11.0+cu128 (CUDA 12.8), and an NVIDIA GeForce RTX 3050 Laptop GPU with 6 GB memory.

For a compatible Windows CUDA installation, use the [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/) for PyTorch 2.11.0 / CUDA 12.8:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

The machine also needs a compatible NVIDIA driver. Check the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for current platform prerequisites. The V4 config uses `local_files_only: true`; make the exact XLM-R revision available in the local Hugging Face cache before running it. PyTorch's CUDA wheel supplies the CUDA runtime libraries; the installed NVIDIA driver must support the selected CUDA build.

## Dataset validation and preparation

From the project root:

```powershell
python -m src.validation_cli --root .
python -m src.quad_dataset_cli --root .
python -m src.span_diagnostics_cli --root .
```

These utilities write reports under `reports/`; they do not rewrite the source annotations or official splits. Open `notebooks/01_dataset_preparation.ipynb` for the dataset preparation walkthrough after obtaining the dataset files.

## Current V4 development result

The current V4 run completed five epochs. Its best checkpoint achieved **0.5933 unique category–sentiment pair micro-F1 on the development split** at the development-selected threshold. This is a development result, **not test performance**. The test split was not used to select this checkpoint or threshold. See [`reports/v4_training_progress.md`](reports/v4_training_progress.md) for epoch-level details.

The project remains ongoing research. V3 and V4 define a unique-pair presence task, while the earlier structured experiments predict duplicate-sensitive sets of quads; scores across these formulations must not be compared as if they shared the same target or metric.
