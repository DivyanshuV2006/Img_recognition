# Img_recognition

This repository contains code for building, training, evaluating, and exporting image classification models (ResNet / EfficientNet variants). It includes data preprocessing helpers, model definitions, a training loop, simple inference utilities, and containerization / environment manifests for reproducible runs.

## What's included

- `train.py` — training loop, dataset split, optimizer/scheduler setup and checkpoint export.
- `main.py` — CLI to build datasets, run training, or run ONNX-based inference.
- `model.py` — model assembly and ONNX export helpers (ResNet example with a custom head).
- `ResNet.py`, `EfficientNet_V2_M.py` — model variant definitions (if you want alternatives).
- `data_processing.py` — dataset builder, transforms, and DataLoader-friendly Dataset class.
- `utils.py` — helpers: pickle I/O, image preprocessing for inference, ONNX utilities.
- `LLM.py` — optional/experimental language-model related utilities (not required for core training).
- `requirements.txt` and `environment.yml` — dependency manifests.
- `Dockerfile` — Docker image to run the project (CPU image by default).

## Quick facts / contract

- Input: images organized into subfolders per class (see `data_processing.py` and `CreateLabels` behaviour).
- Output: ONNX-exported model files in `utils/` by default (see `save_model` in `model.py`).
- Success: you can train a model with `train.py` and run inference with `main.py --img <path>` (after exporting to ONNX or using the PyTorch model conversion in `model.py`).

## Prerequisites

- Recommended: Linux, Python 3.10 (the repository was developed with Python 3.10).
- For GPU training, use a CUDA-enabled PyTorch build (see PyTorch installation matrix). The provided `Dockerfile` builds a CPU-only image — for GPU use prefer the official PyTorch CUDA images or an NVIDIA CUDA base image.

## Install (conda)

1. Create and activate the conda environment:

```bash
conda env create -f environment.yml -n imgrec
conda activate imgrec
```

2. Or install with pip (lightweight):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Note: The `requirements.txt` pins CPU-compatible PyTorch. If you need GPU support, follow the official PyTorch installation instructions to select the correct wheel for CUDA.

## CLI / Usage

main.py implements a small CLI. The available flags (from the current code) are:

- `--img`  : path to an input image to run ONNX inference.
- `--build`: build and pickle the dataset (saves to `utils/training_data.pkl` by default).
- `--train`: run the training loop (uses `train()` defined in `train.py`).
- `--num_epochs`, `--lr` : training hyperparameters (default `100`, `0.01`).

Examples:

Build dataset (one-off):

```bash
python main.py --build
```

Train model (runs training loop defined in `train.py`):

```bash
python main.py --train --num_epochs 30 --lr 0.001
```

Run ONNX inference on a single image (ensure `utils/model.onnx` exists or export a model first):

```bash
python main.py --img /path/to/image.jpg
```

If you prefer to run `train.py` directly, call it from Python (it exposes `train()` which expects `data_path` and `labels`). See `train.py` for the default hyperparameters and batch size.

## Docker

Build (CPU-only image):

```bash
docker build -t imgrec:latest .
```

Run (open a shell in the container with data mounted):

```bash
docker run -it --rm -v /local/data:/data imgrec:latest /bin/bash
```

Notes:

- This Dockerfile installs the Python dependencies and some system libraries required by OpenCV.
- For GPU-based training use an official PyTorch CUDA image or an NVIDIA CUDA base image and install GPU-compatible PyTorch/torchvision inside it. Example (outside this repo):

```bash
# Example only — pick the correct tag for your CUDA/toolkit
docker run --gpus all -it --rm pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime /bin/bash
```

## Dependencies

Key Python packages required by this code:

- torch (pinned in `requirements.txt`)
- torchvision
- onnx, onnxruntime (for export and inference)
- opencv-python (cv2 used in data builder)
- pillow, numpy, tqdm

See `requirements.txt` for the exact pinned versions used during development.

## Implementation notes & gotchas

- `data_processing.py` uses OpenCV + PIL interop. If images are large, consider streaming transformations instead of building a single in-memory list of all (image, label) pairs.
- `model.save_model()` currently exports an ONNX file. It uses opset_version=12. If you need newer opset features, update the export call accordingly.
- `main.py` has a hard-coded Windows-style `data_path` string — update it for your environment before running `--build` or `--train`.
