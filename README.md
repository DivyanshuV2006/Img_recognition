Img_recognition

A lightweight image recognition project for training, evaluating, and running inference on image classification models. Built with Python, PyTorch, OpenCV, and scikit-learn. Includes scripts and/or notebooks for end‑to‑end experimentation.

Badges:
Python PyTorch OpenCV scikit--learn Jupyter
Key features

    Train a CNN for image classification (from scratch or via transfer learning)
    Reproducible training with configurable hyperparameters
    Evaluation with accuracy, precision, recall, F1, and confusion matrix
    Simple prediction script for single images or folders
    Jupyter notebooks for exploration and visualization
    Clean project structure and easy setup

Project structure

If your repository differs, adjust paths accordingly.

text

Img_recognition/
├─ src/                     # Python modules and scripts
│  ├─ train.py              # Training entry-point
│  ├─ evaluate.py           # Evaluation entry-point
│  ├─ predict.py            # Inference on images
│  ├─ datasets.py           # Data loading & augmentation
│  ├─ models.py             # Model definitions / transfer learning
│  └─ utils.py              # Helpers (metrics, logging, etc.)
├─ notebooks/               # Jupyter notebooks
│  └─ exploration.ipynb
├─ data/                    # Your dataset (see “Data layout” below)
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ runs/                    # Checkpoints, logs, and results (created at runtime)
├─ requirements.txt
└─ README.md

Setup

    Clone

Bash

git clone https://github.com/DivyanshuV2006/Img_recognition.git
cd Img_recognition

    Create a virtual environment (choose one)

Bash

# venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

    Install dependencies

Bash

pip install --upgrade pip
pip install -r requirements.txt

If you don’t have a requirements.txt yet, a minimal starting point might be:

txt

torch torchvision torchaudio
opencv-python
scikit-learn
numpy
pandas
matplotlib
tqdm
jupyter

Data layout

Place your images in class‑named folders:

text

data/
├─ train/
│  ├─ class_a/  img1.jpg, img2.jpg, ...
│  └─ class_b/  img3.jpg, ...
├─ val/
│  ├─ class_a/  ...
│  └─ class_b/  ...
└─ test/
   ├─ class_a/  ...
   └─ class_b/  ...

You can add more classes; folder names become class labels.
Quickstart

Train (update script/args to match your repo):

Bash

python src/train.py \
  --data-dir data \
  --train-subdir train \
  --val-subdir val \
  --model resnet18 \
  --img-size 224 \
  --batch-size 32 \
  --epochs 20 \
  --lr 1e-3 \
  --device cuda

Evaluate:

Bash

python src/evaluate.py \
  --data-dir data/test \
  --checkpoint runs/best.ckpt \
  --img-size 224 \
  --batch-size 32 \
  --device cuda

Predict on a single image:

Bash

python src/predict.py \
  --checkpoint runs/best.ckpt \
  --img path/to/image.jpg \
  --img-size 224 \
  --device cpu

Tip: If your scripts are at the repo root (e.g., train.py instead of src/train.py), drop the src/ prefix in the commands above.
Notebooks

    Launch Jupyter:

Bash

jupyter notebook

    Open notebooks/exploration.ipynb (or your notebook) to visualize samples, try augmentations, and run quick experiments.

Configuration (optional)

If you prefer a YAML/JSON config, create something like config.yaml:

YAML

data_dir: data
train_subdir: train
val_subdir: val
test_subdir: test
model: resnet18
img_size: 224
batch_size: 32
epochs: 20
lr: 0.001
optimizer: adam
device: cuda

Then run:

Bash

python src/train.py --config config.yaml

Results and artifacts

    Checkpoints: saved under runs/ (e.g., best.ckpt, last.ckpt)
    Metrics: CSV or JSON logs under runs/
    Plots: loss/accuracy curves, confusion matrix images
    Inference: predictions with per‑class probabilities

Add example images or a small table once you have results:

text

Top‑1 Accuracy: 93.4%
Precision/Recall/F1 (macro): 0.92 / 0.93 / 0.92

Tips

    Use transfer learning for small datasets:
        resnet18/resnet50, mobilenet_v2, efficientnet_b0, etc.
    Data augmentation helps: random crop, flip, color jitter, normalization
    Tune: batch size, learning rate, scheduler, epochs
    Track experiments with TensorBoard or CSV logs

TensorBoard example:

Bash

pip install tensorboard
tensorboard --logdir runs --port 6006

Roadmap

    Add pretrained model selection via CLI
    Automatic dataset download (optional)
    Export to ONNX/TorchScript
    Simple web demo (Streamlit/Gradio)
    Unit tests and CI



    scikit‑learn for metrics and utilities
    OpenCV for image I/O and preprocessing
