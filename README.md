# FRITNet

**FRITNet** is a compact, single-stream deep learning architecture for Facial Emotion Recognition (FER) in the wild. Designed for high parameter efficiency (**18.39M inference parameters**) without external facial landmark backbones or multi-model ensembles, FRITNet combines a truncated 3-stage IR-50 backbone with multi-scale spatial fusion and a 5-token Transformer encoder.

## Architecture Overview

* **Truncated IR-50 Backbone:** Extracts multi-scale facial representations from the first three stages (`x2`: $128 \times 28 \times 28$, `x3`: $256 \times 14 \times 14$) using MS-Celeb-1M pretrained weights and direct $112 \times 112$ Lanczos resizing.
* **TAP (Stage-2 Skip Connection) & LFA:** Projects `x2` via a $1 \times 1$ Conv + BN + ReLU with a zero-initialized learnable gate and fuses it into the Local Feature Augmentation (LFA) module right before the $2 \times 2$ quadrant split.
* **SAFM & 5-Token FRIT Transformer:** Applies spatial attention gating (`kernel_size=7`) and compresses the face into 5 tokens (1 global + 4 local quadrants, $d = 128$) processed by a 2-layer Transformer encoder.
* **Residual SiLU Gate:** Applies a zero-initialized multiplicative feature gate ($f + \gamma_{\text{silu}} \cdot (f \odot \text{SiLU}(Wf))$) before the linear classifier.
* **Zero-Overhead Training Stack:** Trained with Sharpness-Aware Minimization (SAM), Exponential Moving Average (EMA), Soft Contrastive Loss ($w = 0.05, \tau = 0.10$), and auxiliary global/local supervision heads—adding zero inference-time parameters.

## Benchmark Results

Official test-set results (test partitions are never accessed during training or checkpoint selection):

| Dataset | Inference Params | Accuracy (%) |
| :--- | :---: | :---: |
| **RAF-DB** | 18.39M | **91.40** |
| **FERPlus** | 18.39M | **87.03** |

## Installation & Verification

```bash
git clone https://github.com/karthik2207babu/FritNet.git
cd FritNet
pip install -r requirements.txt
```

Run the CPU unit tests to verify the environment:

```bash
python test_v43_unit.py
python test_v46_upgrade.py
python test_v46_upgrade_v2.py
```

## Dataset & Pretrained Weight Paths

By default, the scripts point to Google Drive paths (for Colab workflows), but all paths can be overridden via CLI flags or environment variables:

| Resource | Default Path in Code | How to Override |
| :--- | :--- | :--- |
| **IR-50 Weights** (`ir50.pth`) | `/content/drive/MyDrive/karthik_nitish/pretrain/ir50.pth` | `--ir50-weights /path/to/ir50.pth` |
| **RAF-DB** | `/content/drive/MyDrive/Datasets.zip` | `--raf-root /path/to/RAF-DB` or `--raf-zip /path/to/Datasets.zip` |
| **FERPlus Pixels** (`fer2013.csv`) | `/content/drive/MyDrive/Anjimoon/DATA SETS/fer2013.csv` | `export FERPLUS_PIXELS_CSV=/path/to/fer2013.csv` |
| **FERPlus Labels** (`fer2013new.csv`) | `/content/drive/MyDrive/fer2013new.csv` | `export FERPLUS_LABELS_CSV=/path/to/fer2013new.csv` |
| **Outputs / Checkpoints** | `/content/drive/MyDrive/karthik_nitish/<run_name>` | `--output-dir /path/to/output_dir` |

## Training

### RAF-DB

Uses natural-frequency sampling, label smoothing (`0.1`), Random Erasing, SAM (`rho=0.05`), EMA (`decay=0.999`), and Soft Contrastive Loss:

```bash
python train_v42.py \
  --dataset rafdb --preset control --backbone ir50 \
  --tap on --silu-gate --sampler none --label-smoothing 0.1 \
  --random-erasing --use-sam --sam-rho 0.05 \
  --use-ema --ema-decay 0.999 \
  --soft-contrast-weight 0.05 --soft-contrast-temperature 0.10 \
  --seed 42 --epochs 60 --patience 12 --batch-size 64 --num-workers 2
```

### FERPlus (Official 3-Way Split)

Trains on `Training`, validates on `PublicTest`, and leaves `PrivateTest` untouched (`--ferplus-split official`):

```bash
python train_v42.py \
  --dataset ferplus --ferplus-split official --preset control --backbone ir50 \
  --tap on --silu-gate --sampler none --hard-weight 0.0 \
  --random-erasing --re-prob 0.2 --re-max-scale 0.06 \
  --use-ema --ema-decay 0.999 \
  --soft-contrast-weight 0.05 --soft-contrast-temperature 0.10 \
  --seed 42 --epochs 60 --patience 12 --batch-size 64 --num-workers 2
```

## Official Evaluation

### RAF-DB Official Test Set

```bash
python evaluate_raf_official.py \
  --checkpoint /path/to/best_model.pth \
  --raf-root /path/to/RAF-DB
```

### FERPlus Official Test Partitions (`PublicTest` & `PrivateTest`)

```bash
export FERPLUS_PIXELS_CSV=/path/to/fer2013.csv
export FERPLUS_LABELS_CSV=/path/to/fer2013new.csv

python evaluate_ferplus_official.py \
  --checkpoint /path/to/best_model.pth \
  --split both
```