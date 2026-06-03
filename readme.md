# Advancing Deepfake Detection in High-Compression Scenarios

A configuration-driven framework for studying how image compression degrades
deepfake detection, and whether fingerprint removal combined with contrastive
learning can reduce that degradation. The codebase trains and evaluates two
detectors that share a single, swappable backbone:

- **Baseline** — a conventional binary classifier trained on higher-quality
  (c23) images.
- **Novel** — a Siamese model that adds a push/pull contrastive objective over
  raw-versus-compressed image pairs, designed to make the backbone treat an
  image and its compressed version as equivalent.

The backbone is selected by name from the config file, so swapping
EfficientNet-B0 for Xception, DenseNet121, or any other `timm` model is a
one-line change with no edits to the training or evaluation code.

---

## 1. Repository layout

```
project_root/
├── configs/
│   ├── base_config.yaml        # main experiment configuration
│   └── smoke_config.yaml       # tiny config for a fast end-to-end test
├── preprocessing/
│   ├── __init__.py
│   ├── helpers.py              # apply_compression(), remove_noise()
│   ├── create_manipulations.py # builds the six derived image sets
│   └── create_manifests.py     # builds the CSV manifests + method normaliser
├── src/
│   ├── __init__.py
│   ├── dataset.py              # StandardDataset, QuadrupletDataset
│   ├── models.py               # UniversalBackbone, NovelSiameseWrapper
│   ├── train.py                # trains baseline then novel model
│   └── testing.py              # evaluates both models on RAW + Compressed
├── utils/
│   ├── __init__.py
│   └── config_loader.py        # loads and validates the YAML config
├── data/                       # (not committed) see "Data layout" below
├── results/                    # (created at runtime) logs + weights
├── environment.yaml            # conda environment specification
└── README.md
```

All scripts are run **as modules from the repository root** (see Section 5).
Internal imports are absolute (`from src.x import ...`), so running a file
directly (`python src/train.py`) will fail with an import error — use the
`python -m` form.

---

## 2. Environment setup

```bash
conda env create -f environment.yaml
conda activate deepfake
```

Key dependencies: PyTorch (CUDA 11.8 build), torchvision, `timm`,
`opencv-python`, scikit-learn, pandas, numpy, pyyaml, tqdm, scipy.

A CUDA-capable GPU is expected for training. Heavier backbones (Xception,
DenseNet121) need more GPU memory than EfficientNet-B0; reduce the batch sizes
in the config if you hit out-of-memory errors.

---

## 3. Data layout

The pipeline expects an instructor-provided subset of FaceForensics++ under
`data/raw/<dataset>/`, where `<dataset>` matches `data.dataset` in the config.
The raw tree uses **different folder names for the train and test splits**, and
the preprocessing script already accounts for this:

```
data/raw/faceforensics++/
├── train/
│   ├── real/<id>/<frame>.png                     # e.g. real/000/012.png
│   └── fake/<Method>/<pair>/<frame>.png          # Method ∈ {Deepfakes, Face2Face,
│                                                 #   FaceSwap, FaceShifter, NeuralTextures}
└── test/
    ├── FF-real/<id>/<frame>.png
    └── FF-fake/<FF-tag>/<pair>/<frame>.png        # FF-tag ∈ {FF-DF, FF-F2F, FF-FS,
                                                    #   FF-FaceShifter, FF-NT}
```

> **Note on method names.** The train split encodes full method names; the test
> split uses abbreviated `FF-` tags. `create_manifests.py` contains a
> `METHOD_MAP` that normalises both schemes onto five canonical names so the
> per-method evaluation table is consistent across splits. If your raw data
> contains a method tag not in that map, manifest generation will stop with a
> clear error naming the unrecognised tag — add it to `METHOD_MAP` and re-run.

> **macOS users.** Strip `.DS_Store` files before processing or zipping:
> `find . -name '.DS_Store' -delete`. They do not affect the image globs but
> add clutter to any archive you share.

---

## 4. What the preprocessing produces

`create_manipulations.py` walks the raw tree and writes, for every frame, a
flat-namespaced copy plus its derivatives:

```
data/raw_flattened/<dataset>/
├── real/   {split}_{target}_{frame}.png          # e.g. train_000_012.png
└── fake/   {split}_{method}_{pair}_{frame}.png    # e.g. train_Deepfakes_000_003_012.png

data/manipulated/<dataset>/
├── real_compressed/
├── fake_compressed/
├── fake_fpr/                 # fingerprint-removed fakes
└── fake_fpr_compressed/      # fingerprint-removed AND compressed fakes
```

Compression = downscale (area) → upscale (cubic) → JPEG quality 12.
Fingerprint removal = Gaussian low-pass filter in the frequency domain
(sigma = 40), applied to fake frames only.

`create_manifests.py` then builds the CSV manifests the data loaders read:

| Manifest | Columns | Used by |
|---|---|---|
| `baseline/train.csv`, `baseline/val.csv` | `path,label` | baseline training; val is also the calibration set for testing |
| `novel/train.csv`, `novel/val.csv` | `fake_fpr,fake_fpr_comp,real_raw,real_comp` | novel (quadruplet) training |
| `test/test_raw.csv` | `path,label,method` | evaluation on c23 frames |
| `test/test_compressed.csv` | `path,label,method` | evaluation on compressed frames |

Splits are by filename prefix: `test_*` frames are held out; the rest are
shuffled (fixed seed 42) into an 85/15 train/validation split.

---

## 5. Running the pipeline

From the repository root, in order:

```bash
# 1. Build the six image sets (slow: runs over the whole dataset once)
python -m preprocessing.create_manipulations

# 2. Build the manifests (fast: rewrites CSVs only)
python -m preprocessing.create_manifests

# 3. Train baseline then novel model
python -m src.train

# 4. Evaluate both models on RAW + Compressed test sets
python -m src.testing
```

Steps 1–2 only need re-running if the raw data or the manifest logic changes.
If you only edit method names or splits, re-run **step 2 alone** — the
manipulated images on disk are unaffected.

### Train → test handoff

`src/train.py` prints the timestamped weight filenames it saves, e.g.
`baseline_20260603_HHMMSS.pth` and `novel_20260603_HHMMSS.pth`, under
`results/training/weights/<backbone>/`. Copy those two paths into the
`__main__` block of `src/testing.py` (the `B_WEIGHT` and `N_WEIGHT` variables)
before running step 4. This manual handoff is deliberate: it is a checkpoint to
inspect the training logs and decide whether evaluation is worth running.

---

## 6. Configuration

`configs/base_config.yaml`:

```yaml
project_name: "deepfake_detection"

data:
  dataset: "faceforensics++"
  raw_dir: "data/raw/faceforensics++"
  img_size: 256
  num_workers: 4
  train_fraction: 1.0      # fraction of TRAIN data used; test sets are always full

model:
  backbone: "efficientnet_b0"   # any timm model name; swap to xception / densenet121 here
  pretrained: true
  embedding_dim: 128            # projection-head output dim (novel model only)

train:
  batch_size_base: 64
  batch_size_novel: 16          # smaller: each example is a quadruplet (4 images)
  epochs: 30
  warmup_epochs: 3              # novel model trains heads only during warmup, then unfreezes
  learning_rate: 0.0001
```

`utils/config_loader.py` validates this file before any run. It checks every
key is present, correctly typed, and (for sizes and rates) positive; that
`train_fraction` is in `(0, 1]`; and that `warmup_epochs < epochs`. It also
coerces a learning rate written in scientific notation (e.g. `1e-4`), which
PyYAML would otherwise parse as a string.

### Swapping the backbone

To evaluate a different architecture, change `model.backbone` to any `timm`
model name and re-run training and testing. Nothing else needs editing — the
`UniversalBackbone` reads the model's native feature dimension automatically.
Heavier models may require lowering `batch_size_base` / `batch_size_novel` to
fit in GPU memory.

### Quick end-to-end test

`configs/smoke_config.yaml` runs the entire pipeline on 2% of the training data
for 2 epochs (warmup 1). It is meant to confirm every code path executes, not
to produce meaningful metrics. Point `src/train.py` and `src/testing.py` at
this config to use it.

---

## 7. Outputs

```
results/
├── training/
│   ├── logs/<backbone>/baseline/logs_<ts>.csv     # per-epoch train/val loss
│   ├── logs/<backbone>/novel/logs_<ts>.csv        # per-epoch loss + components (bce/pull/push)
│   └── weights/<backbone>/{baseline,novel}/*.pth  # best-validation checkpoints
└── testing/
    └── logs/<backbone>/test_results_<ts>.csv      # aggregate + per-method metrics
```

### How to read the test results

Each model is evaluated on each test set at **two thresholds**, tagged in the
`threshold_type` column:

- **`fixed_0.5`** (primary) — the conventional 0.5 decision threshold.
- **`val_calibrated`** (secondary) — a threshold frozen from the held-out
  `baseline/val.csv` (Youden's J on the ROC curve), then applied unchanged to
  both test sets. This is never tuned on a test set, so the reported numbers
  carry no test-set leakage.

AUC is threshold-independent and so is identical across the two rows for a given
(model, test set) — a useful built-in sanity check. The headline measure of
compression robustness is the **RAW-to-Compressed gap** within a model, not the
absolute accuracy.

---

## 8. Model design (brief)

- **UniversalBackbone** — a `timm` model loaded with no classification head,
  plus a single linear layer to one logit. Returns both the logit and the
  backbone feature vector on every forward pass.
- **NovelSiameseWrapper** — folds a batch of quadruplets `[B, 4, 3, H, W]` into
  `[4B, 3, H, W]`, passes them through the shared backbone in one call (keeping
  batch-norm statistics stable), then a two-layer projection head produces an
  embedding per image. The contrastive loss acts on the projection embeddings;
  classification reads the backbone logits.
- **Composite objective** (novel): BCE classification + MSE *pull* (image vs its
  compressed counterpart) + margin *push* (real vs fake at each quality level).
  Pull and push are weighted at 0.1 each and switch on only after warmup.
- The projection head is used only during training. The saved checkpoint is the
  backbone plus its linear head, so the novel model loads at evaluation time as
  a plain `UniversalBackbone` — identical inference path to the baseline.

---

## 9. Reproducibility notes

- Random seed 42 fixes the train/validation split and manifest shuffling.
- Both detectors use early stopping (patience 3 on validation loss) and are
  checkpointed on best validation loss, not the final epoch.
- `train_fraction` subsamples only manifests with `train` in their path; the
  validation and test sets are always evaluated in full, so comparisons across
  `train_fraction` settings remain fair.