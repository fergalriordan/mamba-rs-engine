# Mamba-RS-Engine

## Project Goal
High-throughput C++/TensorRT inference engine for Remote Sensing (RS) change detection using State Space Models (RS-Mamba). Target domain: environmental monitoring — specifically wildfire and flood detection from high-resolution satellite imagery.

## Architecture Overview
The project has four conceptually separate pipeline stages:

1. **Model** — PyTorch RS-Mamba architecture definition and training
2. **Export** — Conversion from PyTorch → ONNX → TensorRT engine
3. **Engine** — Standalone C++ inference server using TensorRT
4. **Eval** — Benchmarking throughput, latency, and change detection accuracy

### Roadmap Status
| Phase | Description | Status |
|-------|-------------|--------|
| 1. Literature Review | SSMs / Mamba, ViT limitations in RS, change-detection survey | **Complete** |
| 2. Use Case & Dataset | Wildfire/flood use case; dataset = LEVIR-CD; access + loader format | **Complete** |
| 3. PyTorch Implementation | Run/train RS-Mamba baseline on LEVIR-CD (also the throughput baseline) | **Complete** — pipeline validated; baseline 23.7 img/s (FP16) on T4 |
| 4. TensorRT Export | ONNX/TensorRT export, FP16/INT8 calibration, accuracy retention | **Next** — de-risk `selective_scan` op export first |
| 5. C++ Inference Server | Standalone TensorRT server, benchmark >2× vs PyTorch | Planned |

## Directory Structure

```
Mamba-RS-Engine/
├── model/          # PyTorch RS-Mamba model definition and training pipeline
├── data/           # Dataset loading and preprocessing (LEVIR-CD / OSCD)
├── export/         # ONNX/TensorRT export scripts
├── engine/         # C++ TensorRT inference server
│   ├── src/        # C++ source files
│   ├── include/    # Header files
│   └── CMakeLists.txt
├── eval/           # Benchmarking: latency, throughput, accuracy metrics
├── notebooks/      # Exploratory work and visualisations
├── docs/           # Architecture notes and reference material
├── tests/          # Unit and integration tests
├── scripts/        # Utilities: dataset download, environment setup etc.
├── requirements.txt
├── CLAUDE.md
├── README.md
└── LICENSE
```

## Dependencies
- **Python**: see `requirements.txt` at repo root
- **C++**: see `engine/CMakeLists.txt` (TensorRT, CUDA, OpenCV)

## Build & Run
[Fill in as the project develops]

## Conventions
### Python
- Type hints required on all function signatures
- Docstrings required on all classes and public functions
- Follow PEP8

### C++
- C++17 standard
- Google C++ Style Guide
- Headers in `engine/include/`, sources in `engine/src/`

### General
- Conventional commits format: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`
- No commented-out code in commits — use branches instead

## Key Technical Constraints
- Inference target: superior throughput vs PyTorch baseline
- Must be deployable in edge-constrained environments
- Educational priority: implementation decisions should be explainable;
  prefer clarity over cleverness where performance allows

## Current Status
**Phase 3 complete — training pipeline validated end-to-end on Kaggle (Tesla T4) and the throughput
baseline is captured.** Preserved with outputs in `notebooks/training_baseline.ipynb`.
- Full top-to-bottom run succeeds: env + kernel build → import vendored code → kagglehub download →
  crop 1024→256 on disk (7120/1024/2048 tiles) → `BasicDataset` loaders → `RSM_CD` tiny (**51.95M
  params**) → smoke train → val metrics → throughput benchmark.
- **Training learns cleanly:** Dice+BCE loss falls monotonically `0.920 → 0.790 → 0.691 → 0.653 →
  0.582` over 5 epochs (150 batches/epoch cap, batch 16, AMP). This is a *smoke run* (not converged).
- **Validation (smoke run):** change-class **IoU 0.594, F1 0.745** (precision 0.69 / recall 0.81) —
  well beyond trivial, confirming real learning. Not a final accuracy number (capped, ~1.7 effective
  epochs of the full train set).
- **Throughput baseline (LOCKED).** Forward-only on synthetic `(B,3,256,256)` tensors, Tesla T4 @
  res 256, warmup + `cuda.synchronize()`, averaged:

  | precision | batch | img/s | ms/batch |
  |-----------|-------|-------|----------|
  | fp32 | 1 | **16.0** | 62.6 |
  | fp16 | 1 | 14.6 | 68.7 |
  | fp32 | 8 | 14.1 | 568.3 |
  | fp16 | 8 | **23.7** | 337.3 |
  | fp32 | 16 | 14.2 | 1124.4 |
  | fp16 | 16 | 23.1 | 691.2 |

  **PyTorch reference = 23.7 img/s** (best: FP16, batch 8). FP32 best = 16.0 img/s (batch 1).
  Phase 5 (TensorRT, FP16) must beat the PyTorch best by >2× → **~47 img/s** target.
  **Finding:** FP32 throughput is *flat* (~14–16 img/s) across batch 1→16 (no batching benefit);
  FP16 scales to ~24 img/s by batch 8, then plateaus. The largely-sequential `selective_scan` kernel
  is the suspected bottleneck — strong motivation for the TensorRT engine, but see the export risk
  below.

⚠️ **Phase 4 risk (SSM export):** the model relies on the custom `selective_scan_cuda_oflex` CUDA
op, which is **not** a standard ONNX/TensorRT operator. Exporting Mamba/SSM models is a known hard
problem — expect to either provide a TensorRT plugin for the scan, decompose it into exportable ops,
or use an ONNX custom-op path. Scope this early in Phase 4; it may dominate that phase's effort.

**Prior validation** (still relevant): environment + dataset were separately validated earlier —
`notebooks/environment_validation.ipynb` (kernel builds, `RSM_CD` forward pass) and
`notebooks/levir_cd_inspection.ipynb` (layout, counts, label values, class balance; see **Dataset**).

**Next step:** Phase 4 — ONNX/TensorRT export. Begin by de-risking the `selective_scan` op export
(see the Phase 4 risk above) before pursuing FP16/INT8 calibration and accuracy retention.

## Dataset
**Selected: LEVIR-CD** — binary change detection, 637 image pairs, 1024×1024px.
*(OSCD — multispectral Sentinel-2, 24 urban scenes — was considered as an alternative but not chosen.)*

**Validated on-disk facts (from `notebooks/levir_cd_inspection.ipynb`, Kaggle session):**
- **Source:** `kagglehub.dataset_download("mdrifaturrahman33/levir-cd")` — fully code-based, no UI
  "Add Input" step. Requires Internet enabled in the notebook (already needed for git/pip). It
  resolves to `…/datasets/mdrifaturrahman33/levir-cd/LEVIR CD` (note the **space** in `LEVIR CD`).
- **This is the original full-resolution LEVIR-CD** (637 pairs @ **1024×1024**), *not* the
  pre-cropped LEVIR-CD256 redistribution.
- **Layout:** split dirs `train/`, `val/`, `test/`, each containing `A/` (before), `B/` (after),
  `label/`. Note: folders are **`A`/`B`, not `t1`/`t2`** — the loader takes explicit dir paths, so
  pass `A`→t1 and `B`→t2 directly (no renaming needed).
- **Counts:** train **445**, val **64**, test **128** → **637** matched triplets, zero orphans.
- **Images:** RGB `uint8` [0,255]. **Labels:** mode `L`, values exactly **{0, 255}** → the loader's
  `label[label != 0] = 1` binarization is correct.
- **Class balance (heavy foreground imbalance):** only ~**4–5%** of pixels are "changed" (train mean
  4.59%, val 4.20%, test 5.09%); ~8–12% of full tiles have **no change at all**. This justifies the
  Dice+BCE-style loss the vendored pipeline already uses.

Expected on-disk layout for the vendored loader (`BasicDataset` in `utils/data_loading.py`):
three parallel directories `t1/`, `t2/`, `label/`, with **matching filenames** for each example
(t1 image, t2 image, and label share the same stem). Labels are binarized at load time
(`label[label != 0] = 1`). Inputs are Albumentations-normalized; training augmentation is random
flip/transpose plus a random t1/t2 swap.

## Training Pipeline
The PyTorch training baseline lives in `notebooks/training_baseline.ipynb` (runs top-to-bottom on a
Kaggle T4). It reuses the env recipe below and the kagglehub download from the inspection notebook.
**Decisions made for this baseline:** ephemeral crops in `/kaggle/working` (re-cropped each session,
*not* a Kaggle output dataset); short **smoke** training (throughput-first priority); metrics computed
manually for the change class. The two prerequisites below are handled first, in this order:

**1. Crop 1024→256 (mandatory, on-disk preprocessing).**
`BasicDataset` does **no resizing/cropping** — it only Normalizes + ToTensors (see
`utils/data_loading.py:99`). The tiny config is fixed at `image_size=256` but the data is 1024×1024,
so 256px tiles must exist **on disk** first.
- **Implemented as a small custom crop loop** (not the vendored `crop_img`, which hard-codes
  `t1`/`t2`/`label` subdir names + a `{dataset_name}/{split}/...` root and has a size-divisibility
  assertion). The loop maps **A→t1, B→t2, label→label** with matching stems, non-overlapping 4×4 grid
  → 16 tiles/image: **7120 train / 1024 val / 2048 test** (confirmed in the notebook output).
- Crops persist only in ephemeral `/kaggle/working` (cheap to recreate; ~1–2 min). If a later session
  wants to avoid recropping, switch to saving a Kaggle output dataset.

**2. Make the vendored code importable (so `BasicDataset` + `RSM_CD` load).**
Clone this repo into `/kaggle/working` and add **only** `model/vendor/rs_mamba_cd` (the package root)
to `sys.path` — see the two `sys.path`/import gotchas in *Environment Setup*.

**The baseline itself (as built):**
- Model: `RSM_CD(forward_type="v3")`, **RSM-CD tiny** config, **51.95M params**. Hyperparams: AdamW
  lr `1e-3`, wd `1e-3`, AMP on, grad-clip `20`, `batch_size=16` at 256px on the T4.
- **Loss:** vendored `FCCDN_loss_without_seg` (soft-Dice + BCE-with-logits), returning
  `(total, dice, bce)`. Labels must be cast to `float` for the loss. Keep this combo — don't use plain
  BCE — given the ~4–5% positive-pixel imbalance.
- **Metrics:** change-class **IoU / F1 / precision / recall** computed **manually** from accumulated
  TP/FP/FN (clearer, and sidesteps torchmetrics 0.9.3's binary-API quirks; torchmetrics is still
  installed). Report these, not pixel accuracy (~95%+ trivially due to imbalance). Smoke-run result:
  IoU 0.594 / F1 0.745.
- **Throughput baseline:** measured forward-only on synthetic tensors with warmup +
  `cuda.synchronize()` across batch sizes {1,8,16} × {fp32,fp16}. **Result: 23.7 img/s** (best, FP16
  batch 8) on a T4 @ 256 — the PyTorch reference TensorRT must beat in Phase 5. FP32 is flat ~14–16
  img/s; FP16 scales to ~24. Full table in **Current Status**.

## Vendored RS-Mamba
The original RS-Mamba change-detection code is vendored at `model/vendor/rs_mamba_cd/`
(upstream commit `997866c`, Apache 2.0 — see `ATTRIBUTION.md`). Key facts for using it:

- **Model class is `RSM_CD`** (in `rs_mamba_cd.py`) — *not* `RSMamba_cd`.
- Validated forward pass: `RSM_CD(forward_type="v3").cuda()` with two bitemporal tensors of
  shape `(B, 3, 256, 256)` → output `(B, 1, 256, 256)`, a per-pixel binary change logit map.
  `forward_type="v3"` routes through the compiled `selective_scan_cuda_oflex` kernel.
- **RSM-CD tiny** config (`utils/path_hyperparameter.py`): `dims=96`, `depths=[2,2,9,2]`,
  `ssm_d_state=16`, `ssm_ratio=2.0`, `mlp_ratio=4.0`, `image_size=256`.
- Default training hyperparams (same file): AdamW, lr `1e-3`, weight_decay `1e-3`, AMP on,
  `batch_size=2`, 300 epochs, grad-clip `max_norm=20`.
- Starting points for the baseline pipeline: `train.py` (at the vendor **root**,
  `model/vendor/rs_mamba_cd/train.py`) and `utils/data_loading.py`. Cropping/splitting helpers
  (`crop_img`, `split_image`, `compute_mean_std`) live in `utils/dataset_process.py`.

## Environment Setup (Kaggle)
The model depends on a CUDA selective-scan kernel that is **not** bundled and must be compiled.
This recipe was validated on a Kaggle T4 notebook and should be reused for future sessions.

**Validated environment:** Tesla T4 (compute capability 7.5 — below 8.0, so a BFloat16 warning
is expected but the `oflex` kernel still works), Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8,
gcc/g++ 11.4.0.

**Steps:**
1. Clone upstream RS-Mamba (`https://github.com/walking-shadow/Official_Remote_Sensing_Mamba`).
   ⚠️ Its `requirements.txt` (mirrored at this repo's root) **fails to install** on Python 3.12
   because it pins `numpy==1.22.4` (no wheel, build fails). Install the dependencies individually
   **without the `numpy` / `Pillow` version pins**.
2. Build the selective-scan CUDA kernel from VMamba (`https://github.com/MzeroMiko/VMamba`):
   run `pip install .` inside `kernels/selective_scan/`. This compiles and installs the
   `selective_scan_cuda_oflex` extension (its `setup.py` uses `MODES=["oflex"]`).
3. ⚠️ **Import-ordering gotcha:** `rs_mamba_cd` binds the kernel *at import time*. Either build
   the kernel **before** importing `rs_mamba_cd`, or — if you imported it first — run
   `del sys.modules['rs_mamba_cd']` and reimport after the build so the extension binds.

**Environment-drift fixes (discovered building `training_baseline.ipynb`, Jun 2026).** Kaggle's base
image is now newer than the vendored code expects; the notebook handles all three, but record them so
the next session gets them right the first time:
- ⚠️ **`import torch` BEFORE `import selective_scan_cuda_oflex`.** The compiled kernel links against
  PyTorch's shared libs; importing it cold fails with `libc10.so: cannot open shared object file`.
  Importing `torch` first loads those libs into the process so the extension resolves.
- ⚠️ **Put only the vendor *root* (`model/vendor/rs_mamba_cd`) on `sys.path`, NOT `.../utils`.** The
  vendored code uses absolute imports (`from utils.path_hyperparameter import ph`), so `utils` must
  resolve to the package dir. Adding `.../utils` itself makes `utils/utils.py` import as a top-level
  module named `utils`, shadowing the package → `ModuleNotFoundError: ... 'utils' is not a package`.
- ⚠️ **albumentations ≥2.0 removed `A.Flip`** (used in `BasicDataset.__init__`). Add a compat shim
  before constructing the dataset: if `not hasattr(A, "Flip")`, define `A.Flip` as
  `A.OneOf([A.HorizontalFlip(p=1), A.VerticalFlip(p=1)], p=p)`.
- Harmless noise to expect: pip dependency-resolver conflicts about Kaggle's RAPIDS stack
  (`dask-cuda`/`cuml`/`cudf` vs `numba`) — we don't use those packages; ignore. Also `FutureWarning`s
  from the vendored `@torch.cuda.amp.custom_fwd/bwd` decorators and `timm.models.layers`.

## Reference
- Primary paper: RS-Mamba (omnidirectional SSM for remote sensing)