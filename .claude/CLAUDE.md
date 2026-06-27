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
| 3. PyTorch Implementation | Run/train RS-Mamba baseline on LEVIR-CD (also the throughput baseline) | **In Progress** |
| 4. TensorRT Export | ONNX/TensorRT export, FP16/INT8 calibration, accuracy retention | Planned |
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
**Environment validated end-to-end on Kaggle (Tesla T4).** The upstream RS-Mamba code has been
vendored into `model/vendor/rs_mamba_cd/`; the selective-scan CUDA kernel builds; and a forward
pass through the change-detection model (`RSM_CD`) runs and produces a correctly-shaped output
mask. The full validation session is preserved in `notebooks/environment_validation.ipynb`.

**Next step:** load and validate the LEVIR-CD dataset against the vendored loader format, then
stand up a PyTorch training baseline (adapting `model/vendor/rs_mamba_cd/train.py` and
`utils/data_loading.py`). This baseline doubles as the throughput reference for later TensorRT work.

## Dataset
**Selected: LEVIR-CD** — binary change detection, 637 image pairs, 1024×1024px.
*(OSCD — multispectral Sentinel-2, 24 urban scenes — was considered as an alternative but not chosen.)*

Expected on-disk layout for the vendored loader (`BasicDataset` in `utils/data_loading.py`):
three parallel directories `t1/`, `t2/`, `label/`, with **matching filenames** for each example
(t1 image, t2 image, and label share the same stem). Labels are binarized at load time
(`label[label != 0] = 1`). Inputs are Albumentations-normalized; training augmentation is random
flip/transpose plus a random t1/t2 swap.

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
- Starting points for the baseline pipeline: `utils/train.py` and `utils/data_loading.py`.

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

## Reference
- Primary paper: RS-Mamba (omnidirectional SSM for remote sensing)