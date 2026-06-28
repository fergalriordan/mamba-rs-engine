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
│                   #   └── see model/CLAUDE.md for env setup, vendored-code facts, training/dataset specifics
├── data/           # Dataset loading and preprocessing (LEVIR-CD / OSCD)
├── export/         # ONNX/TensorRT export scripts
├── engine/         # C++ TensorRT inference server
│   ├── src/        # C++ source files
│   ├── include/    # Header files
│   └── CMakeLists.txt
├── eval/           # Benchmarking: latency, throughput, accuracy metrics
├── notebooks/      # Exploratory work and visualisations
├── docs/           # Architecture notes and reference material
│                   #   └── docs/phase3_baseline.md: full Phase-3 results, throughput tables, validation history
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
**Phase 3 complete — PyTorch training pipeline validated end-to-end on Kaggle (Tesla T4); throughput
baseline locked. Phase 4 (TensorRT export) is next.**

- **Throughput baseline (LOCKED): PyTorch reference = 23.7 img/s** (best: FP16, batch 8, T4 @ 256).
  Phase 5 (TensorRT, FP16) must beat this by >2× → **~47 img/s** target. Full results, metrics, and
  the per-batch/precision table are in `docs/phase3_baseline.md`.
- Model is `RSM_CD` tiny, 51.95M params. Working notebook: `notebooks/training_baseline.ipynb`.

⚠️ **Phase 4 risk (SSM export) — scope this FIRST.** The model relies on the custom
`selective_scan_cuda_oflex` CUDA op, which is **not** a standard ONNX/TensorRT operator. Exporting
Mamba/SSM models is a known hard problem — expect to either provide a TensorRT plugin for the scan,
decompose it into exportable ops, or use an ONNX custom-op path. This may dominate Phase 4's effort;
de-risk the `selective_scan` export before pursuing FP16/INT8 calibration.

**Next step:** Phase 4 — begin by de-risking the `selective_scan` op export (above).

> **Working in `model/`, `data/`, or `export/`?** See `model/CLAUDE.md` for the Kaggle environment
> recipe (kernel build + drift fixes), vendored RS-Mamba facts, the training-pipeline build, and
> validated LEVIR-CD dataset facts. Detailed Phase-3 results live in `docs/phase3_baseline.md`.

## Dataset
**Selected: LEVIR-CD** — binary change detection, 637 image pairs, 1024×1024px.
*(OSCD — multispectral Sentinel-2, 24 urban scenes — was considered as an alternative but not chosen.)*

Full validated on-disk facts (source, layout, counts, label values, class balance) are in
`model/CLAUDE.md` under **Dataset**.

## Reference
- Primary paper: RS-Mamba (omnidirectional SSM for remote sensing)