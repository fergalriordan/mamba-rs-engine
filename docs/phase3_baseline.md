# Phase 3 — PyTorch Baseline Results

Detailed results and validation history for Phase 3 (PyTorch training pipeline + throughput baseline).
This is a reference record, not active instructions — for the working build details see
`model/CLAUDE.md`; for current status and the headline numbers see the root `CLAUDE.md`.

## Summary
**Phase 3 complete — training pipeline validated end-to-end on Kaggle (Tesla T4) and the throughput
baseline is captured.** Preserved with outputs in `notebooks/training_baseline.ipynb`.

- Full top-to-bottom run succeeds: env + kernel build → import vendored code → kagglehub download →
  crop 1024→256 on disk (7120/1024/2048 tiles) → `BasicDataset` loaders → `RSM_CD` tiny (**51.95M
  params**) → smoke train → val metrics → throughput benchmark.

## Training (smoke run)
- **Training learns cleanly:** Dice+BCE loss falls monotonically `0.920 → 0.790 → 0.691 → 0.653 →
  0.582` over 5 epochs (150 batches/epoch cap, batch 16, AMP). This is a *smoke run* (not converged).
- **Validation (smoke run):** change-class **IoU 0.594, F1 0.745** (precision 0.69 / recall 0.81) —
  well beyond trivial, confirming real learning. Not a final accuracy number (capped, ~1.7 effective
  epochs of the full train set).

## Throughput baseline (LOCKED)
Forward-only on synthetic `(B,3,256,256)` tensors, Tesla T4 @ res 256, warmup + `cuda.synchronize()`,
averaged:

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
is the suspected bottleneck — strong motivation for the TensorRT engine, but see the Phase 4 SSM-export
risk (root `CLAUDE.md`).

## Prior validation (earlier sessions)
Environment and dataset were separately validated before the Phase-3 baseline run:
- `notebooks/environment_validation.ipynb` — kernel builds, `RSM_CD` forward pass.
- `notebooks/levir_cd_inspection.ipynb` — layout, counts, label values, class balance (validated
  dataset facts now recorded in `model/CLAUDE.md` under **Dataset**).