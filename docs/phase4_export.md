# Phase 4 — TensorRT Export: `selective_scan` De-Risk Findings

Reference record for the Phase-4 export investigation. For current status and headline numbers see
the root `CLAUDE.md`; for the env recipe and vendored-model facts see `model/CLAUDE.md`. The
investigation notebook (with outputs) is `notebooks/phase4_export_derisk.ipynb`.

## TL;DR
The documented Phase-4 risk — that `RSM_CD` could not cross PyTorch → ONNX because of the custom
`selective_scan_cuda_oflex` CUDA op — **did not materialise for the export step.** A naive
`torch.onnx.export` via the **legacy TorchScript exporter** produces a **valid, fully-standard-op,
numerically-correct ONNX graph**. None of the three feared remediation paths (TensorRT plugin /
decompose the scan / ONNX custom-op) are needed to get to ONNX. **The remaining Phase-4 risk has
moved downstream to ONNX → TensorRT ingestion**, which is untested.

## Environment
Validated on Kaggle Tesla T4, Python 3.12, **torch 2.10.0+cu128**, onnx 1.21.0, onnxruntime 1.27.0,
`onnxscript` installed (required by the dynamo exporter). Same kernel-build + import-ordering recipe
as Phase 3 (`model/CLAUDE.md`). Model: `RSM_CD(forward_type="v3")`, RSM-CD tiny, 51.95M params.
Dummy bitemporal input: two `(1,3,256,256)` tensors.

## What was tested and what happened

| Attempt | Backend | Result |
|---|---|---|
| `v3` (real scan) | legacy TorchScript (`dynamo=False`) | **SUCCESS** — verified genuine |
| `v3` (real scan) | dynamo / `torch.export` (`dynamo=True`) | **FAILED** at the CUDA scan |
| `fake` (no-op scan) | legacy TorchScript | SUCCESS |
| `fake` (no-op scan) | dynamo | SUCCESS |

### Verification of the v3 ONNX (the decisive part — "export returned" ≠ "export valid")
On `rsm_cd_v3_torchscript.onnx`:
- `onnx.checker`: **OK**.
- **9126 nodes, all in the default domain `''` → zero custom-domain ops.** The scan and the
  `CrossScan`/`CrossMerge` autograd Functions decomposed into *standard* ONNX ops
  (`Gather`, `ScatterElements`, `Range`, `Mod`, `Where`, `Expand`, `MatMul`, `Conv`,
  `LayerNormalization`, `Resize`, …).
- Output **changes with input** (not constant-folded).
- **max|onnxruntime − PyTorch| = 0.0032**, `allclose(atol=1e-2) = True` on *fresh* random input →
  numerically correct (the SSM is genuinely live in the graph, not baked from the trace dummy).

## Why the two exporter backends disagree (the key mechanism)
- **Legacy TorchScript exporter traces with real tensors.** It actually *runs*
  `selective_scan_cuda_oflex.fwd` on CUDA during tracing; the surrounding tensor algebra is recorded
  as standard ops. Result: a working graph.
- **Dynamo / `torch.export` traces with FakeTensors**, which cannot enter the opaque CUDA kernel
  (no accessible `data_ptr`). It fails with:
  `RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor)… erroneously tracing into a
  custom kernel… wrap the custom kernel into an opaque custom op`, raised at `rs_mamba_cd.py:201`.
  This is the canonical SSM/custom-CUDA export failure.

## Scan is the *sole* export blocker
`fake/dynamo` succeeds while `v3/dynamo` fails **only** at `selective_scan_cuda_oflex.fwd` →
`CrossScan`/`CrossMerge` and everything else are fully exportable. This narrows any future fix to the
scan op alone.

## Recommended next step
**Proceed with the legacy-TorchScript ONNX and de-risk ONNX → TensorRT** (e.g.
`trtexec --onnx=rsm_cd_v3_torchscript.onnx --fp16`, or the TRT Python API). The PyTorch→ONNX problem
is solved; whether TensorRT ingests this 9126-node graph is the open question. The three documented
paths become relevant **only if TRT rejects the graph**; the documented dynamo fix ("wrap the custom
kernel into an opaque custom op" via `torch.library`, then re-export) is the fallback and is likely
unnecessary.

## Open risks (do not over-trust the green result)
1. **ONNX → TensorRT untested.** onnxruntime-CPU acceptance ≠ TensorRT acceptance. The graph relies
   on dynamic-shape-ish ops TRT can restrict: `Range`, `Mod`, `ScatterElements`, `Where`, `Shape`,
   `Resize`. **This is now the primary Phase-4 risk.**
2. **Dynamic batch axis declared but not exercised** — verification ran batch 1 only. Re-run ORT with
   batch > 1 before relying on the `batch` dynamic axis.
3. **Legacy exporter is deprecated** (PyTorch 2.9+ default is dynamo). Works on 2.10 today; pin the
   toolchain or plan the opaque-custom-op migration for durability.
4. **Spatial size baked to 256** (only batch is dynamic) — fine for the 256px-tile pipeline.
5. **fp16/INT8 numerics untested** — only fp32 validated so far; precision calibration is still ahead.
