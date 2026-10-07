# Tier 2 — TensorRT C++ inference path

The Python harness (Tier 1) proves a deployed model is *correct*. This directory
makes it *fast* on NVIDIA hardware and measures where the time goes: a C++
TensorRT pipeline with GPU preprocessing, in-engine NMS, CUDA graphs, and a
two-model contention benchmark — plus one script that runs all of it.

> **Status: ready to run, not yet run on a GPU.** What has been verified without one:
> host-side C++ (decode, NMS, letterbox reference, stats, I/O) is unit-tested on macOS
> and Linux and matches the Python decode exactly (709/709 detections on real model
> output); the NMS graph surgery is checked against ONNX Runtime; all CUDA / TensorRT
> sources compile in CI with `nvcc` against real TensorRT 8.6, 10.8 and 11.x headers. Kernel correctness,
> engine builds and every latency number need the GPU run.

## Run it

**x86 + NVIDIA GPU (cloud L4 / T4 / A10, workstation)** — inside the TensorRT container:

```bash
make gpu-docker-build        # nvcr.io/nvidia/tensorrt:25.01-py3 + deps
make gpu-docker-run          # → reports/gpu/<gpu-name>/SUMMARY.md
```

**Jetson Orin (JetPack 6)** — natively; JetPack already has CUDA, TensorRT, nsys:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks      # stable clocks for benchmarking
pip install cmake numpy opencv-python-headless matplotlib pyyaml onnx pytest
SKIP_PIP=1 bash scripts/gpu/run_all.sh
```

**Any machine with CUDA + TensorRT installed:** `bash scripts/gpu/run_all.sh`
(`QUICK=1` for a ~5-minute smoke run; `-DTENSORRT_ROOT=` if TensorRT is a tarball install).

Everything lands in `reports/gpu/<gpu-name>/`: `SUMMARY.md`, per-config JSON, detection
dumps, Nsight Systems reports, `env.txt`, `run.log`. Failed steps are logged and the run
continues, so one problem doesn't cost the rest of the session.

## What gets measured

| # | experiment | question it answers |
|---|---|---|
| 1 | **Latency matrix**, batch 1, one change per row: FP32 → FP16 → GPU preprocessing → EfficientNMS → CUDA graph → FP32 head tail → INT8 (→ DLA on Orin) | Where does end-to-end time go, and what does each optimisation buy? Tier 1 showed preprocessing was ~50% of e2e on the fastest Apple target. |
| 2 | **Per-stage timing** via CUDA events: host pre, H2D, GPU pre, infer, D2H, post | Is the bottleneck the network, the copies, or the CPU? |
| 3 | **Batch sweep** 1/2/4/8 | Offboard throughput, and whether batching behaves as expected on this GPU (it didn't on the M4 CPU). |
| 4 | **Accuracy parity of the engines** on the 300 held-out scenes vs ONNX Runtime FP32, same budgets as Tier 1 | Does the *whole* C++ path (GPU letterbox + TensorRT precision + EfficientNMS) still produce the trained model's detections? Does TensorRT FP16 show the cone regression CoreML FP16 did, and does pinning the head tail to FP32 fix it? |
| 5 | **Two models, one GPU**: A = detector at 30 Hz (deadline 33 ms), B = batch-8 background model saturating the GPU; A alone / B alone / together / together with A on a high-priority stream | How much does a co-located model inflate the critical model's p99 and deadline misses, and how much do stream priorities buy back? |
| 6 | **Nsight Systems** timelines (NVTX ranges per stage), with and without CUDA graphs; `trtexec` cross-check | Launch gaps, copy/compute overlap, and an independent check of the harness's numbers. |
| 7 | **ONNX Runtime CUDA / TensorRT EPs** — the Tier-1 `nvidia-trt` profile through the same parity/benchmark/gate | Same release gate, now on NVIDIA. |

## Pieces

| file | role |
|---|---|
| `src/engine.cpp` | ONNX → engine (FP16 / INT8-QDQ / DLA, optimisation profiles, timing cache, `--fp32-layers` precision pinning, strongly typed builds) and a runtime wrapper (named I/O tensors, `enqueueV3`, pinned mirrors). TensorRT 8.6 (JetPack 6) through 11.x. |
| `src/preprocess.cu` | Fused letterbox kernel: bilinear resize + pad + BGR→RGB + HWC→CHW + scale in one pass. The H2D copy becomes uint8 HWC (1.2 MB) instead of float32 CHW (4.9 MB). |
| `src/pipeline.cpp` | One detector on one stream: pinned staging → H2D → preprocess → TensorRT → D2H → post; optional CUDA-graph capture of the whole GPU side; CUDA-event + NVTX instrumentation. |
| `src/postprocess.cpp`, `src/letterbox.cpp` | CPU decode/NMS and letterbox reference — identical semantics to `nurosim/inference/`, so C++ and Python outputs compare box for box. |
| `tools/build_engine.cpp` | `nurosim_build_engine` |
| `tools/bench.cpp` | `nurosim_bench` — latency (JSON) or `--dump-dets` for parity |
| `tools/multistream.cpp` | `nurosim_multistream` — experiment 5 |
| `tests/` | host tests (run everywhere) and a GPU kernel-vs-reference test (ctest label `gpu`) |
| `../nurosim/deploy/trt_prep.py` | frame sets (`.nsfr`) + EfficientNMS graph surgery |
| `../nurosim/deploy/cpp_parity.py`, `gpu_report.py` | scoring of C++ detections; `SUMMARY.md` |
| `../scripts/gpu/run_all.sh` | the orchestration above |

## Design notes

* **EfficientNMS in the engine** — decode + per-class NMS run on the GPU and only
  `num_dets` + 100 boxes come back, instead of the 8 × 8400 raw head (269 KB) plus CPU NMS.
* **CUDA graphs** — the GPU side is a fixed sequence (copy, kernel, TensorRT, copies) with
  static addresses, so it's captured once and replayed as one launch. Capture is attempted
  per engine and falls back cleanly if TensorRT can't be captured.
* **Precision pinning** — `--fp32-layers '^/model\.22/(?!cv2|cv3)'` keeps the head's decode
  arithmetic in FP32 under `kFP16` / `kINT8` — the TensorRT equivalent of Tier 1's
  `fp16_mixed`, which fixed the CoreML cone regression.
* **TensorRT 11 is strongly typed only** — it removed `kFP16` / `kINT8` and per-layer
  precision APIs. The builder detects the version: on 8.6 / 10.x engines are built from the
  FP32 graph plus precision flags (and `--fp32-layers` pinning); on ≥ 11 from the Tier-1
  `fp16` / `fp16_mixed` / Q/DQ ONNX variants, whose graphs already carry the precision.
  `run_all.sh` picks the recipe automatically. CI compiles against TensorRT 8.6 (JetPack 6.0),
  10.8 (the container) and the latest release.
* **INT8** — the Tier-1 QDQ model (symmetric, per-channel) is consumed as explicit
  quantisation; no TensorRT calibrator, so ORT-CPU, ORT-TensorRT and native TensorRT
  all use the same scales.
* **Stream priority** — A's stream gets the highest priority from
  `cudaDeviceGetStreamPriorityRange`; B the lowest.

## Troubleshooting

* *Container won't start / CUDA driver too old* — use an older image:
  `docker build --build-arg TRT_IMAGE=nvcr.io/nvidia/tensorrt:24.08-py3 -f gpu/Dockerfile .`
* *ONNX Runtime TensorRT EP fails to load* — onnxruntime-gpu must match the container's
  TensorRT major version; the C++ path doesn't depend on it, and the run continues.
* *`nsys` missing* — `apt install nsight-systems-cli` (x86) — profiles are skipped otherwise.
* *DLA engine fails to build* — expected on non-Orin Jetsons; logged and skipped.
