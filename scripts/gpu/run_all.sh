#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-shot GPU run: everything Tier 2 measures, on whatever NVIDIA GPU this is.
#
#   scripts/gpu/run_all.sh            # full run (~20-40 min, mostly engine builds)
#   QUICK=1 scripts/gpu/run_all.sh    # fewer iterations, shorter phases
#
# Works on x86 + discrete GPU (inside the TensorRT container, see gpu/Dockerfile)
# and natively on Jetson Orin (JetPack 6). Each step is logged; a failing step
# is recorded and the run continues. Results: reports/gpu/<gpu>/SUMMARY.md
#
# Env: PYTHON (python3), SKIP_PIP=1, QUICK=1, CUDA_ARCH (override "native")
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=${PYTHON:-python3}
export PATH=/usr/local/cuda/bin:/usr/src/tensorrt/bin:$PATH
QUICK=${QUICK:-0}
ITERS=$([ "$QUICK" = 1 ] && echo 200 || echo 1000)
WARMUP=$([ "$QUICK" = 1 ] && echo 50 || echo 100)
PHASE_S=$([ "$QUICK" = 1 ] && echo 3 || echo 8)

JETSON=0
if [ -f /etc/nv_tegra_release ]; then
  JETSON=1
  GPU_NAME=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo jetson)
else
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
fi
GPU_NAME=${GPU_NAME:-unknown-gpu}
TAG=$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/nvidia//; s/[^a-z0-9]+/-/g; s/^-+|-+$//g')

OUT=reports/gpu/$TAG
BUILD=build/gpu
ENG=$BUILD/engines
BIN=$BUILD/cmake
mkdir -p "$OUT" "$ENG"
: > "$OUT/run.log"
FAILED=()

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
step() {           # step "<name>" cmd...   — run, log, never abort the script
  local name=$1; shift
  log "▶ $name"
  "$@" >> "$OUT/run.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then log "✓ $name"; return 0; fi
  log "✗ $name (exit $rc)"
  FAILED+=("$name")
  return $rc
}

log "GPU: $GPU_NAME  →  $OUT   (quick=$QUICK)"

# ── 0. environment ───────────────────────────────────────────────────────────
{
  echo "date: $(date -u +%FT%TZ)"; echo "host: $(uname -a)"; echo "gpu: $GPU_NAME"; echo
  nvidia-smi 2>/dev/null || true
  [ "$JETSON" = 1 ] && { cat /etc/nv_tegra_release; nvpmodel -q 2>/dev/null || true; }
  echo; nvcc --version 2>/dev/null || true
  echo; dpkg -l 2>/dev/null | grep -E "libnvinfer[0-9]* |tensorrt " || true
  echo; cmake --version 2>/dev/null | head -1; $PY --version
} > "$OUT/env.txt" 2>&1
if [ "$JETSON" = 1 ]; then
  log "Jetson: for stable numbers run 'sudo nvpmodel -m 0 && sudo jetson_clocks' first"
else
  log "Tip: lock GPU clocks for stable numbers: sudo nvidia-smi -lgc <max> (and -rgc to reset)"
fi

# ── 1. Python side ───────────────────────────────────────────────────────────
if [ "${SKIP_PIP:-0}" != 1 ]; then
  step "pip install" $PY -m pip install -q -r requirements-gpu.txt
fi
ORT_GPU=0
$PY -c "import onnxruntime as o, sys; sys.exit('CUDAExecutionProvider' not in o.get_available_providers())" \
  2>/dev/null && ORT_GPU=1
step "quantize (FP16 / INT8 variants)" $PY -m nurosim.deploy quantize
step "trt-prep (frames + EfficientNMS graphs)" $PY -m nurosim.deploy trt-prep --out "$BUILD"
step "python unit tests" $PY -m pytest -q tests

if [ "$ORT_GPU" = 1 ]; then
  step "ORT parity (nvidia-trt profile)" $PY -m nurosim.deploy --reports "$OUT" parity --profile nvidia-trt
  step "ORT benchmark (nvidia-trt profile)" $PY -m nurosim.deploy --reports "$OUT" bench --profile nvidia-trt
  step "ORT gate (nvidia-trt profile)" $PY -m nurosim.deploy --reports "$OUT" gate --profile nvidia-trt
else
  log "onnxruntime-gpu not available — skipping the ORT CUDA/TensorRT EP profile"
  [ "$JETSON" = 1 ] && log "  (Jetson: install the onnxruntime-gpu wheel for your JetPack from the Jetson AI Lab index)"
fi

# ── 2. C++ build + tests ─────────────────────────────────────────────────────
ARCH=${CUDA_ARCH:-native}
step "cmake configure" cmake -S gpu -B "$BIN" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
step "cmake build" cmake --build "$BIN" -j"$(nproc 2>/dev/null || echo 4)"
step "ctest (host + gpu)" ctest --test-dir "$BIN" --output-on-failure --no-tests=error

# ── 3. TensorRT engines ──────────────────────────────────────────────────────
TAIL='^/model\.22/(?!cv2|cv3)'      # YOLOv8 head tail: decode math, kept FP32
build() {
  local name=$1; shift
  if [ -f "$ENG/$name.engine" ]; then log "  engine $name exists — reusing"; return 0; fi
  if [ ! -x "$BIN/nurosim_build_engine" ]; then log "  skip engine $name (C++ build failed)"; return 0; fi
  step "build engine $name" "$BIN/nurosim_build_engine" --timing-cache "$ENG/timing.cache" \
       --out "$ENG/$name.engine" "$@"
}
TRT_MAJOR=0
[ -x "$BIN/nurosim_build_engine" ] && TRT_MAJOR=$("$BIN/nurosim_build_engine" --version | cut -d. -f1)
log "TensorRT major version: $TRT_MAJOR"
if [ "$TRT_MAJOR" -ge 11 ]; then
  # Strongly typed only: precision comes from the ONNX graph, so build each
  # engine from the matching Tier-1 variant instead of FP32 + builder flags.
  build fp32                 --onnx models/nurosim_det_fp32.onnx --max-batch 8
  build fp16                 --onnx models/nurosim_det_fp16.onnx --max-batch 8
  build fp16_nms             --onnx "$BUILD/nurosim_det_fp16_nms.onnx"
  build fp16_mixed_nms       --onnx "$BUILD/nurosim_det_fp16_mixed_nms.onnx"
  build int8_nms             --onnx "$BUILD/nurosim_det_int8_nms.onnx"
  build fp32_nms_eval        --onnx "$BUILD/nurosim_det_fp32_nms_eval.onnx"
  build fp16_nms_eval        --onnx "$BUILD/nurosim_det_fp16_nms_eval.onnx"
  build fp16_mixed_nms_eval  --onnx "$BUILD/nurosim_det_fp16_mixed_nms_eval.onnx"
  build int8_nms_eval        --onnx "$BUILD/nurosim_det_int8_nms_eval.onnx"
  [ "$JETSON" = 1 ] && build fp16_mixed_nms_dla --onnx "$BUILD/nurosim_det_fp16_mixed_nms.onnx" --dla 0
else
  # TensorRT 8.6 / 10.x: FP32 graph + builder precision flags, head tail pinned to FP32
  build fp32                 --onnx models/nurosim_det_fp32.onnx --max-batch 8
  build fp16                 --onnx models/nurosim_det_fp32.onnx --max-batch 8 --fp16
  build fp16_nms             --onnx "$BUILD/nurosim_det_fp32_nms.onnx" --fp16
  build fp16_mixed_nms       --onnx "$BUILD/nurosim_det_fp32_nms.onnx" --fp16 --fp32-layers "$TAIL"
  build int8_nms             --onnx "$BUILD/nurosim_det_int8_nms.onnx" --int8 --fp32-layers "$TAIL"
  build fp32_nms_eval        --onnx "$BUILD/nurosim_det_fp32_nms_eval.onnx"
  build fp16_nms_eval        --onnx "$BUILD/nurosim_det_fp32_nms_eval.onnx" --fp16
  build fp16_mixed_nms_eval  --onnx "$BUILD/nurosim_det_fp32_nms_eval.onnx" --fp16 --fp32-layers "$TAIL"
  build int8_nms_eval        --onnx "$BUILD/nurosim_det_int8_nms_eval.onnx" --int8 --fp32-layers "$TAIL"
  if [ "$JETSON" = 1 ]; then
    build fp16_mixed_nms_dla --onnx "$BUILD/nurosim_det_fp32_nms.onnx" --fp16 --fp32-layers "$TAIL" --dla 0
  fi
fi

# ── 4. latency matrix (each row changes one thing) ───────────────────────────
bench() {
  local id=$1 label=$2 engine=$3; shift 3
  [ -f "$ENG/$engine.engine" ] || { log "  skip $label (no engine)"; return; }
  step "bench $label" "$BIN/nurosim_bench" --engine "$ENG/$engine.engine" --frames "$BUILD/bench.nsfr" \
       --iters "$ITERS" --warmup "$WARMUP" --label "$label" --json "$OUT/bench_${id}.json" "$@"
}
bench 01 "FP32 · CPU pre · CPU NMS"                fp32           --pre cpu
bench 02 "FP16 · CPU pre · CPU NMS"                fp16           --pre cpu
bench 03 "FP16 · GPU pre · CPU NMS"                fp16           --pre gpu
bench 04 "FP16 · GPU pre · EfficientNMS"           fp16_nms       --pre gpu
bench 05 "FP16 · GPU pre · EfficientNMS · graph"   fp16_nms       --pre gpu --cuda-graph
bench 06 "FP16+FP32 tail · GPU pre · NMS · graph"  fp16_mixed_nms --pre gpu --cuda-graph
bench 07 "INT8 · GPU pre · NMS · graph"            int8_nms       --pre gpu --cuda-graph
bench 08 "FP16+FP32 tail · DLA0 · NMS · graph"     fp16_mixed_nms_dla --pre gpu --cuda-graph

for bs in 1 2 4 8; do
  [ -f "$ENG/fp16.engine" ] && step "sweep fp16 batch $bs" "$BIN/nurosim_bench" --engine "$ENG/fp16.engine" \
    --frames "$BUILD/bench.nsfr" --pre gpu --batch "$bs" --iters $((ITERS / 2)) --warmup "$WARMUP" \
    --label "FP16 bs$bs" --json "$OUT/sweep_bs$bs.json"
done

# ── 5. accuracy parity of the TensorRT path ──────────────────────────────────
DETS=()
for e in fp32_nms_eval fp16_nms_eval fp16_mixed_nms_eval int8_nms_eval; do
  [ -f "$ENG/$e.engine" ] || continue
  step "dump detections $e" "$BIN/nurosim_bench" --engine "$ENG/$e.engine" --frames "$BUILD/eval.nsfr" \
       --pre gpu --dump-dets "$OUT/dets_$e.json" && DETS+=("$OUT/dets_$e.json")
done
if [ -f "$ENG/fp16.engine" ]; then   # raw engine + C++ CPU decode/NMS, same eval threshold
  step "dump detections fp16_cpunms_eval" "$BIN/nurosim_bench" --engine "$ENG/fp16.engine" \
       --frames "$BUILD/eval.nsfr" --pre gpu --conf 0.01 --dump-dets "$OUT/dets_fp16_cpunms_eval.json" \
    && DETS+=("$OUT/dets_fp16_cpunms_eval.json")
fi
[ ${#DETS[@]} -gt 0 ] && step "C++ parity report" $PY -m nurosim.deploy cpp-parity --dets "${DETS[@]}" --out "$OUT"

# ── 6. two models on one GPU ─────────────────────────────────────────────────
if [ -f "$ENG/fp16_mixed_nms.engine" ] && [ -f "$ENG/fp16.engine" ]; then
  step "multistream (A 30 Hz detector vs B batch-8 background)" "$BIN/nurosim_multistream" \
       --engine-a "$ENG/fp16_mixed_nms.engine" --engine-b "$ENG/fp16.engine" --batch-b 8 \
       --frames "$BUILD/bench.nsfr" --rate-a-hz 30 --duration-s "$PHASE_S" --json "$OUT/multistream.json"
fi

# ── 7. profiles ──────────────────────────────────────────────────────────────
if command -v nsys >/dev/null 2>&1 && [ -f "$ENG/fp16_nms.engine" ]; then
  for mode in "" "--cuda-graph"; do
    name=nsys_fp16_nms${mode:+_graph}
    step "nsys profile $name" nsys profile --force-overwrite true --trace cuda,nvtx,osrt -o "$OUT/$name" \
         "$BIN/nurosim_bench" --engine "$ENG/fp16_nms.engine" --frames "$BUILD/bench.nsfr" \
         --pre gpu --iters 300 --warmup 50 $mode
    nsys stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum "$OUT/$name.nsys-rep" > "$OUT/$name.txt" 2>/dev/null \
      || nsys stats --report nvtxsum,gpukernsum,cudaapisum "$OUT/$name.nsys-rep" > "$OUT/$name.txt" 2>&1 || true
  done
else
  log "nsys not found — skipping Nsight Systems profiles (install nsight-systems-cli)"
fi
if command -v trtexec >/dev/null 2>&1 && [ -f "$ENG/fp16_nms.engine" ]; then
  # Independent cross-check of the engine's GPU time against our harness
  step "trtexec cross-check" bash -c "trtexec --loadEngine=$ENG/fp16_nms.engine --useCudaGraph \
       --iterations=$ITERS --warmUp=500 > $OUT/trtexec_fp16_nms.txt 2>&1"
fi

# ── 8. summary ───────────────────────────────────────────────────────────────
step "summary" $PY -m nurosim.deploy gpu-report --dir "$OUT" > /dev/null
log "Summary: $OUT/SUMMARY.md"
if [ ${#FAILED[@]} -gt 0 ]; then
  log "${#FAILED[@]} step(s) failed: ${FAILED[*]}"
  exit 1
fi
log "All steps passed."
