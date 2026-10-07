#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 on a cloud GPU VM (Docker + NVIDIA Container Toolkit):
# real Triton Inference Server serving TensorRT plans, Prometheus + Grafana,
# Ray Data batch job, gated registry, shadow comparison.
#
#   bash scripts/platform/vm_run.sh            # QUICK=1 for a short run
#
# Env: USD_PER_HOUR (the VM's hourly price, for $/1M frames), CORPUS_N
#      (default 100000), QUICK=1
# Results: reports/platform/<gpu>-vm/SUMMARY.md; Grafana stays up on
# localhost:3000 (tunnel: ssh -L 3000:localhost:3000 <vm>).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

QUICK=${QUICK:-0}
USD_PER_HOUR=${USD_PER_HOUR:-}
CORPUS_N=${CORPUS_N:-$([ "$QUICK" = 1 ] && echo 5000 || echo 100000)}
DURATION=$([ "$QUICK" = 1 ] && echo 4 || echo 10)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
TAG=$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/nvidia//; s/[^a-z0-9]+/-/g; s/^-+|-+$//g')-vm
OUT=reports/platform/$TAG
REPO=infra/vm/model_repository
URI=sqlite:////workspace/detect2deploy/build/platform/mlruns/registry.db
TAIL_NOTE="plans are built from the registered ONNX with --strongly-typed: the deployed precision is exactly the registered graph's"
mkdir -p "$OUT" build/platform
: > "$OUT/run.log"
FAILED=()

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
step() {
  local name=$1; shift
  log "▶ $name"
  "$@" >> "$OUT/run.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then log "✓ $name"; return 0; fi
  log "✗ $name (exit $rc)"; FAILED+=("$name"); return $rc
}
# Everything Python / TensorRT runs in the platform image, on the host network
# (so it reaches Triton on localhost), with the repo mounted.
RUN=(docker run --rm --gpus all --network host --ipc host -e MLFLOW_DISABLE_AGENT_HINT=1
     -v "$PWD:/workspace/detect2deploy" -w /workspace/detect2deploy d2d-platform)

log "GPU: $GPU_NAME → $OUT (quick=$QUICK)"
{ date -u; nvidia-smi; docker --version; nproc; free -g; } > "$OUT/env.txt" 2>&1

# ── 0. images ────────────────────────────────────────────────────────────────
step "build d2d-gpu image" docker build -q -f gpu/Dockerfile -t d2d-gpu .
step "build d2d-platform image" docker build -q -f infra/vm/Dockerfile.platform -t d2d-platform .
step "build C++ tools" "${RUN[@]}" bash -c "cmake -S gpu -B build/gpu/cmake -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build/gpu/cmake -j\$(nproc)"

# ── 1. fresh gate evidence on *this* GPU (Tier 1/2 harness, ORT TensorRT EP) ─
step "quantize" "${RUN[@]}" python -m detect2deploy.deploy quantize
mkdir -p "$OUT/evidence"
"${RUN[@]}" python -m detect2deploy.deploy --reports "$OUT/evidence" all --profile nvidia-trt >> "$OUT/run.log" 2>&1 \
  && log "✓ gate evidence: PASS" || log "⚠ gate evidence: some targets failed their budget (expected for int8)"

# ── 2. registry ──────────────────────────────────────────────────────────────
rm -rf build/platform/mlruns
reg() { "${RUN[@]}" python -m detect2deploy.platform registry "$@" --uri "$URI"; }
EV=()
step "register fp16_mixed" reg register --model models/d2d_det_fp16_mixed.onnx --target fp16_mixed-trt \
     --gate "$OUT/evidence/gate_nvidia-trt.json" --parity "$OUT/evidence/parity_nvidia-trt.json" \
  && step "promote v1 → production" reg promote --version 1 && EV+=(--event "v1 (fp16_mixed) promoted to production")
step "register int8_trt" reg register --model models/d2d_det_int8_trt.onnx --target int8-trt \
     --gate "$OUT/evidence/gate_nvidia-trt.json" --parity "$OUT/evidence/parity_nvidia-trt.json"
if reg promote --version 2 >> "$OUT/run.log" 2>&1; then EV+=(--event "v2 (int8) promoted — gate passed on this GPU")
else EV+=(--event "v2 (int8) promotion blocked by the registry"); fi
step "registry table" reg list --json "$OUT/registry.json" "${EV[@]}"

# ── 3. TensorRT plans for Triton: production (from the registry) + candidate ─
step "export serving graphs" "${RUN[@]}" python -m detect2deploy.platform export --from-registry production --uri "$URI"
plan() {   # plan <model-name> <onnx>
  mkdir -p "$REPO/$1/1"
  "${RUN[@]}" build/gpu/cmake/d2d_build_engine --onnx "$2" --out "$REPO/$1/1/model.plan" \
      --strongly-typed --max-batch 8 --opt-batch 4 --timing-cache build/platform/timing.cache
  cat > "$REPO/$1/config.pbtxt" <<EOF
name: "$1"
platform: "tensorrt_plan"
max_batch_size: 8
input [ { name: "frames" data_type: TYPE_UINT8 dims: [ 640, 640, 3 ] } ]
output [
  { name: "num_dets" data_type: TYPE_INT32 dims: [ 1 ] },
  { name: "det_boxes" data_type: TYPE_FP32 dims: [ 100, 4 ] },
  { name: "det_scores" data_type: TYPE_FP32 dims: [ 100 ] },
  { name: "det_classes" data_type: TYPE_INT32 dims: [ 100 ] }
]
dynamic_batching { max_queue_delay_microseconds: 500 }
instance_group [ { kind: KIND_GPU count: 1 } ]
EOF
}
log "  $TAIL_NOTE"
step "plan: d2d_det ← registry:production" plan d2d_det build/platform/serve_registry_production_nms.onnx
step "plan: candidate ← int8_trt" plan candidate build/platform/serve_int8_trt_nms.onnx

# ── 4. Triton + Prometheus + Grafana ─────────────────────────────────────────
step "compose up" docker compose -f infra/vm/docker-compose.yml up -d
step "triton ready" "${RUN[@]}" python -c "from detect2deploy.platform.loadgen import wait_ready as w; import sys; \
     sys.exit(0 if w('d2d_det', 600) and w('candidate', 600) else 1)"

# ── 5. serving sweep (configs applied through Triton's model-control API) ────
step "serving sweep (Triton, TensorRT plan)" "${RUN[@]}" python -m detect2deploy.platform serving-bench \
     --backend triton --provider tensorrt --duration "$DURATION" --out "$OUT"
"${RUN[@]}" python -c "from detect2deploy.platform.serving import triton_load_config as c; \
     c('d2d_det', 8, 500, 'localhost:8001')" >> "$OUT/run.log" 2>&1   # restore the default config

# ── 6. shadow on the live server ─────────────────────────────────────────────
step "shadow (candidate vs production)" "${RUN[@]}" python -m detect2deploy.platform shadow \
     --production d2d_det --candidate candidate --frames 300 --out "$OUT"

# ── 7. offboard batch inference ──────────────────────────────────────────────
step "corpus ($CORPUS_N frames)" "${RUN[@]}" python -m detect2deploy.platform corpus --n "$CORPUS_N" \
     --out data/corpus --info "$OUT/corpus.json"
CPUS=$(nproc)
for bs in 8 32; do
  step "batch job (batch $bs)" "${RUN[@]}" python -m detect2deploy.platform batch --corpus data/corpus \
       --model build/platform/serve_registry_production.onnx --provider tensorrt --batch-size "$bs" \
       --actors 1 --labeler-cpus 2 ${USD_PER_HOUR:+--usd-per-hour "$USD_PER_HOUR"} --out "$OUT" --tag "bs$bs"
done

# ── 8. summary ───────────────────────────────────────────────────────────────
step "summary" "${RUN[@]}" python -m detect2deploy.platform report --dir "$OUT"
log "Summary: $OUT/SUMMARY.md   ($CPUS CPUs)"
log "Grafana: ssh -L 3000:localhost:3000 <vm>  →  http://localhost:3000   (stop: docker compose -f infra/vm/docker-compose.yml down)"
if [ ${#FAILED[@]} -gt 0 ]; then log "${#FAILED[@]} step(s) failed: ${FAILED[*]}"; exit 1; fi
log "All steps passed."
