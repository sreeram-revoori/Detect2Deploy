#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 on Google Colab (T4 GPU runtime): serving, batch inference, registry,
# shadow — no Docker needed (Triton runs via PyTriton).
#
#   bash scripts/gpu/colab_setup.sh          # once per runtime (TensorRT, ORT GPU)
#   bash scripts/platform/colab_run.sh       # ~25 min;  QUICK=1 → ~8 min
#
# Results: reports/platform/<gpu>/SUMMARY.md. Each step is logged; a failing
# step is recorded and the run continues.
#
# Env: QUICK=1, USD_PER_HOUR (cost model; default 0.35 = assumed T4 hourly
#      price), PROVIDER (tensorrt | cuda), SKIP_PIP=1
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/usr/local/cuda/bin:$PATH MLFLOW_DISABLE_AGENT_HINT=1
# Shared token for Triton's protected endpoints (statistics etc.; PyTriton >= 0.7)
export D2D_TRITON_TOKEN=${D2D_TRITON_TOKEN:-$(python3 -c "import secrets; print(secrets.token_hex(16))")}

PY=${PYTHON:-python3}
QUICK=${QUICK:-0}
PROVIDER=${PROVIDER:-tensorrt}
USD_PER_HOUR=${USD_PER_HOUR:-0.35}
DURATION=$([ "$QUICK" = 1 ] && echo 4 || echo 10)
CORPUS_N=$([ "$QUICK" = 1 ] && echo 4000 || echo 20000)
SHADOW_N=$([ "$QUICK" = 1 ] && echo 100 || echo 300)

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
TAG=$(echo "${GPU_NAME:-cpu}" | tr '[:upper:]' '[:lower:]' | sed -E 's/nvidia//; s/[^a-z0-9]+/-/g; s/^-+|-+$//g')
OUT=reports/platform/$TAG
URI="sqlite:///$PWD/build/platform/mlruns/registry.db"
mkdir -p "$OUT" build/platform
: > "$OUT/run.log"
FAILED=()
EVENTS=()

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
step() {
  local name=$1; shift
  log "▶ $name"
  "$@" >> "$OUT/run.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then log "✓ $name"; return 0; fi
  log "✗ $name (exit $rc)"; FAILED+=("$name"); return $rc
}

log "GPU: ${GPU_NAME:-none} → $OUT  (quick=$QUICK, provider=$PROVIDER)"
{ date -u; nvidia-smi 2>/dev/null; $PY --version; nproc; free -g 2>/dev/null; } > "$OUT/env.txt" 2>&1

# ── 0. dependencies ──────────────────────────────────────────────────────────
if [ "${SKIP_PIP:-0}" != 1 ]; then
  step "pip install platform deps" $PY -m pip install -q "tritonclient[grpc]" "ray[data]" \
       mlflow nvidia-ml-py pyarrow
fi
# PyTriton needs numpy < 2 (Colab: Python 3.13, numpy 2) → its own Python 3.12 env
log "▶ serving environment (Python 3.12, numpy < 2, onnxruntime-gpu)"
SERVE_PY=$(bash scripts/platform/make_serve_env.sh --gpu 2>> "$OUT/run.log" | tail -1)
if [ -x "${SERVE_PY:-}" ]; then
  export D2D_SERVE_PYTHON=$SERVE_PY; log "✓ serving environment: $SERVE_PY"
else
  log "✗ serving environment"; FAILED+=("serving environment")
fi

# ── 1. models + serving graphs ───────────────────────────────────────────────
step "quantize (FP16 / INT8 variants)" $PY -m detect2deploy.deploy quantize
step "export serving graphs" $PY -m detect2deploy.platform export

# ── 2. serving: dynamic batching sweep ───────────────────────────────────────
step "serving sweep (PyTriton, $PROVIDER)" $PY -m detect2deploy.platform serving-bench \
     --model build/platform/serve_fp16_mixed.onnx --provider "$PROVIDER" --backend pytriton \
     --duration "$DURATION" --out "$OUT"

# ── 3. offboard batch inference ──────────────────────────────────────────────
step "corpus ($CORPUS_N frames)" $PY -m detect2deploy.platform corpus --n "$CORPUS_N" --out data/corpus \
     --info "$OUT/corpus.json"
for bs in 8 32; do
  step "batch job (Ray Data, batch $bs)" $PY -m detect2deploy.platform batch --corpus data/corpus \
       --model build/platform/serve_fp16_mixed.onnx --provider "$PROVIDER" --batch-size "$bs" \
       --actors 1 --labeler-cpus 1 --usd-per-hour "$USD_PER_HOUR" --out "$OUT" --tag "bs$bs"
done

# ── 4. registry: promotion needs gate evidence for the exact file ────────────
# Evidence = the Tier-2 T4 run's parity/benchmark reports, re-gated here.
mkdir -p "$OUT/evidence"
cp reports/gpu/tesla-t4/parity_nvidia-trt.json reports/gpu/tesla-t4/benchmark_nvidia-trt.json "$OUT/evidence/"
$PY -m detect2deploy.deploy --reports "$OUT/evidence" gate --profile nvidia-trt \
    --targets fp32-cuda,fp16-trt,int8-trt,int8_tail32-trt,fp16_mixed-trt >> "$OUT/run.log" 2>&1 || true
rm -rf build/platform/mlruns
reg() { $PY -m detect2deploy.platform registry "$@" --uri "$URI"; }
if step "register fp16_mixed (evidence: fp16_mixed-trt)" reg register --model models/d2d_det_fp16_mixed.onnx \
     --target fp16_mixed-trt --gate "$OUT/evidence/gate_nvidia-trt.json" --parity "$OUT/evidence/parity_nvidia-trt.json"; then
  step "promote v1 → production" reg promote --version 1 && EVENTS+=("v1 (fp16_mixed) promoted to production")
fi
if step "register int8_trt (evidence: int8-trt)" reg register --model models/d2d_det_int8_trt.onnx \
     --target int8-trt --gate "$OUT/evidence/gate_nvidia-trt.json" --parity "$OUT/evidence/parity_nvidia-trt.json"; then
  # Expected to be refused: the gate failed int8-trt on cone AP (or the file differs from the evidence)
  if reg promote --version 2 >> "$OUT/run.log" 2>&1; then
    log "⚠ v2 (int8) promotion was NOT blocked — unexpected"; EVENTS+=("v2 (int8) promoted — unexpected")
  else
    log "✓ v2 (int8) promotion blocked by the registry"; EVENTS+=("v2 (int8) promotion blocked by the registry")
  fi
fi
EV_ARGS=(); for e in "${EVENTS[@]}"; do EV_ARGS+=(--event "$e"); done
step "registry table" reg list --json "$OUT/registry.json" "${EV_ARGS[@]}"

# ── 5. shadow: candidate vs whatever 'production' resolves to ────────────────
# The serving env has no MLflow: resolve 'production' here and hand the server a file
step "export registry:production" $PY -m detect2deploy.platform export --from-registry production --uri "$URI"
log "▶ shadow (production=registry:production vs candidate=int8_trt)"
"${D2D_SERVE_PYTHON:-$PY}" -m detect2deploy.platform serve --provider "$PROVIDER" --max-batch 8 --queue-delay-us 0 \
    --model production=build/platform/serve_registry_production.onnx \
    --model candidate=build/platform/serve_int8_trt.onnx >> "$OUT/server_shadow.log" 2>&1 &
SERVER=$!
if $PY -c "from detect2deploy.platform.loadgen import wait_ready; import sys; sys.exit(0 if wait_ready('candidate') and wait_ready('production') else 1)"; then
  step "shadow comparison ($SHADOW_N frames)" $PY -m detect2deploy.platform shadow --production production \
       --candidate candidate --frames "$SHADOW_N" --out "$OUT"
else
  log "✗ shadow server not ready (see server_shadow.log)"; FAILED+=("shadow server")
fi
kill -INT $SERVER 2>/dev/null; wait $SERVER 2>/dev/null

# ── 6. summary ───────────────────────────────────────────────────────────────
step "summary" $PY -m detect2deploy.platform report --dir "$OUT" > /dev/null
log "Summary: $OUT/SUMMARY.md"
if [ ${#FAILED[@]} -gt 0 ]; then log "${#FAILED[@]} step(s) failed: ${FAILED[*]}"; exit 1; fi
log "All steps passed."
