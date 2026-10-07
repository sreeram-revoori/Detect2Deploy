#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Google Colab (T4 GPU runtime) setup for scripts/gpu/run_all.sh.
# Colab ships the CUDA toolkit but not TensorRT. This installs, from NVIDIA's
# apt repo for *this* Ubuntu release, the TensorRT 10 build that matches the
# installed CUDA toolkit, plus Nsight Systems and the Python deps.
#
#   bash scripts/gpu/colab_setup.sh
#   TRT_VERSION=10.14.1.48-1+cuda13.0 bash scripts/gpu/colab_setup.sh   # force a version
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
cd "$(dirname "$0")/../.."

if ! nvidia-smi > /dev/null 2>&1; then
  echo "No NVIDIA GPU visible. In Colab: Runtime > Change runtime type > T4 GPU (not TPU)."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

export PATH=/usr/local/cuda/bin:$PATH
CUDA_VER=$(nvcc --version | sed -nE 's/.*release ([0-9]+)\.([0-9]+).*/\1.\2/p')
CUDA_MAJOR=${CUDA_VER%%.*}
CUDA_MINOR=${CUDA_VER#*.}
. /etc/os-release
DIST=ubuntu${VERSION_ID//./}
echo "CUDA toolkit $CUDA_VER on $DIST"

# NVIDIA CUDA apt repo for this release (drop any repo for another release —
# e.g. one added by an earlier version of this script — or apt can't resolve)
for f in /etc/apt/sources.list.d/cuda-ubuntu*.list; do
  [ -e "$f" ] && [[ "$f" != *"cuda-$DIST-"* ]] && { echo "removing mismatched repo $f"; rm -f "$f"; }
done
if ! grep -rqs "compute/cuda/repos/$DIST/" /etc/apt/sources.list.d/ /etc/apt/sources.list; then
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$DIST/x86_64/cuda-keyring_1.1-1_all.deb"
  dpkg -i cuda-keyring_1.1-1_all.deb > /dev/null
fi
apt-get update -qq 2>&1 | grep -v "^W:" || true

# Newest TensorRT 10.x built for this CUDA major with a minor <= the toolkit's
# (10.x because onnxruntime-gpu's TensorRT EP links libnvinfer.so.10)
pick_trt() {
  apt-cache madison libnvinfer-dev | awk '{print $3}' | while read -r v; do
    tag=${v##*+cuda}; maj=${tag%%.*}; min=${tag#*.}
    [[ "$v" == 10.* && "$maj" == "$CUDA_MAJOR" && "$min" -le "$CUDA_MINOR" ]] && { echo "$v"; break; }
  done
}
V=${TRT_VERSION:-$(pick_trt)}
if [ -z "$V" ]; then
  echo "No TensorRT 10 package for CUDA $CUDA_VER in the $DIST repo. Available:"
  apt-cache madison libnvinfer-dev | awk '{print "  " $3}' | head -20
  exit 1
fi
echo "Installing TensorRT $V (~1.5 GB download)"
apt-get install -y -q --no-install-recommends --allow-downgrades --allow-change-held-packages \
  "libnvinfer-dev=$V" "libnvinfer-headers-dev=$V" "libnvinfer-plugin-dev=$V" \
  "libnvinfer-headers-plugin-dev=$V" "libnvonnxparsers-dev=$V" "libnvinfer-bin=$V" \
  | grep -E "^(Setting up libnv|E:)" || true
dpkg -s libnvinfer-dev > /dev/null 2>&1 || { echo "TensorRT install failed"; exit 1; }
ln -sf /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec

# Nsight Systems CLI (optional — the profiling step is skipped without it)
NSYS_PKG=$(apt-cache search --names-only '^nsight-systems-20[0-9.]+$' | awk '{print $1}' | sort -V | tail -1)
if [ -n "$NSYS_PKG" ] && apt-get install -y -qq --no-install-recommends "$NSYS_PKG" > /dev/null 2>&1; then
  NSYS_BIN=$(ls -d /opt/nvidia/nsight-systems/*/bin 2>/dev/null | sort -V | tail -1)
  [ -n "$NSYS_BIN" ] && ln -sf "$NSYS_BIN/nsys" /usr/local/bin/nsys
  echo "Nsight Systems: $(nsys --version 2>/dev/null || echo installed)"
else
  echo "Nsight Systems not installed — profiling step will be skipped"
fi

# Python: CUDA build of ONNX Runtime (replaces any CPU build), onnx, cmake
pip uninstall -y -q onnxruntime 2>/dev/null || true
pip install -q onnx onnxruntime-gpu "cmake>=3.24" pytest pyyaml

# The pip onnxruntime-gpu build may target a different CUDA major than the
# toolkit; if its CUDA EP can't start, add that major's runtime wheels
# (loaded via onnxruntime.preload_dlls, see detect2deploy/inference/runtime.py).
ort_cuda_ok() {
  python3 - <<'EOF'
import sys, onnxruntime as ort
try:
    ort.preload_dlls()
except Exception:
    pass
s = ort.InferenceSession("models/d2d_det_fp32.onnx", providers=["CUDAExecutionProvider"])
sys.exit(0 if s.get_providers()[0] == "CUDAExecutionProvider" else 1)
EOF
}
if ort_cuda_ok 2>/dev/null; then
  echo "onnxruntime CUDA EP: OK"
else
  ORT_CUDA=$(python3 -c "from onnxruntime.capi import build_and_package_info as b; print(getattr(b, 'cuda_version', '12').split('.')[0])")
  echo "onnxruntime-gpu is built for CUDA $ORT_CUDA; installing its CUDA runtime wheels"
  pip install -q "nvidia-cuda-runtime-cu$ORT_CUDA" "nvidia-cublas-cu$ORT_CUDA" "nvidia-cudnn-cu$ORT_CUDA" \
                 "nvidia-cufft-cu$ORT_CUDA" "nvidia-curand-cu$ORT_CUDA" "nvidia-cuda-nvrtc-cu$ORT_CUDA" || true
  if ort_cuda_ok 2>/dev/null; then echo "onnxruntime CUDA EP: OK"
  else echo "onnxruntime CUDA EP still unavailable — the Python ORT profile will be skipped (C++ path unaffected)"; fi
fi

echo "Setup done. Next:  QUICK=1 SKIP_PIP=1 bash scripts/gpu/run_all.sh"
