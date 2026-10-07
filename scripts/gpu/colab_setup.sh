#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Google Colab (Ubuntu 22.04, T4 GPU runtime) setup for scripts/gpu/run_all.sh.
# Colab has the CUDA toolkit but not TensorRT: install the TensorRT 10 dev
# packages from NVIDIA's apt repo, optionally Nsight Systems, and the Python deps.
#
#   bash scripts/gpu/colab_setup.sh
#   TRT_VERSION=10.13.3.9-1+cuda12.9 bash scripts/gpu/colab_setup.sh   # pin another version
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if ! nvidia-smi > /dev/null 2>&1; then
  echo "No NVIDIA GPU visible. In Colab: Runtime > Change runtime type > T4 GPU (not TPU)."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# CUDA toolkit on PATH (Colab installs it under /usr/local/cuda)
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version | tail -2 || { echo "nvcc not found"; exit 1; }

# NVIDIA CUDA apt repo (hosts the TensorRT and Nsight Systems packages)
if ! grep -rqs "developer.download.nvidia.com/compute/cuda/repos/ubuntu2204" /etc/apt/sources.list.d/; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb > /dev/null
fi
apt-get update -qq

# TensorRT 10.x for CUDA 12 (onnxruntime-gpu's TensorRT EP links libnvinfer.so.10)
V=${TRT_VERSION:-$(apt-cache madison libnvinfer-dev | awk '{print $3}' | grep -E '^10\..*\+cuda12' | head -1)}
echo "Installing TensorRT $V (~1.5 GB download)"
apt-get install -y -qq --no-install-recommends --allow-downgrades --allow-change-held-packages \
  "libnvinfer-dev=$V" "libnvinfer-headers-dev=$V" "libnvinfer-plugin-dev=$V" \
  "libnvinfer-headers-plugin-dev=$V" "libnvonnxparsers-dev=$V" "libnvinfer-bin=$V" > /dev/null
ln -sf /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec

# Nsight Systems CLI (optional — profiles are skipped without it)
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
python3 -c "import onnxruntime as o; print('onnxruntime', o.__version__, o.get_available_providers())"

echo "Setup done. Next:  QUICK=1 SKIP_PIP=1 bash scripts/gpu/run_all.sh"
