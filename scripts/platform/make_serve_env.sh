#!/usr/bin/env bash
# Build the PyTriton serving environment: Python 3.12 + numpy < 2 (PyTriton's
# requirement) + onnxruntime(-gpu). Prints the interpreter path on the last line.
#
#   export NUROSIM_SERVE_PYTHON=$(bash scripts/platform/make_serve_env.sh --gpu | tail -1)
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:---cpu}
ENV=build/serve-venv

if ! command -v uv >/dev/null 2>&1; then python3 -m pip install -q uv; fi
[ -x "$ENV/bin/python" ] || uv venv -q --python 3.12 "$ENV"          # uv fetches 3.12 if needed
PY="$PWD/$ENV/bin/python"
uv pip install -q --python "$PY" -r requirements-serve.txt
if [ "$MODE" = "--gpu" ]; then
  uv pip install -q --python "$PY" onnxruntime-gpu
  # onnxruntime-gpu may target a different CUDA major than the host toolkit:
  # add that major's runtime wheels (loaded through onnxruntime.preload_dlls)
  if ! "$PY" -c "from nurosim.inference.runtime import OrtRuntime as R; import sys; \
       sys.exit(0 if R('models/nurosim_det_fp32.onnx', provider='cuda').provider_active else 1)" >/dev/null 2>&1; then
    CU=$("$PY" -c "from onnxruntime.capi import build_and_package_info as b; print(getattr(b, 'cuda_version', '12').split('.')[0])")
    uv pip install -q --python "$PY" "nvidia-cuda-runtime-cu$CU" "nvidia-cublas-cu$CU" "nvidia-cudnn-cu$CU" \
       "nvidia-cufft-cu$CU" "nvidia-curand-cu$CU" "nvidia-cuda-nvrtc-cu$CU" || true
  fi
  "$PY" -c "from nurosim.inference.runtime import OrtRuntime as R; \
            print('serve env CUDA EP active:', R('models/nurosim_det_fp32.onnx', provider='cuda').provider_active)" >&2 || true
else
  uv pip install -q --python "$PY" onnxruntime
fi
"$PY" -c "import numpy, pytriton, onnxruntime; print('serve env: numpy', numpy.__version__, \
          'onnxruntime', onnxruntime.__version__)" >&2
echo "$PY"
