# Run environment for the Tier-2 GPU work on x86 + NVIDIA GPU.
#   docker build -f gpu/Dockerfile -t nurosim-gpu .
#   docker run --rm --gpus all -v "$PWD/reports:/workspace/nurosim/reports" nurosim-gpu
#
# The TensorRT NGC image brings CUDA, cuDNN, TensorRT, trtexec and the
# headers/libs the C++ tools link against. 25.01 = TensorRT 10.8 / CUDA 12.8;
# on older drivers pick an older tag (e.g. 24.08 = TensorRT 10.3 / CUDA 12.6).
ARG TRT_IMAGE=nvcr.io/nvidia/tensorrt:25.01-py3
FROM ${TRT_IMAGE}

# Ubuntu 24.04-based images mark the system Python as externally managed (PEP 668)
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install --no-cache-dir "cmake>=3.24" ninja

WORKDIR /workspace/nurosim
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY . .
ENV SKIP_PIP=1
CMD ["bash", "scripts/gpu/run_all.sh"]
