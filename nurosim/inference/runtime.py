"""
runtime.py
Thin ONNX Runtime session wrapper with explicit execution-provider handling.

The same ONNX artefact runs on:
    cpu       → CPUExecutionProvider            (CI, x86/ARM hosts)
    coreml    → CoreMLExecutionProvider         (Apple Neural Engine / GPU)
    cuda      → CUDAExecutionProvider           (NVIDIA, framework kernels)
    tensorrt  → TensorrtExecutionProvider       (NVIDIA, TensorRT engines)

A classic deployment bug is a session that silently falls back to CPU when the
accelerator EP fails to load. `active_providers` records what actually ran so
the budget gate can fail on it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROVIDERS = {
    "cpu":      "CPUExecutionProvider",
    "coreml":   "CoreMLExecutionProvider",
    "cuda":     "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}

DEFAULT_PROVIDER_OPTIONS: Dict[str, Dict[str, Any]] = {
    "coreml": {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
    "tensorrt": {
        "trt_engine_cache_enable": True,          # don't rebuild engines every start
        "trt_engine_cache_path": "models/trt_cache",
        "trt_timing_cache_enable": True,
    },
}


class ProviderUnavailableError(RuntimeError):
    pass


def fix_dim_params(model_bytes: bytes, dims: Dict[str, int]) -> bytes:
    """Pin symbolic dims (e.g. {"batch": 1}) — static shapes let CoreML /
    TensorRT build a single specialised graph instead of a dynamic one."""
    import onnx
    from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed

    model = onnx.load_from_string(model_bytes)
    for name, value in dims.items():
        make_dim_param_fixed(model.graph, name, value)
    return model.SerializeToString()


class OrtRuntime:
    """One InferenceSession, one input, one output."""

    def __init__(self,
                 model_path: str,
                 provider: str = "cpu",
                 provider_options: Optional[Dict[str, Any]] = None,
                 static_batch: Optional[int] = None,
                 intra_op_threads: Optional[int] = None,
                 graph_opt: str = "all"):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("onnxruntime is required: pip install onnxruntime") from e

        if provider in ("cuda", "tensorrt") and hasattr(ort, "preload_dlls"):
            # Load the CUDA / cuDNN runtime libs this onnxruntime-gpu build needs
            # from the nvidia-* pip wheels when present (e.g. a CUDA 12 ORT build
            # on a CUDA 13 host). No-op when they aren't installed.
            try:
                ort.preload_dlls()
            except Exception:                         # pragma: no cover
                pass

        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider '{provider}', expected one of {list(PROVIDERS)}")
        ep_name = PROVIDERS[provider]
        if ep_name not in ort.get_available_providers():
            raise ProviderUnavailableError(
                f"{ep_name} is not available in this onnxruntime build "
                f"(available: {ort.get_available_providers()})")

        with open(model_path, "rb") as f:
            model_bytes = f.read()
        if static_batch:
            model_bytes = fix_dim_params(model_bytes, {"batch": static_batch})

        so = ort.SessionOptions()
        # "disable" hands the graph to the EP untouched — e.g. so ORT's own
        # Q/DQ rewrites can't change what TensorRT quantises.
        so.graph_optimization_level = {
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        }[graph_opt]
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
        so.log_severity_level = 3

        opts = {**DEFAULT_PROVIDER_OPTIONS.get(provider, {}), **(provider_options or {})}
        providers: List[Any] = [(ep_name, opts)] if opts else [ep_name]
        if ep_name != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")     # for unsupported nodes

        t0 = time.perf_counter()
        self.session = ort.InferenceSession(model_bytes, sess_options=so, providers=providers)
        self.session_init_ms = (time.perf_counter() - t0) * 1000

        inp = self.session.get_inputs()[0]
        self.input_name   = inp.name
        self.input_shape  = inp.shape
        self.input_dtype  = {"tensor(float16)": np.float16,
                             "tensor(uint8)": np.uint8}.get(inp.type, np.float32)
        self.output_name  = self.session.get_outputs()[0].name
        self.provider     = provider
        self.model_path   = model_path
        self.static_batch = static_batch
        self.active_providers = self.session.get_providers()

        if self.active_providers[0] != ep_name:
            logger.warning("Requested %s but session is running on %s",
                           ep_name, self.active_providers)

    @property
    def provider_active(self) -> bool:
        return self.active_providers[0] == PROVIDERS[self.provider]

    def run(self, batch: np.ndarray) -> np.ndarray:
        """NCHW float32 in → raw head output out (host memory, so the call is
        synchronous: wall-clock timing around it is valid even on GPU EPs)."""
        if batch.dtype != self.input_dtype:
            batch = batch.astype(self.input_dtype)
        return self.session.run([self.output_name], {self.input_name: batch})[0]
