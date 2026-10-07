"""
quantize.py
FP32 ONNX → FP16 and INT8 variants.

INT8 uses post-training static quantisation in QDQ format (QuantizeLinear /
DequantizeLinear pairs around each op). QDQ is the format TensorRT consumes for
explicit quantisation, so the same artefact runs on the ORT CPU EP here and on
the TensorRT EP on NVIDIA hardware. Weights and activations are symmetric INT8
for the same reason.

FP16 comes in the same two flavours (fp16 / fp16_mixed): the decode tail
holds pixel coordinates, and FP16's 0.5 px step above 512 hurts tiny boxes.

INT8 variants (the first two let the parity report show the trade-off):
    int8       – backbone + neck + head convs in INT8; the head "tail" (DFL
                 softmax, box decode arithmetic, class sigmoid, concat) stays
                 FP32. Those ops carry pixel coordinates and probabilities
                 whose dynamic range a single INT8 scale represents poorly.
    int8_full  – every quantisable op in INT8 (the naive baseline).
    int8_trt   – as int8, but conv biases stay float (what TensorRT expects).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Dict, List, Optional

import numpy as np

from detect2deploy.deploy.seeds import CALIB_SEED
from detect2deploy.inference.preprocess import preprocess_batch
from detect2deploy.scenario_generator import generate_scenario, WEATHER_CONDITIONS

logger = logging.getLogger(__name__)


# ── Calibration data ──────────────────────────────────────────────────────────

def calibration_frame(i: int, base_seed: int = CALIB_SEED) -> np.ndarray:
    """Weather-stratified frames: an INT8 scale fitted only to bright 'clear'
    frames clips the dark night / low-contrast fog activations."""
    wx = WEATHER_CONDITIONS[i % len(WEATHER_CONDITIONS)]
    return generate_scenario(base_seed + i, seed=base_seed + i, weather=wx).frame


try:
    from onnxruntime.quantization import CalibrationDataReader as _ReaderBase
except ImportError:                                    # pragma: no cover
    _ReaderBase = object


class ScenarioCalibrationReader(_ReaderBase):
    """Feeds calibration tensors through the *same* preprocessing as runtime."""

    def __init__(self, input_name: str, n_frames: int = 128,
                 imgsz: int = 640, base_seed: int = CALIB_SEED):
        self.input_name = input_name
        self.n_frames   = n_frames
        self.imgsz      = imgsz
        self.base_seed  = base_seed
        self.rewind()

    def __len__(self) -> int:
        return self.n_frames

    def rewind(self) -> None:
        self.set_range(0, self.n_frames)

    def set_range(self, start_index: int, end_index: int) -> None:
        """Used by ORT's strided calibration to feed the data in chunks."""
        self._it = (calibration_frame(i, self.base_seed)
                    for i in range(start_index, min(end_index, self.n_frames)))

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        frame = next(self._it, None)
        if frame is None:
            return None
        batch, _ = preprocess_batch([frame], self.imgsz)
        return {self.input_name: batch}


# ── Helpers ───────────────────────────────────────────────────────────────────

def head_tail_nodes(model) -> List[str]:
    """Nodes of the final Detect module that are NOT in its conv branches
    (cv2 = box regression convs, cv3 = class convs)."""
    ids = [int(m.group(1)) for n in model.graph.node
           if (m := re.match(r"/model\.(\d+)/", n.name))]
    if not ids:
        return []
    idx = max(ids)
    prefix = f"/model.{idx}/"
    return [n.name for n in model.graph.node
            if n.name.startswith(prefix)
            and not n.name.startswith((prefix + "cv2", prefix + "cv3"))]


def _calib_stride(n: int, max_stride: int = 16) -> int:
    """Largest chunk size <= max_stride that divides n (ORT requires it)."""
    return next(k for k in range(min(n, max_stride), 0, -1) if n % k == 0)


def _tag_variant(path: str, variant: str, extra: Optional[Dict[str, str]] = None) -> None:
    import onnx
    model = onnx.load(path)
    meta = {p.key: p.value for p in model.metadata_props}
    # Key keeps the project's original name: it lives inside the model bytes, and
    # renaming it would change every model hash the gate evidence is bound to.
    meta["nurosim_variant"] = variant
    meta.update(extra or {})
    del model.metadata_props[:]
    for k, v in meta.items():
        model.metadata_props.add(key=k, value=str(v))
    onnx.save(model, path)


# ── FP16 ──────────────────────────────────────────────────────────────────────

def _dedupe_nodes(model) -> int:
    """The ORT float16 converter can insert the same Cast twice when a tensor
    feeds several blocked nodes (identical name, input and output), which
    makes the graph invalid. Drop exact duplicates."""
    seen, keep = set(), []
    for n in model.graph.node:
        key = n.SerializeToString()
        if key not in seen:
            seen.add(key)
            keep.append(n)
    removed = len(model.graph.node) - len(keep)
    del model.graph.node[:]
    model.graph.node.extend(keep)
    return removed


def to_fp16(fp32_path: str, out_path: str, keep_head_tail_fp32: bool = False) -> str:
    """Weights + compute in FP16, FP32 I/O so pre/post-processing is unchanged.

    keep_head_tail_fp32: leave the Detect-head decode in FP32. FP16 has 0.5 px
    resolution for values in [512, 1024), and the tail outputs pixel
    coordinates up to 640, which is enough to cost 4–5 px cones IoU.
    """
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(fp32_path)
    block = head_tail_nodes(model) if keep_head_tail_fp32 else None
    model = convert_float_to_float16(model, keep_io_types=True, node_block_list=block)
    _dedupe_nodes(model)
    onnx.save(model, out_path)
    variant = "fp16_mixed" if keep_head_tail_fp32 else "fp16"
    _tag_variant(out_path, variant, {"fp32_excluded_nodes": str(len(block or []))})
    logger.info("%s → %s (%.1f MB, %d nodes kept FP32)", variant.upper(), out_path,
                os.path.getsize(out_path) / 1e6, len(block or []))
    return out_path


# ── INT8 ──────────────────────────────────────────────────────────────────────

def to_int8(fp32_path: str,
            out_path: str,
            n_calib: int = 128,
            method: str = "minmax",
            per_channel: bool = True,
            symmetric_activations: bool = True,
            keep_head_tail_fp32: bool = True,
            quantize_bias: bool = True,
            imgsz: int = 640) -> str:
    """quantize_bias=False keeps conv biases in float. TensorRT's ONNX parser
    rejects the INT32 bias DequantizeLinear nodes ORT emits by default
    ("IDequantizeLayer can only run in INT8/FP8/FP4/INT4"). On the ORT CPU EP
    both forms measured identical (mAP@0.5 0.9585, 12.6 ms on M4); int8 keeps
    INT32 biases only so the committed CPU reports stay valid."""
    import onnx
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    methods = {"minmax": CalibrationMethod.MinMax,
               "entropy": CalibrationMethod.Entropy,
               "percentile": CalibrationMethod.Percentile}

    with tempfile.TemporaryDirectory() as tmp:
        prep = os.path.join(tmp, "prep.onnx")
        # ONNX shape info is already complete from export; ORT symbolic inference
        # trips over the dynamic-batch Reshapes, so skip it.
        quant_pre_process(fp32_path, prep, skip_symbolic_shape=True)
        model = onnx.load(prep)
        exclude = head_tail_nodes(model) if keep_head_tail_fp32 else []
        reader = ScenarioCalibrationReader(model.graph.input[0].name, n_calib, imgsz)

        quantize_static(
            prep, out_path, reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8 if symmetric_activations else QuantType.QUInt8,
            calibrate_method=methods[method],
            nodes_to_exclude=exclude,
            extra_options={
                "ActivationSymmetric": symmetric_activations,
                "WeightSymmetric": True,
                "QuantizeBias": quantize_bias,
                # Calibrate in chunks: the calibrator keeps every intermediate
                # activation of every frame in memory (~GBs for 128 frames).
                # NB: don't use CalibMaxIntermediateOutputs for this — in ORT
                # 1.30 MinMax drops each flushed chunk instead of merging it.
                "CalibStridedMinMax": _calib_stride(n_calib),
            },
        )

    variant = ("int8" if keep_head_tail_fp32 else "int8_full") + ("" if quantize_bias else "_trt")
    _tag_variant(out_path, variant, {
        "calib_frames": str(n_calib), "calib_method": method,
        "fp32_excluded_nodes": str(len(exclude)),
    })
    logger.info("%s → %s (%.1f MB, %d nodes kept FP32, %d calib frames, %s)",
                variant.upper(), out_path, os.path.getsize(out_path) / 1e6,
                len(exclude), n_calib, method)
    return out_path
