"""
trt_prep.py
Inputs for the C++ / TensorRT tools in gpu/.

  * Frame sets as raw binaries ("NSFR" header + BGR uint8), so the C++ side
    needs no image decoder:  bench.nsfr (BENCH_SEED) and eval.nsfr (EVAL_SEED,
    the same held-out scenes the Python parity check uses).
  * ONNX graphs with TensorRT's EfficientNMS plugin appended, so decode + NMS
    run on the GPU inside the engine and only ~100 boxes come back to the host
    instead of the 8 x 8400 raw head tensor.
"""

from __future__ import annotations

import logging
import os
import struct
from typing import Dict, Iterable

import numpy as np

from nurosim.deploy.seeds import BENCH_SEED, EVAL_SEED
from nurosim.scenario_generator import generate_scenario

logger = logging.getLogger(__name__)

NMS_OUTPUTS = ("num_dets", "det_boxes", "det_scores", "det_classes")


def write_frames(path: str, frames: Iterable[np.ndarray]) -> str:
    frames = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]
    h, w, c = frames[0].shape
    with open(path, "wb") as f:
        f.write(b"NSFR" + struct.pack("<4I", len(frames), h, w, c))
        for fr in frames:
            f.write(fr.tobytes())
    return path


def read_frames(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        assert f.read(4) == b"NSFR", "not an NSFR frame file"
        n, h, w, c = struct.unpack("<4I", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h, w, c)


def add_efficient_nms(src: str, dst: str, score_threshold: float = 0.25,
                      iou_threshold: float = 0.6, max_det: int = 100) -> str:
    """Append EfficientNMS_TRT to a YOLOv8 head output (B, 4 + nc, A).

    Boxes go in as (cx, cy, w, h) — box_coding=1 — and come out as xyxy in
    model space; scores are already sigmoided (score_activation=0); NMS is
    per class (class_agnostic=0), matching the Python/C++ CPU decode.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(src)
    g = model.graph
    head = g.output[0]
    nc = head.type.tensor_type.shape.dim[1].dim_value - 4
    batch = head.type.tensor_type.shape.dim[0]
    batch_dim = batch.dim_param or batch.dim_value

    g.initializer.append(numpy_helper.from_array(np.array([4, nc], dtype=np.int64), "nms_split"))
    g.node.extend([
        helper.make_node("Transpose", [head.name], ["nms_in"], name="nms/Transpose", perm=[0, 2, 1]),
        helper.make_node("Split", ["nms_in", "nms_split"], ["nms_boxes", "nms_scores"],
                         name="nms/Split", axis=2),
        helper.make_node(
            "EfficientNMS_TRT", ["nms_boxes", "nms_scores"], list(NMS_OUTPUTS),
            name="nms/EfficientNMS_TRT", domain="TRT",
            plugin_version="1", background_class=-1, box_coding=1, score_activation=0,
            class_agnostic=0, iou_threshold=float(iou_threshold),
            score_threshold=float(score_threshold), max_output_boxes=int(max_det)),
    ])
    del g.output[:]
    g.output.extend([
        helper.make_tensor_value_info("num_dets", TensorProto.INT32, [batch_dim, 1]),
        helper.make_tensor_value_info("det_boxes", TensorProto.FLOAT, [batch_dim, max_det, 4]),
        helper.make_tensor_value_info("det_scores", TensorProto.FLOAT, [batch_dim, max_det]),
        helper.make_tensor_value_info("det_classes", TensorProto.INT32, [batch_dim, max_det]),
    ])
    if not any(o.domain == "TRT" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("TRT", 1))
    for k, v in {"nms": "EfficientNMS_TRT", "nms_score_threshold": str(score_threshold),
                 "nms_iou_threshold": str(iou_threshold), "nms_max_det": str(max_det)}.items():
        model.metadata_props.add(key=k, value=v)
    onnx.save(model, dst)
    return dst


def prepare(cfg: Dict, out_dir: str = "build/gpu", n_bench: int = 64) -> Dict[str, str]:
    """Frame sets + NMS graphs for every variant the GPU run builds engines from."""
    os.makedirs(out_dir, exist_ok=True)
    n_eval = cfg["eval"]["n_scenarios"]
    paths = {
        "bench_frames": write_frames(
            os.path.join(out_dir, "bench.nsfr"),
            (generate_scenario(BENCH_SEED + i, seed=BENCH_SEED + i).frame for i in range(n_bench))),
        "eval_frames": write_frames(
            os.path.join(out_dir, "eval.nsfr"),
            (generate_scenario(EVAL_SEED + i, seed=EVAL_SEED + i).frame for i in range(n_eval))),
    }
    deploy, ev = cfg["deploy"], cfg["eval"]
    # fp32 + flags drives TensorRT <= 10 (weakly typed); the fp16 / fp16_mixed /
    # int8 graphs carry their own precision for strongly typed builds (TRT >= 11).
    for variant in ("fp32", "fp16", "fp16_mixed", "int8"):
        src = cfg["models"][variant]
        if not os.path.exists(src):
            logger.warning("%s missing — run `quantize` first; skipping its NMS graphs", src)
            continue
        paths[f"{variant}_nms"] = add_efficient_nms(
            src, os.path.join(out_dir, f"nurosim_det_{variant}_nms.onnx"),
            deploy["conf_threshold"], deploy["iou_threshold"], max_det=100)
        paths[f"{variant}_nms_eval"] = add_efficient_nms(
            src, os.path.join(out_dir, f"nurosim_det_{variant}_nms_eval.onnx"),
            ev["conf_threshold"], deploy["iou_threshold"], max_det=300)
    for k, v in paths.items():
        logger.info("  %-16s %s", k, v)
    return paths
