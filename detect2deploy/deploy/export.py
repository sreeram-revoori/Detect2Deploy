"""
export.py
PyTorch checkpoint → ONNX (FP32 reference artefact).

Batch stays dynamic (for the batch-size sweep and offboard batch inference);
spatial dims are pinned to imgsz so ORT / TensorRT can specialise kernels.
"""

from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)


def sha256_file(path: str, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def export_onnx(weights: str,
                out_path: str = "models/d2d_det_fp32.onnx",
                imgsz: int = 640,
                opset: int = 17) -> str:
    import onnx
    from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
    from ultralytics import YOLO

    exported = YOLO(weights).export(format="onnx", imgsz=imgsz, opset=opset,
                                    dynamic=True, simplify=True, device="cpu")

    model = onnx.load(exported)
    n_anchors = sum((imgsz // s) ** 2 for s in (8, 16, 32))
    for name, value in {"height": imgsz, "width": imgsz, "anchors": n_anchors}.items():
        make_dim_param_fixed(model.graph, name, value)
    # With H/W static the anchor-grid subgraph (Shape/Range/Expand…) becomes
    # constant — re-slim so it folds away before quantisation / TensorRT.
    import onnxslim
    model = onnxslim.slim(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    meta = {p.key: p.value for p in model.metadata_props}
    meta["source_checkpoint_sha256"] = sha256_file(weights)
    # Key keeps the project's original name: it lives inside the model bytes, and
    # renaming it would change every model hash the gate evidence is bound to.
    meta["nurosim_variant"] = "fp32"
    del model.metadata_props[:]
    for k, v in meta.items():
        model.metadata_props.add(key=k, value=str(v))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    onnx.save(model, out_path)
    if os.path.abspath(exported) != os.path.abspath(out_path):
        os.remove(exported)

    inp = model.graph.input[0]
    dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
    logger.info("Exported %s  input=%s%s  (%.1f MB, sha %s)", out_path, inp.name,
                dims, os.path.getsize(out_path) / 1e6, sha256_file(out_path))
    return out_path
