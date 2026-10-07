"""
graph.py
Fold preprocessing into the model so a serving endpoint takes raw camera
frames: uint8 NHWC BGR (B, S, S, 3) in, detections out.

    Cast(uint8→float) → Transpose(NHWC→NCHW) → Gather(BGR→RGB) → Mul(1/255) → model

Clients then send 1.2 MB of uint8 per frame instead of 4.9 MB of float32,
and there's no preprocessing service to deploy or keep in sync with the
model. TensorRT only accepts a uint8 network input that feeds a Cast, hence
the order. Assumes frames already at model resolution (true for the 640x640
BEV frames); other sizes still need the letterbox resize first.
"""

from __future__ import annotations

import numpy as np

FRAME_INPUT = "frames"


def add_uint8_frontend(src: str, dst: str, input_name: str = FRAME_INPUT) -> str:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(src)
    g = model.graph
    old = g.input[0]
    dims = old.type.tensor_type.shape.dim            # [batch, 3, S, S]
    batch = dims[0].dim_param or dims[0].dim_value
    h, w = dims[2].dim_value, dims[3].dim_value

    g.initializer.extend([
        numpy_helper.from_array(np.array(1.0 / 255.0, dtype=np.float32), "frontend/inv255"),
        numpy_helper.from_array(np.array([2, 1, 0], dtype=np.int64), "frontend/bgr2rgb"),
    ])
    frontend = [
        helper.make_node("Cast", [input_name], ["frontend/f32"], name="frontend/Cast", to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["frontend/f32"], ["frontend/nchw"], name="frontend/Transpose",
                         perm=[0, 3, 1, 2]),
        helper.make_node("Gather", ["frontend/nchw", "frontend/bgr2rgb"], ["frontend/rgb"],
                         name="frontend/Gather", axis=1),
        helper.make_node("Mul", ["frontend/rgb", "frontend/inv255"], [old.name], name="frontend/Mul"),
    ]
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(frontend + nodes)
    g.input.remove(old)
    g.input.insert(0, helper.make_tensor_value_info(input_name, TensorProto.UINT8, [batch, h, w, 3]))
    model.metadata_props.add(key="serving_input", value=f"{input_name}: uint8 NHWC BGR")
    onnx.save(model, dst)
    return dst
