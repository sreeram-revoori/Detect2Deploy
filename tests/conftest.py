"""
Shared fixtures. `tiny_detector_onnx` builds a minimal model with the same I/O
contract as the exported YOLOv8 head — images (batch, 3, 640, 640) →
output0 (batch, 8, 64) — so runtime / quantisation / gate code can be tested
in CI without the real weights or torch.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def tiny_detector_onnx(tmp_path_factory):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    init = {
        "w0": (rng.standard_normal((4, 3, 80, 80)) * 0.01).astype(np.float32),
        "w_box": (rng.standard_normal((4, 4, 1, 1)) * 5).astype(np.float32),
        "b_box": np.array([320, 320, 40, 40], dtype=np.float32),
        "w_cls": rng.standard_normal((4, 4, 1, 1)).astype(np.float32),
        "b_cls": np.array([-2, -1, -2, -3], dtype=np.float32),
        "shape": np.array([0, 4, -1], dtype=np.int64),
    }
    # Mirrors the YOLOv8 Detect head: separate box (cv2) and class (cv3) conv
    # branches, then a "tail" (reshape / sigmoid / concat) under /model.1/.
    nodes = [
        helper.make_node("Conv", ["images", "w0"], ["feat"], name="/model.0/conv/Conv",
                         kernel_shape=[80, 80], strides=[80, 80]),
        helper.make_node("Conv", ["feat", "w_box", "b_box"], ["box4"], name="/model.1/cv2.0/Conv"),
        helper.make_node("Conv", ["feat", "w_cls", "b_cls"], ["cls4"], name="/model.1/cv3.0/Conv"),
        helper.make_node("Reshape", ["box4", "shape"], ["box"], name="/model.1/Reshape"),
        helper.make_node("Reshape", ["cls4", "shape"], ["logit"], name="/model.1/Reshape_1"),
        helper.make_node("Sigmoid", ["logit"], ["score"], name="/model.1/Sigmoid"),
        helper.make_node("Concat", ["box", "score"], ["output0"], name="/model.1/Concat", axis=1),
    ]
    graph = helper.make_graph(
        nodes, "tiny_det",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", 8, 64])],
        initializer=[numpy_helper.from_array(v, k) for k, v in init.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    path = tmp_path_factory.mktemp("models") / "tiny_fp32.onnx"
    onnx.save(model, str(path))
    return str(path)
