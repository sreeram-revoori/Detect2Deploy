"""
tests/test_gpu_prep.py
Python side of the TensorRT path: frame files, EfficientNMS graph surgery,
C++ detection scoring and the GPU run summary. None of it needs a GPU.
"""

import json
import os

import numpy as np
import pytest

from detect2deploy.scenario_generator import generate_scenario


def test_frames_roundtrip(tmp_path):
    from detect2deploy.deploy.trt_prep import read_frames, write_frames
    frames = [generate_scenario(i, seed=i).frame for i in range(3)]
    p = write_frames(str(tmp_path / "f.nsfr"), frames)
    back = read_frames(p)
    assert back.shape == (3, 640, 640, 3)
    assert np.array_equal(back[2], frames[2])
    with open(p, "rb") as f:
        assert f.read(4) == b"NSFR"


def test_efficient_nms_graph(tiny_detector_onnx, tmp_path):
    onnx = pytest.importorskip("onnx")
    import onnxruntime as ort
    from onnx.utils import extract_model
    from detect2deploy.deploy.trt_prep import NMS_OUTPUTS, add_efficient_nms

    out = add_efficient_nms(tiny_detector_onnx, str(tmp_path / "nms.onnx"), 0.3, 0.5, max_det=50)
    m = onnx.load(out)
    onnx.checker.check_model(m)
    assert [o.name for o in m.graph.output] == list(NMS_OUTPUTS)
    nms = next(n for n in m.graph.node if n.op_type == "EfficientNMS_TRT")
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in nms.attribute}
    assert nms.domain == "TRT" and attrs["box_coding"] == 1 and attrs["max_output_boxes"] == 50
    assert attrs["score_threshold"] == pytest.approx(0.3) and attrs["class_agnostic"] == 0

    # The plugin's inputs must be exactly the head output, transposed and split
    sub = str(tmp_path / "pre_nms.onnx")
    extract_model(out, sub, ["images"], ["nms_boxes", "nms_scores", "output0"])
    x = np.random.default_rng(0).random((2, 3, 640, 640), dtype=np.float32)
    boxes, scores, raw = ort.InferenceSession(sub, providers=["CPUExecutionProvider"]).run(None, {"images": x})
    t = raw.transpose(0, 2, 1)
    assert np.array_equal(boxes, t[..., :4]) and np.array_equal(scores, t[..., 4:])


def _write_dets(path, frames):
    with open(path, "w") as f:
        json.dump({"engine": "x", "post": "efficient_nms",
                   "frames": [[[b.x1, b.y1, b.x2, b.y2, b.class_id, b.confidence] for b in fr]
                              for fr in frames]}, f)
    return path


@pytest.mark.skipif(not os.path.exists("models/d2d_det_fp32.onnx"), reason="needs the FP32 model")
def test_cpp_parity_scores_dumped_detections(tmp_path):
    from detect2deploy.deploy.config import load_config
    from detect2deploy.deploy.cpp_parity import run_cpp_parity
    from detect2deploy.deploy.seeds import EVAL_SEED
    from detect2deploy.inference.detector import InferenceDetector

    cfg = load_config()
    det = InferenceDetector(cfg["models"]["fp32"], conf_threshold=cfg["eval"]["conf_threshold"])
    preds = [det.predict(generate_scenario(EVAL_SEED + i, seed=EVAL_SEED + i).frame) for i in range(6)]
    same = _write_dets(str(tmp_path / "dets_same.json"), preds)
    empty = _write_dets(str(tmp_path / "dets_empty.json"), [[] for _ in preds])

    rep = run_cpp_parity(cfg, [same, empty])
    from detect2deploy.deploy.cpp_parity import write_cpp_parity
    with open(write_cpp_parity(rep, str(tmp_path / "out"))) as f:      # NumPy-safe JSON
        assert json.load(f)["targets"]["same"]["within_budget"] is True
    assert rep["n_frames"] == 6
    assert rep["targets"]["same"]["delta_vs_reference"]["map50_drop"] == pytest.approx(0.0)
    assert rep["targets"]["same"]["detection_agreement"]["ref_recall"] == 1.0
    assert rep["targets"]["same"]["within_budget"]
    assert not rep["targets"]["empty"]["within_budget"]


def test_gpu_summary_renders(tmp_path):
    from detect2deploy.deploy.gpu_report import build_summary
    summ = {"n": 10, "mean": 1, "std": 0.1, "p50": 1, "p90": 1.1, "p99": 1.2, "max": 1.3, "jitter_p99_p50": 0.2}
    dev = {"name": "NVIDIA L4", "compute_capability": "8.9", "sms": 58, "integrated": False,
           "cuda_driver": 12080, "cuda_runtime": 12080, "tensorrt": "10.8.0"}
    bench = {"label": "fp16 nms gpu-pre graph", "pre": "gpu", "post": "efficient_nms", "cuda_graph": True,
             "batch": 1, "device": dev, "e2e": summ, "throughput_fps": 1000,
             "stages": {"host_pre": summ, "post": summ, "gpu_total": summ}}
    (tmp_path / "bench_01_x.json").write_text(json.dumps(bench))
    (tmp_path / "sweep_bs8.json").write_text(json.dumps({**bench, "batch": 8}))
    phases = [{"name": "a_solo", "a": summ, "a_frames": 240, "a_deadline_misses": 0, "b_fps": 0},
              {"name": "b_solo", "a": summ, "a_frames": 0, "a_deadline_misses": 0, "b_fps": 900}]
    (tmp_path / "multistream.json").write_text(json.dumps(
        {"rate_a_hz": 30, "batch_b": 8, "duration_s": 8, "phases": phases}))
    (tmp_path / "run.log").write_text(
        "[t] ✓ build\n[t] ✗ nsys profile (exit 127)\n"
        "[t] ⚠ ORT gate (nvidia-trt profile): FAIL — a model missed its budget (see the gate report)\n")
    md = build_summary(str(tmp_path))
    assert "NVIDIA L4" in md and "fp16 nms gpu-pre graph" in md
    assert "| 8 |" in md and "a_solo" in md and "nsys profile" in md
    assert "## Gate verdicts" in md and "ORT gate (nvidia-trt profile): FAIL" in md
