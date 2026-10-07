"""
tests/test_inference.py
Inference path: preprocessing, decode/NMS, ORT runtime, quantisation,
parity helpers and the release gate.
"""

import numpy as np
import pytest

from nurosim.inference.preprocess import letterbox, preprocess_batch
from nurosim.inference.postprocess import decode_single, nms
from nurosim.metrics import compute_iou, iou_matrix
from nurosim.scenario_generator import BBox, generate_scenario


def _raw(boxes_scores, nc=4):
    """[(cx, cy, w, h, cls, score), ...] → (4 + nc, A) head output."""
    out = np.zeros((4 + nc, len(boxes_scores)), dtype=np.float32)
    for a, (cx, cy, w, h, c, s) in enumerate(boxes_scores):
        out[:4, a] = (cx, cy, w, h)
        out[4 + c, a] = s
    return out


# ── Preprocess ────────────────────────────────────────────────────────────────

class TestPreprocess:

    def test_square_frame_is_identity_geometry(self):
        frame = np.zeros((640, 640, 3), np.uint8)
        img, meta = letterbox(frame, 640)
        assert img.shape == (640, 640, 3)
        assert (meta.ratio, meta.pad_x, meta.pad_y) == (1.0, 0, 0)

    def test_non_square_frame_is_padded(self):
        frame = np.zeros((480, 960, 3), np.uint8)          # 2:1 → ratio 2/3
        img, meta = letterbox(frame, 640)
        assert img.shape == (640, 640, 3)
        assert meta.ratio == pytest.approx(640 / 960)
        assert meta.pad_x == 0 and meta.pad_y == 160

    def test_batch_tensor_layout(self):
        frames = [np.full((640, 640, 3), (255, 0, 0), np.uint8)] * 3   # pure blue (BGR)
        batch, metas = preprocess_batch(frames)
        assert batch.shape == (3, 3, 640, 640) and batch.dtype == np.float32
        assert len(metas) == 3
        assert batch[0, 2].max() == pytest.approx(1.0)   # blue lands in RGB channel 2
        assert batch[0, 0].max() == 0.0


# ── Postprocess ───────────────────────────────────────────────────────────────

class TestPostprocess:

    def test_boxes_mapped_back_through_letterbox(self):
        frame = np.zeros((480, 960, 3), np.uint8)
        _, meta = letterbox(frame, 640)
        # model-space box centred at (320, 160 + 100), 60x30
        dets = decode_single(_raw([(320, 260, 60, 30, 0, 0.9)]), meta, conf_thr=0.25)
        assert len(dets) == 1
        d = dets[0]
        # undo pad (0, 160) then scale by 1/ratio = 1.5
        assert (d.x1, d.y1, d.x2, d.y2) == (435, 128, 525, 172)
        assert d.class_name == "vehicle" and d.confidence == pytest.approx(0.9)

    def test_confidence_threshold(self):
        _, meta = letterbox(np.zeros((640, 640, 3), np.uint8))
        raw = _raw([(100, 100, 20, 20, 1, 0.9), (300, 300, 20, 20, 2, 0.1)])
        dets = decode_single(raw, meta, conf_thr=0.25)
        assert [d.class_name for d in dets] == ["pedestrian"]

    def test_nms_is_class_aware(self):
        _, meta = letterbox(np.zeros((640, 640, 3), np.uint8))
        raw = _raw([(100, 100, 40, 40, 0, 0.9),     # vehicle
                    (102, 101, 40, 40, 0, 0.8),     # duplicate vehicle → suppressed
                    (101, 100, 40, 40, 3, 0.7)])    # cone at same spot → kept
        dets = decode_single(raw, meta, conf_thr=0.25, iou_thr=0.6)
        assert sorted((d.class_name, round(d.confidence, 1)) for d in dets) == \
            [("cone", 0.7), ("vehicle", 0.9)]

    def test_nms_empty(self):
        assert nms(np.zeros((0, 4)), np.zeros(0), 0.5).size == 0

    def test_iou_matrix_matches_scalar_iou(self):
        rng = np.random.default_rng(0)

        def rand_box():
            x1, y1 = rng.integers(0, 600, 2)
            w, h = rng.integers(1, 60, 2)
            return BBox(int(x1), int(y1), int(x1 + w), int(y1 + h), 0, "vehicle")

        a = [rand_box() for _ in range(7)]
        b = [rand_box() for _ in range(5)] + a[:2]
        mat = iou_matrix(a, b)
        for i, p in enumerate(a):
            for j, g in enumerate(b):
                assert mat[i, j] == pytest.approx(compute_iou(p, g))


# ── Generator fixes the inference work depends on ────────────────────────────

class TestGeneratorDeterminism:

    def test_rain_frames_are_deterministic(self):
        a = generate_scenario(3, seed=11, weather="rain")
        b = generate_scenario(3, seed=11, weather="rain")
        np.testing.assert_array_equal(a.frame, b.frame)

    def test_no_degenerate_gt_boxes(self):
        for s in range(50):
            for g in generate_scenario(s, seed=s).ground_truth:
                assert g.x2 - g.x1 >= 4 and g.y2 - g.y1 >= 4


# ── Runtime / detector (tiny ONNX model) ─────────────────────────────────────

class TestRuntime:

    def test_dynamic_batch(self, tiny_detector_onnx):
        from nurosim.inference.runtime import OrtRuntime
        rt = OrtRuntime(tiny_detector_onnx)
        for bs in (1, 3):
            out = rt.run(np.zeros((bs, 3, 640, 640), np.float32))
            assert out.shape == (bs, 8, 64)
        assert rt.provider_active

    def test_static_batch_pins_shape(self, tiny_detector_onnx):
        from nurosim.inference.runtime import OrtRuntime
        rt = OrtRuntime(tiny_detector_onnx, static_batch=1)
        assert rt.input_shape[0] == 1

    def test_unavailable_provider_raises(self, tiny_detector_onnx):
        import onnxruntime as ort
        from nurosim.inference.runtime import OrtRuntime, ProviderUnavailableError
        if "TensorrtExecutionProvider" in ort.get_available_providers():
            pytest.skip("TensorRT is available on this host")
        with pytest.raises(ProviderUnavailableError):
            OrtRuntime(tiny_detector_onnx, provider="tensorrt")

    def test_detector_interface_and_timing(self, tiny_detector_onnx):
        from nurosim.inference.detector import InferenceDetector
        det = InferenceDetector(tiny_detector_onnx, conf_threshold=0.01)
        sc = generate_scenario(0, seed=0)
        boxes, t = det.predict_timed(sc.frame)
        assert isinstance(det.predict(sc.frame, ground_truth=sc.ground_truth), list)
        assert all(isinstance(b, BBox) for b in boxes)
        assert t.pre_ms > 0 and t.infer_ms > 0 and t.total_ms >= t.infer_ms

    def test_parallel_evaluator_with_onnx_spec(self, tiny_detector_onnx):
        from nurosim.ray_worker import ParallelEvaluator
        spec = {"kind": "onnx", "model_path": tiny_detector_onnx, "intra_op_threads": 1}
        pe = ParallelEvaluator(n_scenarios=8, n_workers=2, use_ray=False,
                               base_seed=0, detector_spec=spec)
        result, stats = pe.run()
        assert result.n_scenarios == 8 and stats["n_scenarios"] == 8


# ── Quantisation ──────────────────────────────────────────────────────────────

class TestQuantize:

    def test_fp16_keeps_fp32_io_and_matches(self, tiny_detector_onnx, tmp_path):
        from nurosim.deploy.quantize import to_fp16
        from nurosim.inference.runtime import OrtRuntime
        out = to_fp16(tiny_detector_onnx, str(tmp_path / "fp16.onnx"))
        x, _ = preprocess_batch([generate_scenario(0, seed=0).frame])
        rt16 = OrtRuntime(out)
        assert rt16.input_dtype == np.float32
        np.testing.assert_allclose(rt16.run(x), OrtRuntime(tiny_detector_onnx).run(x),
                                   rtol=1e-2, atol=0.5)

    def test_fp16_mixed_keeps_tail_fp32(self, tiny_detector_onnx, tmp_path):
        import onnx
        from nurosim.deploy.quantize import to_fp16
        from nurosim.inference.runtime import OrtRuntime
        out = to_fp16(tiny_detector_onnx, str(tmp_path / "fp16m.onnx"), keep_head_tail_fp32=True)
        model = onnx.load(out)
        names = [n.name for n in model.graph.node]
        assert len(names) == len(set(names))                 # converter dup-Cast fixed
        x, _ = preprocess_batch([generate_scenario(0, seed=0).frame])
        assert OrtRuntime(out).run(x).shape == (1, 8, 64)

    def test_int8_qdq_close_to_fp32(self, tiny_detector_onnx, tmp_path):
        import onnx
        from nurosim.deploy.quantize import to_int8
        from nurosim.inference.runtime import OrtRuntime
        out = to_int8(tiny_detector_onnx, str(tmp_path / "int8.onnx"), n_calib=8)
        ops = {n.op_type for n in onnx.load(out).graph.node}
        assert {"QuantizeLinear", "DequantizeLinear"} <= ops
        x, _ = preprocess_batch([generate_scenario(1, seed=1).frame])
        ref, q = OrtRuntime(tiny_detector_onnx).run(x), OrtRuntime(out).run(x)
        assert np.abs(ref[:, 4:] - q[:, 4:]).max() < 0.05       # scores (FP32 tail)

    def test_int8_for_tensorrt_has_no_int32_bias(self, tiny_detector_onnx, tmp_path):
        import onnx
        from nurosim.deploy.quantize import to_int8
        out = to_int8(tiny_detector_onnx, str(tmp_path / "int8_trt.onnx"), n_calib=8, quantize_bias=False)
        m = onnx.load(out)
        inits = {i.name: i.data_type for i in m.graph.initializer}
        dq_types = {inits[n.input[0]] for n in m.graph.node
                    if n.op_type == "DequantizeLinear" and n.input[0] in inits}
        assert onnx.TensorProto.INT32 not in dq_types and onnx.TensorProto.INT8 in dq_types

    def test_full_int8_collapses_scores(self, tiny_detector_onnx, tmp_path):
        """Regression test for the finding behind keep_head_tail_fp32: one INT8
        scale on the [boxes(0-640) | scores(0-1)] concat rounds scores to 0."""
        from nurosim.deploy.quantize import to_int8
        from nurosim.inference.runtime import OrtRuntime
        out = to_int8(tiny_detector_onnx, str(tmp_path / "int8_full.onnx"),
                      n_calib=8, keep_head_tail_fp32=False)
        x, _ = preprocess_batch([generate_scenario(1, seed=1).frame])
        assert OrtRuntime(out).run(x)[:, 4:].max() == 0.0

    def test_calibration_reader_is_weather_stratified_and_chunkable(self):
        from nurosim.deploy.quantize import ScenarioCalibrationReader
        r = ScenarioCalibrationReader("images", n_frames=8)
        assert len(r) == 8
        r.set_range(4, 8)
        n = 0
        while r.get_next() is not None:
            n += 1
        assert n == 4


# ── Parity helpers ────────────────────────────────────────────────────────────

class TestParity:

    def test_identical_outputs_agree(self):
        from nurosim.deploy.parity import detection_agreement, tensor_parity
        dets = [[BBox(10, 10, 50, 40, 0, "vehicle", 0.9)], []]
        a = detection_agreement(dets, dets)
        assert a["ref_recall"] == 1.0 and a["mean_abs_conf_delta"] == 0.0
        raw = np.random.default_rng(0).random((2, 8, 64)).astype(np.float32)
        t = tensor_parity(raw, raw, conf=0.5)
        assert t["box_mae_px"] == 0.0 and t["class_agreement"] == 1.0

    def test_class_flip_is_not_a_match(self):
        from nurosim.deploy.parity import detection_agreement
        ref = [[BBox(10, 10, 50, 40, 0, "vehicle", 0.9)]]
        tgt = [[BBox(10, 10, 50, 40, 2, "cyclist", 0.9)]]
        assert detection_agreement(ref, tgt)["ref_recall"] == 0.0


# ── Gate ──────────────────────────────────────────────────────────────────────

class TestGate:

    CFG = {"accuracy_budget": {"max_map50_drop": 0.01,
                               "max_map50_drop_per_weather": 0.02,
                               "max_ap50_drop_per_class": 0.03}}

    def _setup(self, model_path, map50_drop=0.0, provider_active=True, p99=10.0, sha=None):
        from nurosim.deploy.config import Profile, Target
        from nurosim.deploy.export import sha256_file
        sha = sha or sha256_file(model_path)
        profile = Profile("t", "", "ref", [
            Target("ref", "fp32", model_path, p99_budget_ms=50),
            Target("q", "int8", model_path, p99_budget_ms=20),
        ])
        delta = {"map50_drop": map50_drop,
                 "per_class_ap50_drop": {"vehicle": map50_drop},
                 "weather_map50_drop": {"fog": map50_drop}}
        entry = lambda: {"provider_active": provider_active, "model_sha256": sha,
                         "active_providers": ["CPUExecutionProvider"]}
        parity = {"targets": {"ref": entry(), "q": {**entry(), "delta_vs_reference": delta}}}
        bench = {"targets": {n: {**entry(), "e2e_batch1": {"total": {"p99": p99}}}
                             for n in ("ref", "q")}}
        return profile, parity, bench

    def _run(self, *args, **kw):
        from nurosim.deploy.gate import run_gate
        profile, parity, bench = self._setup(*args, **kw)
        return run_gate(self.CFG, profile, parity, bench)

    def test_pass(self, tiny_detector_onnx):
        assert all(c.passed for c in self._run(tiny_detector_onnx))

    def test_accuracy_regression_fails(self, tiny_detector_onnx):
        failed = {c.check for c in self._run(tiny_detector_onnx, map50_drop=0.05) if not c.passed}
        assert failed == {"mAP@0.5 drop", "worst weather drop", "worst class drop"}

    def test_latency_regression_fails(self, tiny_detector_onnx):
        failed = [(c.target, c.check) for c in self._run(tiny_detector_onnx, p99=30.0) if not c.passed]
        assert failed == [("q", "e2e p99 latency")]

    def test_silent_cpu_fallback_fails(self, tiny_detector_onnx):
        failed = {c.check for c in self._run(tiny_detector_onnx, provider_active=False) if not c.passed}
        assert "provider" in failed

    def test_stale_report_fails(self, tiny_detector_onnx):
        failed = {c.check for c in self._run(tiny_detector_onnx, sha="deadbeef0000") if not c.passed}
        assert failed == {"freshness"}

    def test_negative_control(self, tiny_detector_onnx):
        from nurosim.deploy.gate import run_gate
        for drop, should_pass in ((0.5, True), (0.0, False)):
            profile, parity, bench = self._setup(tiny_detector_onnx, map50_drop=drop)
            profile.targets[1].expect_fail = True
            checks = {c.check: c.passed for c in run_gate(self.CFG, profile, parity, bench)
                      if c.target == "q"}
            assert checks["negative control caught"] is should_pass
            assert "mAP@0.5 drop" not in checks and "e2e p99 latency" not in checks

    def test_missing_reports_fail(self, tiny_detector_onnx):
        from nurosim.deploy.gate import run_gate
        profile, _, _ = self._setup(tiny_detector_onnx)
        assert not any(c.passed for c in run_gate(self.CFG, profile, None, None))
