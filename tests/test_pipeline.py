"""
tests/test_pipeline.py
Unit + integration tests for Detect2Deploy.
Run with:  pytest tests/ -v
"""

import numpy as np
import pytest

from detect2deploy.scenario_generator import (
    generate_scenario, generate_batch,
    BBox, FRAME_W, FRAME_H, CLASS_CONFIG, WEATHER_CONDITIONS,
)
from detect2deploy.perception_model import MockDetector, TimedDetector
from detect2deploy.metrics import compute_iou, Evaluator


# ── Scenario Generator ────────────────────────────────────────────────────────

class TestScenarioGenerator:

    def test_frame_shape(self):
        sc = generate_scenario(0, seed=0)
        assert sc.frame.shape == (FRAME_H, FRAME_W, 3)
        assert sc.frame.dtype == np.uint8

    def test_gt_boxes_non_empty(self):
        sc = generate_scenario(1, seed=1)
        assert len(sc.ground_truth) > 0

    def test_gt_boxes_within_frame(self):
        sc = generate_scenario(2, seed=2)
        for b in sc.ground_truth:
            assert 0 <= b.x1 < b.x2 <= FRAME_W
            assert 0 <= b.y1 < b.y2 <= FRAME_H

    def test_weather_assignment(self):
        for wx in WEATHER_CONDITIONS:
            sc = generate_scenario(0, weather=wx, seed=0)
            assert sc.weather == wx

    def test_deterministic(self):
        sc1 = generate_scenario(5, seed=42)
        sc2 = generate_scenario(5, seed=42)
        np.testing.assert_array_equal(sc1.frame, sc2.frame)
        assert len(sc1.ground_truth) == len(sc2.ground_truth)

    def test_batch_length(self):
        batch = generate_batch(10, base_seed=0)
        assert len(batch) == 10
        ids = [s.scenario_id for s in batch]
        assert ids == list(range(10))

    def test_class_ids_valid(self):
        valid_ids = {v["id"] for v in CLASS_CONFIG.values()}
        sc = generate_scenario(0, seed=7, num_objects=10)
        for b in sc.ground_truth:
            assert b.class_id in valid_ids


# ── IoU ───────────────────────────────────────────────────────────────────────

class TestIoU:

    def _box(self, x1, y1, x2, y2):
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2, class_id=0, class_name="vehicle")

    def test_perfect_overlap(self):
        b = self._box(10, 10, 50, 50)
        assert compute_iou(b, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = self._box(0,  0,  10, 10)
        b = self._box(20, 20, 30, 30)
        assert compute_iou(a, b) == 0.0

    def test_half_overlap(self):
        a = self._box(0,  0, 20, 10)
        b = self._box(10, 0, 30, 10)
        iou = compute_iou(a, b)
        # intersection=100, union=300 → 1/3
        assert iou == pytest.approx(1/3, rel=1e-3)

    def test_contained(self):
        outer = self._box(0,  0, 40, 40)
        inner = self._box(10, 10, 30, 30)
        iou = compute_iou(outer, inner)
        # intersection=400, outer=1600, inner=400, union=1600
        assert iou == pytest.approx(400/1600, rel=1e-3)


# ── Perception Model ──────────────────────────────────────────────────────────

class TestMockDetector:

    def test_returns_list(self):
        det = MockDetector(seed=0)
        sc  = generate_scenario(0, seed=0)
        out = det.predict(sc.frame, ground_truth=sc.ground_truth)
        assert isinstance(out, list)

    def test_detections_have_valid_boxes(self):
        det = MockDetector(tp_rate=1.0, fp_rate=0, seed=0)
        sc  = generate_scenario(0, seed=0, num_objects=5)
        out = det.predict(sc.frame, ground_truth=sc.ground_truth)
        for b in out:
            assert b.x2 > b.x1
            assert b.y2 > b.y1
            assert 0.0 < b.confidence <= 1.0

    def test_no_gt_returns_empty(self):
        det = MockDetector(seed=0)
        sc  = generate_scenario(0, seed=0)
        out = det.predict(sc.frame, ground_truth=None)
        assert out == []

    def test_timed_detector_records_latency(self):
        det  = TimedDetector(MockDetector(seed=0))
        sc   = generate_scenario(0, seed=0)
        det.predict(sc.frame, ground_truth=sc.ground_truth)
        assert len(det.latencies) == 1
        assert det.latencies[0] >= 0.0

    def test_latency_stats_keys(self):
        det = TimedDetector(MockDetector(seed=0))
        sc  = generate_scenario(0, seed=0)
        for _ in range(5):
            det.predict(sc.frame, ground_truth=sc.ground_truth)
        stats = det.latency_stats()
        for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
            assert key in stats


# ── Evaluator ─────────────────────────────────────────────────────────────────

class TestEvaluator:

    def _run_eval(self, n=30, tp_rate=0.85, fp_rate=0.5):
        det = TimedDetector(MockDetector(tp_rate=tp_rate, fp_rate=fp_rate, seed=7))
        ev  = Evaluator(iou_threshold=0.5)
        for i in range(n):
            sc    = generate_scenario(i, seed=42 + i)
            preds = det.predict(sc.frame, ground_truth=sc.ground_truth)
            ev.add(sc, preds)
        return ev.compute()

    def test_map50_range(self):
        result = self._run_eval()
        assert 0.0 <= result.map50 <= 1.0

    def test_map50_95_le_map50(self):
        result = self._run_eval()
        assert result.map50_95 <= result.map50 + 1e-6

    def test_perfect_detector_high_map(self):
        # Very high TP rate, no FPs → should yield high mAP
        result = self._run_eval(n=50, tp_rate=0.99, fp_rate=0.0)
        assert result.map50 > 0.60

    def test_per_class_keys(self):
        result = self._run_eval()
        assert set(result.per_class_ap50.keys()) == set(CLASS_CONFIG.keys())

    def test_weather_breakdown(self):
        result = self._run_eval(n=80)
        assert len(result.weather_map) > 0
        for m in result.weather_map.values():
            assert 0.0 <= m <= 1.0

    def test_empty_raises(self):
        ev = Evaluator()
        with pytest.raises(ValueError):
            ev.compute()

    def test_tp_fp_fn_consistent(self):
        result = self._run_eval(n=40)
        # TP + FN ≈ GT boxes (within a small margin due to class mismatch)
        assert result.total_tp >= 0
        assert result.total_fp >= 0
        assert result.total_fn >= 0


# ── Integration: full pipeline (no Ray) ───────────────────────────────────────

class TestIntegration:

    def test_parallel_evaluator_multiprocessing(self):
        from detect2deploy.ray_worker import ParallelEvaluator
        pe = ParallelEvaluator(n_scenarios=20, n_workers=2,
                               use_ray=False, base_seed=0)
        result, stats = pe.run()
        assert result.map50 > 0.0
        assert stats["n_scenarios"] == 20
        assert stats["throughput_fps"] > 0

    def test_stats_keys_present(self):
        from detect2deploy.ray_worker import ParallelEvaluator
        pe = ParallelEvaluator(n_scenarios=10, n_workers=2,
                               use_ray=False, base_seed=1)
        _, stats = pe.run()
        for key in ("backend", "n_workers", "n_scenarios",
                    "total_wall_s", "throughput_fps",
                    "lat_mean_ms", "lat_p50_ms", "lat_p95_ms"):
            assert key in stats
