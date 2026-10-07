"""
perception_model.py
Lightweight perception model wrapper.

We ship two backends:
  1. MockDetector   – deterministic stub that adds realistic noise to GT boxes.
                      Zero dependencies beyond NumPy. Used for fast CI runs.
  2. ONNXDetector   – a real ONNX model (YOLOv8n trained on Detect2Deploy frames)
                      via detect2deploy.inference. Requires `onnxruntime`.

Both expose the same interface:
    detector.predict(frame: np.ndarray) -> List[BBox]
"""

from __future__ import annotations

import time
import numpy as np
from typing import Any, Dict, List, Protocol, runtime_checkable

from detect2deploy.scenario_generator import BBox, FRAME_W, FRAME_H, CLASS_CONFIG
from detect2deploy.inference.detector import InferenceDetector


# ── Protocol (interface) ──────────────────────────────────────────────────────

@runtime_checkable
class Detector(Protocol):
    def predict(self, frame: np.ndarray) -> List[BBox]: ...
    @property
    def name(self) -> str: ...


# ── Mock Detector (no heavy deps) ─────────────────────────────────────────────

class MockDetector:
    """
    Simulates a perception model by adding Gaussian noise to ground-truth
    boxes and randomly dropping / hallucinating detections.

    Params
    ------
    tp_rate     : probability a GT box is detected
    fp_rate     : expected number of false-positive detections per frame
    noise_px    : std-dev of bounding-box coordinate jitter (pixels)
    conf_mean   : mean detection confidence score
    seed        : RNG seed
    """

    def __init__(self,
                 tp_rate:    float = 0.82,
                 fp_rate:    float = 0.8,
                 noise_px:   float = 6.0,
                 conf_mean:  float = 0.71,
                 seed:       int   = 0):
        self._tp_rate   = tp_rate
        self._fp_rate   = fp_rate
        self._noise_px  = noise_px
        self._conf_mean = conf_mean
        self._rng       = np.random.default_rng(seed)
        self._name      = "MockDetector-v1"

    @property
    def name(self) -> str:
        return self._name

    def predict(self, frame: np.ndarray,
                ground_truth: List[BBox] | None = None) -> List[BBox]:
        """
        Return predicted bounding boxes.

        When ground_truth is provided (training/eval mode) the mock uses it
        to simulate realistic TP/FP/FN behaviour.  Without GT it returns an
        empty list (inference-only mode).
        """
        detections: List[BBox] = []

        if ground_truth is None:
            return detections

        class_ids   = list(range(len(CLASS_CONFIG)))
        class_names = list(CLASS_CONFIG.keys())

        # True positives with jitter
        for gt in ground_truth:
            if self._rng.random() > self._tp_rate:
                continue                                # missed detection (FN)

            # Scale noise relative to box size so small objects aren't destroyed
            box_w = max(gt.x2 - gt.x1, 1)
            box_h = max(gt.y2 - gt.y1, 1)
            adaptive_noise = min(self._noise_px,
                                 max(box_w, box_h) * 0.12)
            noise = self._rng.normal(0, adaptive_noise, 4).astype(int)
            x1 = int(np.clip(gt.x1 + noise[0], 0, FRAME_W - 1))
            y1 = int(np.clip(gt.y1 + noise[1], 0, FRAME_H - 1))
            x2 = int(np.clip(gt.x2 + noise[2], 0, FRAME_W - 1))
            y2 = int(np.clip(gt.y2 + noise[3], 0, FRAME_H - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            conf = float(np.clip(
                self._rng.normal(self._conf_mean, 0.08), 0.30, 0.99))

            detections.append(BBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                class_id=gt.class_id,
                class_name=gt.class_name,
                confidence=conf,
            ))

        # False positives
        n_fp = int(self._rng.poisson(self._fp_rate))
        for _ in range(n_fp):
            x1 = int(self._rng.integers(0, FRAME_W - 40))
            y1 = int(self._rng.integers(0, FRAME_H - 40))
            x2 = int(self._rng.integers(x1 + 10, min(x1 + 80, FRAME_W)))
            y2 = int(self._rng.integers(y1 + 10, min(y1 + 80, FRAME_H)))
            cls_idx = int(self._rng.integers(0, len(class_ids)))
            conf = float(self._rng.uniform(0.30, 0.55))
            detections.append(BBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                class_id=class_ids[cls_idx],
                class_name=class_names[cls_idx],
                confidence=conf,
            ))

        return detections


# ── ONNX Detector ─────────────────────────────────────────────────────────────

class ONNXDetector(InferenceDetector):
    """
    Back-compat name for the real ONNX inference path.

    The original version had three bugs: no NMS, boxes left in 640x640 model
    space instead of being mapped back through the resize, and `cls_id % 4`
    mapping COCO's 80 classes onto ours. It now delegates to
    detect2deploy.inference (letterbox → ORT → vectorised decode + class-aware NMS).

    Export a model with:  python -m detect2deploy.deploy export
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.45, **kwargs):
        super().__init__(model_path, conf_threshold=conf_threshold, **kwargs)


def build_detector(spec: Dict[str, Any] | None, seed: int = 0):
    """
    Construct a detector from a picklable spec, so worker processes / Ray
    actors can each build (and keep) their own instance.

        {"kind": "mock", "tp_rate": 0.82, ...}
        {"kind": "onnx", "model_path": "models/x.onnx", "provider": "cpu", ...}
    """
    spec = dict(spec or {"kind": "mock"})
    kind = spec.pop("kind", "mock")
    if kind == "mock":
        spec.setdefault("seed", seed)
        return MockDetector(**spec)
    if kind == "onnx":
        return InferenceDetector(**spec)
    raise ValueError(f"unknown detector kind '{kind}'")


# ── Latency-aware wrapper ─────────────────────────────────────────────────────

class TimedDetector:
    """Wraps any Detector and records per-call latency in milliseconds."""

    def __init__(self, detector: Detector):
        self._det     = detector
        self.latencies: List[float] = []

    @property
    def name(self) -> str:
        return self._det.name

    def predict(self, frame: np.ndarray,
                ground_truth=None) -> List[BBox]:
        t0  = time.perf_counter()
        out = self._det.predict(frame, ground_truth=ground_truth)
        self.latencies.append((time.perf_counter() - t0) * 1000)
        return out

    def latency_stats(self) -> dict:
        if not self.latencies:
            return {}
        arr = np.array(self.latencies)
        return {
            "mean_ms":   float(np.mean(arr)),
            "p50_ms":    float(np.percentile(arr, 50)),
            "p95_ms":    float(np.percentile(arr, 95)),
            "p99_ms":    float(np.percentile(arr, 99)),
            "min_ms":    float(np.min(arr)),
            "max_ms":    float(np.max(arr)),
        }
