"""
detector.py
InferenceDetector = preprocess → OrtRuntime → decode/NMS, with per-stage
timing. Satisfies the same `predict(frame, ground_truth=None)` interface as
MockDetector, so it drops into the Evaluator and ParallelEvaluator unchanged.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nurosim.inference.postprocess import decode_batch
from nurosim.inference.preprocess import preprocess_batch
from nurosim.inference.runtime import OrtRuntime
from nurosim.scenario_generator import BBox


@dataclass
class StageTiming:
    pre_ms:   float
    infer_ms: float
    post_ms:  float

    @property
    def total_ms(self) -> float:
        return self.pre_ms + self.infer_ms + self.post_ms


class InferenceDetector:
    def __init__(self,
                 model_path: str,
                 provider: str = "cpu",
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.6,
                 imgsz: int = 640,
                 provider_options: Optional[Dict[str, Any]] = None,
                 static_batch: Optional[int] = None,
                 intra_op_threads: Optional[int] = None,
                 class_names: Optional[Dict[int, str]] = None,
                 graph_opt: str = "all"):
        self.runtime = OrtRuntime(model_path, provider=provider,
                                  provider_options=provider_options,
                                  static_batch=static_batch,
                                  intra_op_threads=intra_op_threads,
                                  graph_opt=graph_opt)
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.imgsz          = imgsz
        self.class_names    = class_names
        self._name = f"{os.path.basename(model_path)}@{provider}"

    @property
    def name(self) -> str:
        return self._name

    def _decode_kwargs(self) -> dict:
        return dict(conf_thr=self.conf_threshold, iou_thr=self.iou_threshold,
                    class_names=self.class_names)

    def raw(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Raw head output for a batch — used for tensor-level parity checks."""
        batch, _ = preprocess_batch(frames, self.imgsz)
        return self.runtime.run(batch)

    def predict_batch_timed(self, frames: Sequence[np.ndarray]
                            ) -> Tuple[List[List[BBox]], StageTiming]:
        t0 = time.perf_counter()
        batch, metas = preprocess_batch(frames, self.imgsz)
        t1 = time.perf_counter()
        out = self.runtime.run(batch)
        t2 = time.perf_counter()
        dets = decode_batch(out, metas, **self._decode_kwargs())
        t3 = time.perf_counter()
        return dets, StageTiming((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)

    def predict_timed(self, frame: np.ndarray) -> Tuple[List[BBox], StageTiming]:
        dets, timing = self.predict_batch_timed([frame])
        return dets[0], timing

    def predict(self, frame: np.ndarray, ground_truth=None) -> List[BBox]:
        """ground_truth is accepted for interface compatibility and ignored."""
        return self.predict_timed(frame)[0]
