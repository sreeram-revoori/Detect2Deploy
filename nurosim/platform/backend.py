"""
backend.py
ServingModel: a batch of raw frames → padded detection arrays.

Output schema is identical to the TensorRT EfficientNMS engine used by the
Triton deployment, so clients, the shadow comparison and the batch job don't
care which runtime produced it:

    num_dets    int32   (B, 1)
    det_boxes   float32 (B, K, 4)   xyxy, frame pixels
    det_scores  float32 (B, K)
    det_classes int32   (B, K)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from nurosim.inference.postprocess import decode_single
from nurosim.inference.preprocess import LetterboxMeta
from nurosim.inference.runtime import OrtRuntime
from nurosim.platform.graph import FRAME_INPUT
from nurosim.scenario_generator import BBox, CLASS_CONFIG

MAX_DET = 100
OUTPUTS = ("num_dets", "det_boxes", "det_scores", "det_classes")
CLASS_NAMES = {v["id"]: k for k, v in CLASS_CONFIG.items()}


class ServingModel:
    def __init__(self, model_path: str, provider: str = "cpu", max_batch: int = 8,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.6,
                 max_det: int = MAX_DET, intra_op_threads: Optional[int] = None):
        opts = {}
        if provider == "tensorrt":
            # One engine covering every batch size the dynamic batcher can form
            s = "640x640x3"
            opts = {"trt_profile_min_shapes": f"{FRAME_INPUT}:1x{s}",
                    "trt_profile_opt_shapes": f"{FRAME_INPUT}:{max(1, max_batch // 2)}x{s}",
                    "trt_profile_max_shapes": f"{FRAME_INPUT}:{max_batch}x{s}"}
        self.runtime = OrtRuntime(model_path, provider=provider, provider_options=opts,
                                  intra_op_threads=intra_op_threads)
        if self.runtime.input_dtype != np.uint8:
            raise ValueError(f"{model_path} is not a serving graph (expects uint8 '{FRAME_INPUT}'); "
                             "build one with `python -m nurosim.platform export`")
        self.size = int(self.runtime.input_shape[1])
        self.meta = LetterboxMeta(ratio=1.0, pad_x=0, pad_y=0, frame_w=self.size, frame_h=self.size)
        self.conf, self.iou, self.max_det = conf_threshold, iou_threshold, max_det
        self.model_path, self.provider = model_path, provider

    def infer(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        """frames: uint8 (B, S, S, 3) BGR."""
        raw = self.runtime.run(np.ascontiguousarray(frames, dtype=np.uint8))
        return pack([decode_single(raw[i], self.meta, conf_thr=self.conf, iou_thr=self.iou,
                                   max_det=self.max_det) for i in range(len(frames))], self.max_det)


def pack(dets: List[List[BBox]], max_det: int = MAX_DET) -> Dict[str, np.ndarray]:
    b = len(dets)
    out = {"num_dets": np.zeros((b, 1), np.int32),
           "det_boxes": np.zeros((b, max_det, 4), np.float32),
           "det_scores": np.zeros((b, max_det), np.float32),
           "det_classes": np.zeros((b, max_det), np.int32)}
    for i, frame in enumerate(dets):
        frame = sorted(frame, key=lambda d: -d.confidence)[:max_det]
        out["num_dets"][i, 0] = len(frame)
        for k, d in enumerate(frame):
            out["det_boxes"][i, k] = d.xyxy
            out["det_scores"][i, k] = d.confidence
            out["det_classes"][i, k] = d.class_id
    return out


def unpack(out: Dict[str, np.ndarray]) -> List[List[BBox]]:
    """Padded arrays (from any backend) → per-frame BBox lists."""
    res = []
    for i in range(len(out["num_dets"])):
        n = int(np.asarray(out["num_dets"][i]).reshape(-1)[0])
        frame = []
        for k in range(n):
            x1, y1, x2, y2 = (int(round(float(v))) for v in out["det_boxes"][i][k])
            c = int(out["det_classes"][i][k])
            frame.append(BBox(x1, y1, x2, y2, c, CLASS_NAMES.get(c, str(c)), float(out["det_scores"][i][k])))
        res.append(frame)
    return res
