"""
postprocess.py
Vectorised YOLOv8 output decoding + class-aware NMS.

YOLOv8 ONNX head output: (B, 4 + num_classes, num_anchors), boxes as
(cx, cy, w, h) in letterboxed model pixels, class scores already sigmoided.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from nurosim.inference.preprocess import LetterboxMeta
from nurosim.scenario_generator import BBox, CLASS_CONFIG

DEFAULT_CLASS_NAMES: Dict[int, str] = {v["id"]: k for k, v in CLASS_CONFIG.items()}
_MAX_WH = 4096.0         # class offset for batched NMS (> any box coordinate)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float,
        max_det: int = 300) -> np.ndarray:
    """Greedy NMS. boxes: (N, 4) xyxy, scores: (N,). Returns kept indices."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores, kind="stable")
    keep: List[int] = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def decode_single(pred: np.ndarray,
                  meta: LetterboxMeta,
                  conf_thr: float = 0.25,
                  iou_thr: float = 0.6,
                  max_det: int = 300,
                  class_names: Dict[int, str] | None = None) -> List[BBox]:
    """Decode one image's raw head output (4 + nc, A) into frame-space BBoxes."""
    class_names = class_names or DEFAULT_CLASS_NAMES
    pred = pred.astype(np.float32, copy=False)
    cls_scores = pred[4:]                                   # (nc, A)
    cls_ids = cls_scores.argmax(axis=0)                     # (A,)
    conf = cls_scores[cls_ids, np.arange(cls_scores.shape[1])]

    mask = conf >= conf_thr
    if not mask.any():
        return []
    cx, cy, w, h = pred[:4, mask]
    conf, cls_ids = conf[mask], cls_ids[mask]

    # cxcywh (model px) → xyxy (frame px): undo letterbox padding and scale
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta.pad_x) / meta.ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta.pad_y) / meta.ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, meta.frame_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, meta.frame_h)

    # Class-aware NMS in one pass: shift each class into its own coordinate band
    keep = nms(boxes + cls_ids[:, None] * _MAX_WH, conf, iou_thr, max_det)

    out: List[BBox] = []
    for k in keep:
        x1, y1, x2, y2 = np.rint(boxes[k]).astype(int)
        if x2 <= x1 or y2 <= y1:
            continue
        cid = int(cls_ids[k])
        out.append(BBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                        class_id=cid, class_name=class_names.get(cid, str(cid)),
                        confidence=float(conf[k])))
    return out


def decode_batch(output: np.ndarray,
                 metas: Sequence[LetterboxMeta],
                 **kwargs) -> List[List[BBox]]:
    """Decode a (B, 4 + nc, A) head output into one BBox list per image."""
    return [decode_single(output[i], metas[i], **kwargs) for i in range(len(metas))]
