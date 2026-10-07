"""
preprocess.py
Letterbox preprocessing shared by the runtime detector AND the INT8
calibration reader. Keeping one implementation matters: if calibration sees
differently-normalised tensors than deployment, the INT8 scales are wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

PAD_VALUE = 114          # YOLO convention for letterbox padding


@dataclass(frozen=True)
class LetterboxMeta:
    """Everything needed to map boxes from model space back to the frame."""
    ratio: float          # model_px / frame_px
    pad_x: float          # left padding in model space
    pad_y: float          # top padding in model space
    frame_w: int
    frame_h: int


def letterbox(frame: np.ndarray, size: int = 640) -> Tuple[np.ndarray, LetterboxMeta]:
    """Resize keeping aspect ratio, pad to size x size. Returns (img, meta)."""
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = round(w * r), round(h * r)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2

    img = frame
    if (new_w, new_h) != (w, h):
        img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return img, LetterboxMeta(ratio=r, pad_x=left, pad_y=top, frame_w=w, frame_h=h)


def to_tensor(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 → float32 3xHxW in [0, 1], RGB."""
    rgb = img_bgr[:, :, ::-1]
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) * (1.0 / 255.0)
    return np.ascontiguousarray(chw)


def preprocess_batch(frames: Sequence[np.ndarray],
                     size: int = 640) -> Tuple[np.ndarray, List[LetterboxMeta]]:
    """Frames → NCHW float32 batch + per-frame letterbox metadata."""
    tensors, metas = [], []
    for f in frames:
        img, meta = letterbox(f, size)
        tensors.append(to_tensor(img))
        metas.append(meta)
    return np.stack(tensors, axis=0), metas
