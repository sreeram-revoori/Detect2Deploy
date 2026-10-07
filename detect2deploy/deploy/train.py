"""
train.py
Trains a YOLOv8n detector from scratch on the synthetic Detect2Deploy dataset.
From-scratch (yolov8n.yaml) rather than COCO weights: the BEV frames look
nothing like COCO, and it keeps the pipeline free of external weight downloads.
"""

from __future__ import annotations

import logging
import os
import shutil

logger = logging.getLogger(__name__)


def _default_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_detector(data_yaml: str,
                   out_path: str = "models/d2d_det.pt",
                   epochs: int = 40,
                   imgsz: int = 640,
                   batch: int = 16,
                   device: str | None = None,
                   project: str = "runs") -> str:
    from ultralytics import YOLO

    model = YOLO("yolov8n.yaml")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device or _default_device(),
        project=os.path.abspath(project),
        name="d2d_det",
        exist_ok=True,
        pretrained=False,
        plots=False,
        patience=15,
        seed=0,
        deterministic=True,
        # Colour encodes class in these frames, so no hue jitter; the scene is
        # axis-aligned top-down, so no rotation but vertical flips are valid.
        hsv_h=0.0,
        degrees=0.0,
        flipud=0.5,
        scale=0.3,
    )
    best = os.path.join(str(model.trainer.save_dir), "weights", "best.pt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    shutil.copy(best, out_path)
    logger.info("Best checkpoint copied to %s", out_path)
    return out_path
