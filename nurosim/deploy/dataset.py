"""
dataset.py
Writes NuroSim scenarios to disk in YOLO format so a real detector can be
trained on them. Labels come straight from the generator's ground truth.

    images/{train,val}/000123.png
    labels/{train,val}/000123.txt     # "<cls> <cx> <cy> <w> <h>" normalised
    data.yaml
"""

from __future__ import annotations

import logging
import os

import cv2
import yaml

from nurosim.scenario_generator import generate_scenario, CLASS_CONFIG, FRAME_W, FRAME_H
from nurosim.deploy.seeds import TRAIN_SEED, VAL_SEED

logger = logging.getLogger(__name__)


def _write_split(root: str, split: str, n: int, base_seed: int) -> None:
    img_dir = os.path.join(root, "images", split)
    lbl_dir = os.path.join(root, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(n):
        sc = generate_scenario(base_seed + i, seed=base_seed + i)
        stem = f"{base_seed + i:06d}"
        cv2.imwrite(os.path.join(img_dir, stem + ".png"), sc.frame)
        lines = []
        for b in sc.ground_truth:
            cx = (b.x1 + b.x2) / 2 / FRAME_W
            cy = (b.y1 + b.y2) / 2 / FRAME_H
            w  = (b.x2 - b.x1) / FRAME_W
            h  = (b.y2 - b.y1) / FRAME_H
            lines.append(f"{b.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))


def build_yolo_dataset(root: str, n_train: int = 3000, n_val: int = 300) -> str:
    """Generate the dataset and return the path to its data.yaml."""
    root = os.path.abspath(root)
    logger.info("Writing %d train / %d val scenarios to %s", n_train, n_val, root)
    _write_split(root, "train", n_train, TRAIN_SEED)
    _write_split(root, "val",   n_val,   VAL_SEED)

    names = {cfg["id"]: name for name, cfg in CLASS_CONFIG.items()}
    data_yaml = os.path.join(root, "data.yaml")
    with open(data_yaml, "w") as f:
        yaml.safe_dump({"path": root, "train": "images/train",
                        "val": "images/val", "names": names}, f, sort_keys=False)
    return data_yaml
