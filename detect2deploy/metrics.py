"""
metrics.py
Evaluation metrics for object detection.

Implements:
  - IoU (Intersection over Union)
  - Precision / Recall at a given IoU threshold
  - Average Precision (AP) via 101-point interpolation (COCO-style)
  - mean Average Precision (mAP@0.5, mAP@0.5:0.95)
  - Per-class breakdown
  - Confusion matrix
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from detect2deploy.scenario_generator import BBox, CLASS_CONFIG


# ── IoU ───────────────────────────────────────────────────────────────────────

def compute_iou(box_a: BBox, box_b: BBox) -> float:
    """Compute IoU between two BBoxes."""
    ix1 = max(box_a.x1, box_b.x1)
    iy1 = max(box_a.y1, box_b.y1)
    ix2 = min(box_a.x2, box_b.x2)
    iy2 = min(box_a.y2, box_b.y2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    union = box_a.area() + box_b.area() - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(preds: List[BBox], gts: List[BBox]) -> np.ndarray:
    """Return NxM IoU matrix (N=preds, M=gts), vectorised with NumPy."""
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)))
    p = np.array([b.xyxy for b in preds], dtype=np.float64)       # (N, 4)
    g = np.array([b.xyxy for b in gts],   dtype=np.float64)       # (M, 4)
    ix1 = np.maximum(p[:, None, 0], g[None, :, 0])
    iy1 = np.maximum(p[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(p[:, None, 2], g[None, :, 2])
    iy2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter  = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_p = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
    area_g = np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None)
    union  = area_p[:, None] + area_g[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


# ── Average Precision ─────────────────────────────────────────────────────────

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute AP using 101-point COCO interpolation.
    recalls and precisions must be sorted by recall ascending.
    """
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for thr in recall_thresholds:
        prec_at_rec = precisions[recalls >= thr]
        ap += (np.max(prec_at_rec) if len(prec_at_rec) > 0 else 0.0)
    return ap / 101.0


# ── Per-class PR curve ────────────────────────────────────────────────────────

def _class_pr(all_preds: List[Tuple[float, bool, int]],
              n_gt: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build precision-recall curve for one class.

    all_preds : list of (confidence, is_tp, img_idx) sorted by conf desc
    n_gt      : total GT boxes for this class across all images
    """
    if n_gt == 0 or not all_preds:
        return np.array([0.]), np.array([0.]), 0.0

    all_preds = sorted(all_preds, key=lambda x: -x[0])  # sort by conf desc

    tp_cum = np.zeros(len(all_preds))
    fp_cum = np.zeros(len(all_preds))

    for i, (_, is_tp, _) in enumerate(all_preds):
        tp_cum[i] = (tp_cum[i-1] if i > 0 else 0) + int(is_tp)
        fp_cum[i] = (fp_cum[i-1] if i > 0 else 0) + int(not is_tp)

    recalls    = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

    ap = compute_ap(recalls, precisions)
    return recalls, precisions, ap


# ── Main Evaluator ────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Holds all evaluation metrics for one experiment run."""
    map50:           float
    map50_95:        float
    per_class_ap50:  Dict[str, float]
    precision:       float
    recall:          float
    f1:              float
    total_tp:        int
    total_fp:        int
    total_fn:        int
    n_scenarios:     int
    n_gt_boxes:      int
    n_pred_boxes:    int
    weather_map:     Dict[str, float]    = field(default_factory=dict)
    confusion_matrix: np.ndarray | None  = None

    def summary(self) -> str:
        lines = [
            "=" * 52,
            "  Detect2Deploy  ·  Evaluation Results",
            "=" * 52,
            f"  Scenarios evaluated  : {self.n_scenarios}",
            f"  GT boxes             : {self.n_gt_boxes}",
            f"  Predicted boxes      : {self.n_pred_boxes}",
            f"  mAP@0.5              : {self.map50:.4f}",
            f"  mAP@0.5:0.95         : {self.map50_95:.4f}",
            f"  Precision            : {self.precision:.4f}",
            f"  Recall               : {self.recall:.4f}",
            f"  F1                   : {self.f1:.4f}",
            f"  TP / FP / FN         : {self.total_tp} / {self.total_fp} / {self.total_fn}",
            "-" * 52,
            "  Per-class AP@0.5:",
        ]
        for cls, ap in self.per_class_ap50.items():
            lines.append(f"    {cls:<12} : {ap:.4f}")
        if self.weather_map:
            lines.append("-" * 52)
            lines.append("  mAP@0.5 by weather:")
            for wx, m in self.weather_map.items():
                lines.append(f"    {wx:<8} : {m:.4f}")
        lines.append("=" * 52)
        return "\n".join(lines)


class Evaluator:
    """
    Accumulates predictions across many scenarios then computes mAP.

    Usage:
        ev = Evaluator(iou_threshold=0.5)
        for scenario in scenarios:
            preds = detector.predict(scenario.frame, scenario.ground_truth)
            ev.add(scenario, preds)
        result = ev.compute()
    """

    IOU_THRESHOLDS_095 = np.arange(0.5, 1.0, 0.05)   # 10 thresholds for mAP@0.5:0.95

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_thr  = iou_threshold
        self._records: List[dict] = []           # one dict per scenario

    def reset(self) -> None:
        self._records.clear()

    def add(self, scenario, predictions: List[BBox]) -> None:
        self._records.append({
            "scenario_id": scenario.scenario_id,
            "weather":     scenario.weather,
            "gt":          scenario.ground_truth,
            "preds":       predictions,
        })

    # ── internal: match preds → gt for a single IoU threshold ────────────────

    def _match_single_thr(self, thr: float):
        """
        Returns:
            class_preds  : {class_id: [(conf, is_tp, img_idx), ...]}
            class_n_gt   : {class_id: int}
            total_tp/fp/fn
        """
        class_preds: Dict[int, List[Tuple[float, bool, int]]] = defaultdict(list)
        class_n_gt:  Dict[int, int] = defaultdict(int)
        total_tp = total_fp = total_fn = 0

        for img_idx, rec in enumerate(self._records):
            gts   = rec["gt"]
            preds = rec["preds"]

            for g in gts:
                class_n_gt[g.class_id] += 1

            if not preds:
                total_fn += len(gts)
                continue
            if not gts:
                total_fp += len(preds)
                for p in preds:
                    class_preds[p.class_id].append((p.confidence, False, img_idx))
                continue

            iou_mat    = iou_matrix(preds, gts)
            gt_matched = [False] * len(gts)

            # Sort preds by descending confidence for greedy matching
            pred_order = sorted(range(len(preds)),
                                key=lambda i: -preds[i].confidence)

            for pi in pred_order:
                p   = preds[pi]
                row = iou_mat[pi]

                # Only consider GTs of the same class
                best_iou = thr - 1e-9
                best_gi  = -1
                for gi, g in enumerate(gts):
                    if g.class_id != p.class_id:
                        continue
                    if gt_matched[gi]:
                        continue
                    if row[gi] > best_iou:
                        best_iou = row[gi]
                        best_gi  = gi

                is_tp = best_gi >= 0
                if is_tp:
                    gt_matched[best_gi] = True
                    total_tp += 1
                else:
                    total_fp += 1

                class_preds[p.class_id].append((p.confidence, is_tp, img_idx))

            total_fn += sum(1 for m in gt_matched if not m)

        return class_preds, class_n_gt, total_tp, total_fp, total_fn

    # ── public compute ────────────────────────────────────────────────────────

    def compute(self) -> EvalResult:
        if not self._records:
            raise ValueError("No records added — call .add() first.")

        class_names   = list(CLASS_CONFIG.keys())
        class_id_map  = {v["id"]: k for k, v in CLASS_CONFIG.items()}

        # ── mAP@0.5 ──
        cp50, ng50, ttp, tfp, tfn = self._match_single_thr(0.5)

        per_class_ap50: Dict[str, float] = {}
        ap50_list = []
        for cls_name in class_names:
            cls_id = CLASS_CONFIG[cls_name]["id"]
            _, _, ap = _class_pr(cp50.get(cls_id, []), ng50.get(cls_id, 0))
            per_class_ap50[cls_name] = ap
            ap50_list.append(ap)
        map50 = float(np.mean(ap50_list))

        # ── mAP@0.5:0.95 ──
        map_per_thr = []
        for thr in self.IOU_THRESHOLDS_095:
            cp_t, ng_t, *_ = self._match_single_thr(float(thr))
            aps = []
            for cls_name in class_names:
                cid = CLASS_CONFIG[cls_name]["id"]
                _, _, ap = _class_pr(cp_t.get(cid, []), ng_t.get(cid, 0))
                aps.append(ap)
            map_per_thr.append(np.mean(aps))
        map50_95 = float(np.mean(map_per_thr))

        # ── Precision / Recall / F1 ──
        precision = ttp / (ttp + tfp + 1e-9)
        recall    = ttp / (ttp + tfn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        # ── Weather breakdown ──
        weather_map: Dict[str, float] = {}
        from itertools import groupby
        weather_groups = defaultdict(list)
        for r in self._records:
            weather_groups[r["weather"]].append(r)

        ev_tmp = Evaluator(iou_threshold=0.5)
        for wx, recs in weather_groups.items():
            ev_tmp.reset()
            # Re-use internal record list
            ev_tmp._records = recs
            cp_w, ng_w, *_ = ev_tmp._match_single_thr(0.5)
            aps_w = []
            for cls_name in class_names:
                cid = CLASS_CONFIG[cls_name]["id"]
                _, _, ap = _class_pr(cp_w.get(cid, []), ng_w.get(cid, 0))
                aps_w.append(ap)
            weather_map[wx] = float(np.mean(aps_w))

        # ── Aggregate counts ──
        n_gt   = sum(len(r["gt"])    for r in self._records)
        n_pred = sum(len(r["preds"]) for r in self._records)

        return EvalResult(
            map50=map50,
            map50_95=map50_95,
            per_class_ap50=per_class_ap50,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            total_tp=ttp,
            total_fp=tfp,
            total_fn=tfn,
            n_scenarios=len(self._records),
            n_gt_boxes=n_gt,
            n_pred_boxes=n_pred,
            weather_map=weather_map,
        )
