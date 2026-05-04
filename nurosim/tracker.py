"""
tracker.py
MLflow experiment tracking + matplotlib visualisation for NuroSim-Lite.

Logs:
  - All EvalResult metrics as MLflow scalars
  - Per-class AP bar chart
  - Weather mAP breakdown
  - Sample annotated frames (GT + predicted boxes)
  - Latency distribution histogram
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, List

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")                     # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from nurosim.metrics            import EvalResult
from nurosim.scenario_generator import BBox, CLASS_CONFIG

logger = logging.getLogger(__name__)

# ── Colour palette for plots ──────────────────────────────────────────────────
CLASS_COLORS_NORM = {
    "vehicle":    (0.0,  0.78, 1.0),
    "pedestrian": (0.0,  1.0,  0.31),
    "cyclist":    (1.0,  0.55, 0.0),
    "cone":       (0.0,  0.31, 1.0),
}


# ── Frame annotator ───────────────────────────────────────────────────────────

def annotate_frame(frame: np.ndarray,
                   gt_boxes:   List[BBox],
                   pred_boxes: List[BBox]) -> np.ndarray:
    """Draw GT (green dashed) and predicted (coloured) boxes on a frame copy."""
    out = frame.copy()

    # Ground truth — white dashed border
    for gt in gt_boxes:
        cv2.rectangle(out, (gt.x1, gt.y1), (gt.x2, gt.y2), (255, 255, 255), 1)
        # Dashed effect
        for i in range(gt.x1, gt.x2, 8):
            cv2.line(out, (i, gt.y1), (min(i+4, gt.x2), gt.y1), (255,255,255), 1)
            cv2.line(out, (i, gt.y2), (min(i+4, gt.x2), gt.y2), (255,255,255), 1)

    # Predictions — solid coloured boxes
    for pred in pred_boxes:
        cfg = CLASS_CONFIG[pred.class_name]
        color = cfg["color"]
        cv2.rectangle(out, (pred.x1, pred.y1), (pred.x2, pred.y2), color, 2)
        label = f"{pred.class_name[0].upper()}:{pred.confidence:.2f}"
        cv2.putText(out, label,
                    (pred.x1 + 2, pred.y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Legend
    y_off = 30
    for cls_name, col in CLASS_COLORS_NORM.items():
        bgr = tuple(int(c * 255) for c in reversed(col))
        cv2.rectangle(out, (4, y_off - 8), (14, y_off + 2), bgr, -1)
        cv2.putText(out, cls_name, (18, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1)
        y_off += 16

    return out


# ── Matplotlib figures ────────────────────────────────────────────────────────

def plot_per_class_ap(result: EvalResult, save_path: str) -> str:
    classes = list(result.per_class_ap50.keys())
    values  = [result.per_class_ap50[c] for c in classes]
    colors  = [CLASS_COLORS_NORM.get(c, (0.6, 0.6, 0.6)) for c in classes]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(classes, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(result.map50, color="red", linestyle="--", linewidth=1.2,
               label=f"mAP@0.5 = {result.map50:.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AP @ IoU=0.5", fontsize=11)
    ax.set_title("Per-Class Average Precision  (NuroSim-Lite)", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#13131f")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines[:].set_color("#444")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, color="white")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_weather_map(result: EvalResult, save_path: str) -> str:
    if not result.weather_map:
        return ""
    weathers = list(result.weather_map.keys())
    values   = [result.weather_map[w] for w in weathers]
    wx_colors = {"clear": "#00e5ff", "rain": "#4fc3f7",
                 "fog":   "#b0bec5", "night": "#7e57c2"}

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(weathers,
                  values,
                  color=[wx_colors.get(w, "#888") for w in weathers],
                  edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mAP@0.5", fontsize=11)
    ax.set_title("mAP@0.5 by Weather Condition", fontsize=12, pad=10)
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#13131f")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines[:].set_color("#444")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, color="white")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_latency_histogram(latencies: List[float], save_path: str) -> str:
    arr = np.array(latencies)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(arr, bins=40, color="#00bcd4", edgecolor="#006978", alpha=0.85)
    ax.axvline(np.percentile(arr, 50), color="yellow",
               linestyle="--", linewidth=1.2, label=f"p50={np.percentile(arr,50):.1f}ms")
    ax.axvline(np.percentile(arr, 95), color="orange",
               linestyle="--", linewidth=1.2, label=f"p95={np.percentile(arr,95):.1f}ms")
    ax.set_xlabel("Inference Latency (ms)", color="white", fontsize=10)
    ax.set_ylabel("Count", color="white", fontsize=10)
    ax.set_title("Detector Latency Distribution", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#13131f")
    ax.tick_params(colors="white")
    ax.title.set_color("white")
    ax.spines[:].set_color("#444")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── MLflow logger ─────────────────────────────────────────────────────────────

def log_to_mlflow(result: EvalResult,
                  stats: Dict[str, Any],
                  sample_frames: list | None = None,
                  latencies:     list | None = None,
                  run_name: str = "nurosim-eval") -> str:
    """
    Log a full evaluation run to MLflow.
    Returns the run_id.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed — skipping MLflow logging.")
        return ""

    os.makedirs("outputs", exist_ok=True)

    with mlflow.start_run(run_name=run_name) as run:
        # ── Scalar metrics ──
        mlflow.log_metrics({
            "map50":      result.map50,
            "map50_95":   result.map50_95,
            "precision":  result.precision,
            "recall":     result.recall,
            "f1":         result.f1,
            "total_tp":   result.total_tp,
            "total_fp":   result.total_fp,
            "total_fn":   result.total_fn,
        })

        for cls_name, ap in result.per_class_ap50.items():
            mlflow.log_metric(f"ap50_{cls_name}", ap)

        for wx, m in result.weather_map.items():
            mlflow.log_metric(f"map50_{wx}", m)

        # ── Run params ──
        mlflow.log_params({
            "n_scenarios":   stats.get("n_scenarios", result.n_scenarios),
            "n_workers":     stats.get("n_workers", 1),
            "backend":       stats.get("backend", "local"),
            "throughput_fps": stats.get("throughput_fps", 0),
            "lat_p95_ms":    stats.get("lat_p95_ms", 0),
        })

        # ── Figures ──
        ap_path  = "outputs/per_class_ap.png"
        wx_path  = "outputs/weather_map.png"
        lat_path = "outputs/latency_hist.png"

        plot_per_class_ap(result, ap_path)
        mlflow.log_artifact(ap_path, artifact_path="plots")

        if result.weather_map:
            plot_weather_map(result, wx_path)
            mlflow.log_artifact(wx_path, artifact_path="plots")

        if latencies:
            plot_latency_histogram(latencies, lat_path)
            mlflow.log_artifact(lat_path, artifact_path="plots")

        # ── Sample annotated frames ──
        if sample_frames:
            for i, (scenario, preds) in enumerate(sample_frames[:6]):
                ann = annotate_frame(scenario.frame,
                                     scenario.ground_truth, preds)
                frame_path = f"outputs/sample_{i:02d}_{scenario.weather}.png"
                cv2.imwrite(frame_path, ann)
                mlflow.log_artifact(frame_path, artifact_path="frames")

        run_id = run.info.run_id
        logger.info("MLflow run logged: %s", run_id)
        return run_id
