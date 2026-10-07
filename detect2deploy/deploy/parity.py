"""
parity.py
Does the deployed model still behave like the one we trained?

Every target in a profile runs over the same held-out scenarios (EVAL_SEED) and
is compared with the reference target (FP32 on CPU) at three levels:

  1. Task metrics   – mAP@0.5 / mAP@0.5:0.95, per class and per weather
                      condition. Aggregate mAP hides regressions: a variant can
                      lose 3 points on night pedestrians and still look fine.
  2. Detections     – after NMS at the deployment threshold: what fraction of
                      the reference detections the target reproduces (same
                      class, IoU >= 0.5), plus the score / box drift on matches.
  3. Raw tensors    – head output before NMS: box-coordinate error (px) and
                      class-score error over anchors the reference is
                      confident about.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from detect2deploy.deploy.config import Profile, Target
from detect2deploy.deploy.export import sha256_file
from detect2deploy.deploy.seeds import EVAL_SEED
from detect2deploy.inference.detector import InferenceDetector
from detect2deploy.metrics import Evaluator, iou_matrix
from detect2deploy.scenario_generator import BBox, generate_scenario

logger = logging.getLogger(__name__)


@dataclass
class TargetRun:
    target: Target
    active_providers: List[str]
    provider_active: bool
    metrics: Dict[str, Any]
    deploy_dets: List[List[BBox]] = field(repr=False, default_factory=list)
    raw: Optional[np.ndarray] = field(repr=False, default=None)


def _metrics_dict(result) -> Dict[str, Any]:
    return {
        "map50": result.map50,
        "map50_95": result.map50_95,
        "per_class_ap50": result.per_class_ap50,
        "weather_map50": result.weather_map,
        "n_gt": result.n_gt_boxes,
    }


def run_target(target: Target, n_scenarios: int, eval_conf: float,
               deploy_conf: float, nms_iou: float, raw_frames: int,
               base_seed: int = EVAL_SEED) -> TargetRun:
    det = InferenceDetector(conf_threshold=eval_conf, iou_threshold=nms_iou,
                            **target.detector_kwargs())
    ev = Evaluator(iou_threshold=0.5)
    deploy_dets: List[List[BBox]] = []
    raws = []

    t0 = time.perf_counter()
    for i in range(n_scenarios):
        sc = generate_scenario(base_seed + i, seed=base_seed + i)
        preds = det.predict(sc.frame)
        ev.add(sc, preds)
        # Operating-point detections are a subset of the low-threshold ones
        deploy_dets.append([p for p in preds if p.confidence >= deploy_conf])
        if i < raw_frames:
            raws.append(det.raw([sc.frame])[0])
    logger.info("  %-14s %d scenarios in %.1fs", target.name, n_scenarios,
                time.perf_counter() - t0)

    return TargetRun(
        target=target,
        active_providers=det.runtime.active_providers,
        provider_active=det.runtime.provider_active,
        metrics=_metrics_dict(ev.compute()),
        deploy_dets=deploy_dets,
        raw=np.stack(raws).astype(np.float32) if raws else None,
    )


def detection_agreement(ref: List[List[BBox]], tgt: List[List[BBox]]) -> Dict[str, float]:
    n_ref = n_tgt = n_match = 0
    conf_deltas, ious = [], []
    for r, t in zip(ref, tgt):
        n_ref += len(r)
        n_tgt += len(t)
        if not r or not t:
            continue
        iou = iou_matrix(r, t)
        same_cls = np.array([[a.class_id == b.class_id for b in t] for a in r])
        iou = np.where(same_cls, iou, 0.0)
        used = set()
        for ri in np.argsort([-b.confidence for b in r]):
            cand = [(iou[ri, ti], ti) for ti in range(len(t)) if ti not in used]
            if not cand:
                break
            best_iou, ti = max(cand)
            if best_iou >= 0.5:
                used.add(ti)
                n_match += 1
                ious.append(best_iou)
                conf_deltas.append(abs(r[ri].confidence - t[ti].confidence))
    return {
        "ref_dets": n_ref,
        "target_dets": n_tgt,
        "ref_recall": n_match / n_ref if n_ref else 1.0,      # ref dets reproduced
        "target_precision": n_match / n_tgt if n_tgt else 1.0, # target dets that exist in ref
        "mean_match_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_abs_conf_delta": float(np.mean(conf_deltas)) if conf_deltas else 0.0,
    }


def tensor_parity(ref: np.ndarray, tgt: np.ndarray, conf: float) -> Dict[str, float]:
    """ref/tgt: (N, 4 + nc, A). Box error over anchors the reference scores
    >= conf (elsewhere box regressions are meaningless noise)."""
    ref_scores, tgt_scores = ref[:, 4:], tgt[:, 4:]
    confident = ref_scores.max(axis=1) >= conf                    # (N, A)
    box_err = np.abs(ref[:, :4] - tgt[:, :4]).transpose(0, 2, 1)[confident]
    score_err = np.abs(ref_scores - tgt_scores)
    cls_agree = (ref_scores.argmax(1) == tgt_scores.argmax(1))[confident]
    return {
        "confident_anchors": int(confident.sum()),
        "box_mae_px": float(box_err.mean()) if box_err.size else 0.0,
        "box_max_err_px": float(box_err.max()) if box_err.size else 0.0,
        "score_mae": float(score_err.mean()),
        "score_max_err": float(score_err.max()),
        "class_agreement": float(cls_agree.mean()) if cls_agree.size else 1.0,
    }


def _deltas(ref: Dict[str, Any], tgt: Dict[str, Any]) -> Dict[str, Any]:
    """Positive = target is worse than reference."""
    return {
        "map50_drop": ref["map50"] - tgt["map50"],
        "map50_95_drop": ref["map50_95"] - tgt["map50_95"],
        "per_class_ap50_drop": {k: ref["per_class_ap50"][k] - tgt["per_class_ap50"].get(k, 0.0)
                                for k in ref["per_class_ap50"]},
        "weather_map50_drop": {k: ref["weather_map50"][k] - tgt["weather_map50"].get(k, 0.0)
                               for k in ref["weather_map50"]},
    }


def run_parity(profile: Profile, cfg: Dict[str, Any]) -> Dict[str, Any]:
    ecfg, dcfg = cfg["eval"], cfg["deploy"]
    runs: Dict[str, TargetRun] = {}
    logger.info("Parity: %d targets × %d held-out scenarios", len(profile.targets),
                ecfg["n_scenarios"])
    for t in profile.targets:
        runs[t.name] = run_target(t, ecfg["n_scenarios"], ecfg["conf_threshold"],
                                  dcfg["conf_threshold"], dcfg["iou_threshold"],
                                  ecfg["raw_parity_frames"])

    ref = runs[profile.reference]
    report: Dict[str, Any] = {
        "profile": profile.name,
        "reference": profile.reference,
        "host": platform.platform(),
        "eval": {**ecfg, "seed": EVAL_SEED, "deploy_conf": dcfg["conf_threshold"]},
        "accuracy_budget": cfg.get("accuracy_budget", {}),
        "negative_controls": [t.name for t in profile.targets if t.expect_fail],
        "targets": {},
    }
    for name, r in runs.items():
        entry = {
            "variant": r.target.variant,
            "provider": r.target.provider,
            "model_sha256": sha256_file(r.target.model_path),
            "active_providers": r.active_providers,
            "provider_active": r.provider_active,
            "metrics": r.metrics,
        }
        if name != profile.reference:
            entry["delta_vs_reference"] = _deltas(ref.metrics, r.metrics)
            entry["detection_agreement"] = detection_agreement(ref.deploy_dets, r.deploy_dets)
            if ref.raw is not None and r.raw is not None:
                entry["tensor_parity"] = tensor_parity(ref.raw, r.raw, dcfg["conf_threshold"])
        report["targets"][name] = entry
    return report


# ── Reporting ────────────────────────────────────────────────────────────────

def _neg(x: float) -> float:
    return -x + 0.0          # + 0.0 turns -0.0 into 0.0 for display


def parity_markdown(report: Dict[str, Any]) -> str:
    ref = report["reference"]
    tg = report["targets"]
    classes = list(tg[ref]["metrics"]["per_class_ap50"])
    weathers = list(tg[ref]["metrics"]["weather_map50"])

    lines = [f"### Accuracy parity — profile `{report['profile']}` "
             f"({report['eval']['n_scenarios']} held-out scenarios, reference `{ref}`)", "",
             "| target | EP active | mAP@0.5 | Δ | mAP@.5:.95 | Δ | worst class Δ | worst weather Δ |",
             "|---|---|---|---|---|---|---|---|"]
    for name, e in tg.items():
        m = e["metrics"]
        if name == ref:
            lines.append(f"| **{name}** | {'✅' if e['provider_active'] else '❌'} | "
                         f"{m['map50']:.4f} | — | {m['map50_95']:.4f} | — | — | — |")
            continue
        d = e["delta_vs_reference"]
        wc = max(d["per_class_ap50_drop"].items(), key=lambda kv: kv[1])
        ww = max(d["weather_map50_drop"].items(), key=lambda kv: kv[1])
        lines.append(f"| {name} | {'✅' if e['provider_active'] else '❌'} | {m['map50']:.4f} | "
                     f"{_neg(d['map50_drop']):+.4f} | {m['map50_95']:.4f} | {_neg(d['map50_95_drop']):+.4f} | "
                     f"{wc[0]} {_neg(wc[1]):+.4f} | {ww[0]} {_neg(ww[1]):+.4f} |")

    lines += ["", "**AP@0.5 per class**", "",
              "| target | " + " | ".join(classes) + " |",
              "|---|" + "---|" * len(classes)]
    for name, e in tg.items():
        ap = e["metrics"]["per_class_ap50"]
        lines.append(f"| {name} | " + " | ".join(f"{ap[c]:.3f}" for c in classes) + " |")

    lines += ["", "**mAP@0.5 per weather**", "",
              "| target | " + " | ".join(weathers) + " |",
              "|---|" + "---|" * len(weathers)]
    for name, e in tg.items():
        wm = e["metrics"]["weather_map50"]
        lines.append(f"| {name} | " + " | ".join(f"{wm[w]:.3f}" for w in weathers) + " |")

    lines += ["", f"**Output parity vs `{ref}`** (detections at conf ≥ "
              f"{report['eval']['deploy_conf']}; tensors over {report['eval']['raw_parity_frames']} frames)", "",
              "| target | ref dets reproduced | extra dets | match IoU | |Δconf| | box MAE px | box max px | score MAE | class agree |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, e in tg.items():
        if name == ref:
            continue
        a, t = e["detection_agreement"], e.get("tensor_parity", {})
        lines.append(f"| {name} | {a['ref_recall']:.3f} | {1 - a['target_precision']:.3f} | "
                     f"{a['mean_match_iou']:.3f} | {a['mean_abs_conf_delta']:.4f} | "
                     f"{t.get('box_mae_px', float('nan')):.3f} | {t.get('box_max_err_px', float('nan')):.2f} | "
                     f"{t.get('score_mae', float('nan')):.5f} | {t.get('class_agreement', float('nan')):.4f} |")
    return "\n".join(lines) + "\n"


def plot_parity_deltas(report: Dict[str, Any], save_path: str) -> str:
    """Δ vs reference (percentage points) per weather and per class, with the
    gate's budget drawn in. Negative controls are left out — they sit at −100."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tg = report["targets"]
    skip = set(report.get("negative_controls", [])) | {report["reference"]}
    names = [n for n in tg if n not in skip]
    budget = report.get("accuracy_budget", {})
    panels = [("weather_map50_drop", "mAP@0.5 Δ by weather", "max_map50_drop_per_weather"),
              ("per_class_ap50_drop", "AP@0.5 Δ by class", "max_ap50_drop_per_class")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (key, title, bkey) in zip(axes, panels):
        cats = list(tg[names[0]]["delta_vs_reference"][key]) if names else []
        x = np.arange(len(cats))
        w = 0.8 / max(1, len(names))
        for i, n in enumerate(names):
            vals = [-100 * tg[n]["delta_vs_reference"][key][c] for c in cats]
            ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=n)
        if bkey in budget:
            ax.axhline(-100 * budget[bkey], color="crimson", linestyle="--", linewidth=1)
            ax.text(len(cats) - 0.5, -100 * budget[bkey], " budget", color="crimson",
                    va="center", ha="left", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x, cats)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"Δ vs {report['reference']} (points)")
    axes[1].legend(fontsize=8, loc="lower left")
    note = ", ".join(report.get("negative_controls", []))
    fig.suptitle(f"Accuracy parity per deployment target ({report['profile']})"
                 + (f" — negative control {note} omitted (≈ −100)" if note else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def write_parity(report: Dict[str, Any], out_dir: str = "reports") -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"parity_{report['profile']}")
    with open(stem + ".json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(stem + ".md", "w") as f:
        f.write(parity_markdown(report))
    plot_parity_deltas(report, stem + "_deltas.png")
    return stem + ".json"
