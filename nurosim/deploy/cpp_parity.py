"""
cpp_parity.py
Accuracy parity for the C++ / TensorRT path.

`nurosim_bench --dump-dets` writes detections for the held-out eval frames.
This scores them exactly like the Python parity check: mAP per class and per
weather vs ground truth, deltas vs the ONNX Runtime FP32 CPU reference, and
detection-level agreement at the deployment threshold. It validates the whole
C++ path end to end — GPU letterbox kernel, TensorRT precision, EfficientNMS —
not just the network.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Sequence

from nurosim.deploy.parity import _deltas, _metrics_dict, _neg, detection_agreement
from nurosim.deploy.seeds import EVAL_SEED
from nurosim.metrics import Evaluator
from nurosim.scenario_generator import BBox, CLASS_CONFIG, generate_scenario

logger = logging.getLogger(__name__)
CLASS_NAMES = {v["id"]: k for k, v in CLASS_CONFIG.items()}


def load_dets(path: str) -> List[List[BBox]]:
    with open(path) as f:
        data = json.load(f)
    return [[BBox(int(round(d[0])), int(round(d[1])), int(round(d[2])), int(round(d[3])),
                  int(d[4]), CLASS_NAMES.get(int(d[4]), str(int(d[4]))), float(d[5]))
             for d in frame] for frame in data["frames"]]


def _evaluate(scenarios, preds: Sequence[List[BBox]]) -> Dict[str, Any]:
    ev = Evaluator(iou_threshold=0.5)
    for sc, p in zip(scenarios, preds):
        ev.add(sc, p)
    return _metrics_dict(ev.compute())


def run_cpp_parity(cfg: Dict[str, Any], det_files: Sequence[str]) -> Dict[str, Any]:
    from nurosim.inference.detector import InferenceDetector

    runs = {os.path.splitext(os.path.basename(p))[0].removeprefix("dets_"): load_dets(p)
            for p in det_files}
    n = min(len(d) for d in runs.values())
    scenarios = [generate_scenario(EVAL_SEED + i, seed=EVAL_SEED + i) for i in range(n)]

    deploy_conf = cfg["deploy"]["conf_threshold"]
    ref_det = InferenceDetector(cfg["models"]["fp32"], conf_threshold=cfg["eval"]["conf_threshold"],
                                iou_threshold=cfg["deploy"]["iou_threshold"])
    ref_preds = [ref_det.predict(sc.frame) for sc in scenarios]
    ref_metrics = _evaluate(scenarios, ref_preds)
    ref_deploy = [[b for b in p if b.confidence >= deploy_conf] for p in ref_preds]

    budget = cfg["accuracy_budget"]
    report: Dict[str, Any] = {"reference": "ort-fp32-cpu", "n_frames": n,
                              "accuracy_budget": budget,
                              "targets": {"ort-fp32-cpu": {"metrics": ref_metrics}}}
    for name, preds in runs.items():
        preds = preds[:n]
        m = _evaluate(scenarios, preds)
        d = _deltas(ref_metrics, m)
        report["targets"][name] = {
            "metrics": m,
            "delta_vs_reference": d,
            "detection_agreement": detection_agreement(
                ref_deploy, [[b for b in p if b.confidence >= deploy_conf] for p in preds]),
            "within_budget": (d["map50_drop"] <= budget["max_map50_drop"]
                              and max(d["weather_map50_drop"].values()) <= budget["max_map50_drop_per_weather"]
                              and max(d["per_class_ap50_drop"].values()) <= budget["max_ap50_drop_per_class"]),
        }
    return report


def cpp_parity_markdown(report: Dict[str, Any]) -> str:
    tg = report["targets"]
    ref = tg[report["reference"]]["metrics"]
    classes = list(ref["per_class_ap50"])
    lines = [f"### TensorRT C++ path — accuracy parity ({report['n_frames']} held-out scenarios, "
             f"reference ONNX Runtime FP32 CPU)", "",
             "| engine | mAP@0.5 | Δ | mAP@.5:.95 | Δ | " + " | ".join(f"{c} AP" for c in classes)
             + " | worst weather Δ | ref dets reproduced | budget |",
             "|---|---|---|---|---|" + "---|" * len(classes) + "---|---|---|"]
    for name, e in tg.items():
        m = e["metrics"]
        aps = " | ".join(f"{m['per_class_ap50'][c]:.3f}" for c in classes)
        if name == report["reference"]:
            lines.append(f"| **{name}** | {m['map50']:.4f} | — | {m['map50_95']:.4f} | — | {aps} | — | — | — |")
            continue
        d = e["delta_vs_reference"]
        wx, wd = max(d["weather_map50_drop"].items(), key=lambda kv: kv[1])
        lines.append(f"| {name} | {m['map50']:.4f} | {_neg(d['map50_drop']):+.4f} | {m['map50_95']:.4f} | "
                     f"{_neg(d['map50_95_drop']):+.4f} | {aps} | {wx} {_neg(wd):+.4f} | "
                     f"{e['detection_agreement']['ref_recall']:.3f} | "
                     f"{'✅' if e['within_budget'] else '❌'} |")
    return "\n".join(lines) + "\n"


def write_cpp_parity(report: Dict[str, Any], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "parity_cpp.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "parity_cpp.md"), "w") as f:
        f.write(cpp_parity_markdown(report))
    return path
