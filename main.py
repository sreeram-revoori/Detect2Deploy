#!/usr/bin/env python3
"""
main.py
NuroSim-Lite  ·  End-to-end AV simulation & perception evaluation pipeline.

Quick start:
    python main.py                          # 500 scenarios, 4 workers
    python main.py --n 100 --workers 2      # fast smoke test
    python main.py --n 1000 --workers 8 --mlflow  # full run with MLflow
    python main.py --model models/nurosim_det_int8.onnx   # real detector (ONNX)
"""

import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np

from nurosim.scenario_generator import generate_scenario, WEATHER_CONDITIONS
from nurosim.perception_model   import TimedDetector, build_detector
from nurosim.metrics            import Evaluator
from nurosim.ray_worker         import ParallelEvaluator
from nurosim.deploy.seeds       import EVAL_SEED
from nurosim.tracker            import (
    annotate_frame, log_to_mlflow,
    plot_per_class_ap, plot_weather_map, plot_latency_histogram,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nurosim.main")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NuroSim-Lite evaluation pipeline")
    p.add_argument("--n",        type=int,   default=500,   help="Number of scenarios")
    p.add_argument("--workers",  type=int,   default=4,     help="Parallel workers")
    p.add_argument("--seed",     type=int,   default=EVAL_SEED,
                   help="Base RNG seed (default: held-out range, disjoint from training)")
    p.add_argument("--iou",      type=float, default=0.5,   help="IoU threshold")
    p.add_argument("--no-ray",   action="store_true",       help="Disable Ray, use multiprocessing")
    p.add_argument("--mlflow",   action="store_true",       help="Log to MLflow")
    p.add_argument("--save-frames", action="store_true",    help="Save annotated sample frames")
    p.add_argument("--output-dir", default="outputs",       help="Output directory")
    p.add_argument("--model",    default=None,
                   help="ONNX model to evaluate instead of the MockDetector")
    p.add_argument("--provider", default="cpu",
                   help="ONNX Runtime EP: cpu | coreml | cuda | tensorrt")
    return p.parse_args()


def detector_spec(args) -> dict:
    if args.model is None:
        return {"kind": "mock"}
    # One ORT session per worker; split the cores between workers instead of
    # letting every session spawn a thread per core (oversubscription).
    threads = max(1, (os.cpu_count() or 1) // max(1, args.workers))
    return {"kind": "onnx", "model_path": args.model, "provider": args.provider,
            "intra_op_threads": threads}


# ── Demo: single scenario visual ─────────────────────────────────────────────

def demo_single_scenario(output_dir: str, spec: dict) -> None:
    logger.info("Generating demo scenarios for each weather condition …")
    os.makedirs(output_dir, exist_ok=True)

    detector = TimedDetector(build_detector(spec, seed=99))

    for wx in WEATHER_CONDITIONS:
        sc    = generate_scenario(0, seed=999, weather=wx, num_objects=8)
        preds = detector.predict(sc.frame, ground_truth=sc.ground_truth)
        ann   = annotate_frame(sc.frame, sc.ground_truth, preds)
        path  = os.path.join(output_dir, f"demo_{wx}.png")
        cv2.imwrite(path, ann)
        logger.info("  Saved demo frame: %s  (GT=%d  Pred=%d)",
                    path, len(sc.ground_truth), len(preds))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 56)
    logger.info("  NuroSim-Lite  ·  AV Perception Evaluation")
    logger.info("  Scenarios : %d  |  Workers : %d  |  IoU : %.2f",
                args.n, args.workers, args.iou)
    logger.info("=" * 56)

    # ── Step 1: demo frames ──
    spec = detector_spec(args)
    logger.info("  Detector  : %s", args.model or "MockDetector")
    demo_single_scenario(args.output_dir, spec)

    # ── Step 2: parallel evaluation ──
    logger.info("Launching parallel evaluator …")
    pe = ParallelEvaluator(
        n_scenarios=args.n,
        n_workers=args.workers,
        iou_thr=args.iou,
        base_seed=args.seed,
        use_ray=not args.no_ray,
        detector_spec=spec,
    )

    result, stats = pe.run()

    # ── Step 3: print summary ──
    print("\n" + result.summary())
    print(f"\n  Backend           : {stats['backend']}")
    print(f"  Wall time         : {stats['total_wall_s']:.2f}s")
    print(f"  Throughput        : {stats['throughput_fps']:.1f} scenarios/s")
    print(f"  Latency p50/p95   : {stats['lat_p50_ms']:.1f}ms / {stats['lat_p95_ms']:.1f}ms\n")

    # ── Step 4: save plots ──
    plot_per_class_ap(result,
                      os.path.join(args.output_dir, "per_class_ap.png"))
    plot_weather_map(result,
                     os.path.join(args.output_dir, "weather_map.png"))

    logger.info("Plots saved to %s/", args.output_dir)

    # ── Step 5: save sample annotated frames ──
    if args.save_frames:
        from nurosim.scenario_generator import generate_scenario as gs
        det = TimedDetector(build_detector(spec, seed=0))
        samples = []
        for sid in range(min(12, args.n)):
            sc    = gs(sid, seed=args.seed + sid)
            preds = det.predict(sc.frame, ground_truth=sc.ground_truth)
            samples.append((sc, preds))
            ann   = annotate_frame(sc.frame, sc.ground_truth, preds)
            path  = os.path.join(args.output_dir, f"frame_{sid:03d}_{sc.weather}.png")
            cv2.imwrite(path, ann)
        logger.info("Saved %d annotated frames to %s/", len(samples), args.output_dir)

        # also build latency list for plot
        lat_path = os.path.join(args.output_dir, "latency_hist.png")
        plot_latency_histogram(det.latencies, lat_path)

    # ── Step 6: optional MLflow logging ──
    if args.mlflow:
        sample_pairs = []
        if args.save_frames:
            sample_pairs = samples[:6]          # type: ignore[possibly-undefined]
        run_id = log_to_mlflow(
            result=result,
            stats=stats,
            sample_frames=sample_pairs if sample_pairs else None,
            latencies=None,
            run_name=f"nurosim-n{args.n}-w{args.workers}",
        )
        if run_id:
            logger.info("MLflow run ID: %s", run_id)
            logger.info("View UI:  mlflow ui --port 5000")

    logger.info("Done. ✓")
    return result


if __name__ == "__main__":
    main()
