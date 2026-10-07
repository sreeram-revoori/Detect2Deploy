"""
batch.py
Offboard batch inference with Ray Data: label a frame corpus.

    read_parquet ─▶ decode JPEG (CPU tasks) ─▶ detect (GPU actors, model loaded once) ─▶ write_parquet

The two stages scale independently: decode is CPU-bound and fans out over
cores, inference runs in a pool of long-lived GPU actors. Reported: frames/s,
GPU utilisation sampled during the run (is the GPU actually busy, or starved
by decode?), Ray's per-operator stats, cost per million frames at a given
$/hour, and the label quality (mAP vs ground truth) of the output.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from nurosim.metrics import Evaluator
from nurosim.platform.backend import ServingModel, unpack
from nurosim.scenario_generator import generate_scenario

logger = logging.getLogger(__name__)


def decode_jpeg(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    frames = np.stack([cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in batch["jpeg"]])
    return {"scenario_id": batch["scenario_id"], "weather": batch["weather"], "frames": frames}


class Labeler:
    """Ray actor: one model instance, reused for every batch it receives."""

    def __init__(self, model_path: str, provider: str, max_batch: int, threads: Optional[int] = None):
        self.model = ServingModel(model_path, provider=provider, max_batch=max_batch,
                                  intra_op_threads=threads)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = self.model.infer(batch["frames"])
        return {"scenario_id": batch["scenario_id"], "weather": batch["weather"], **out}


class GpuSampler(threading.Thread):
    """Samples GPU utilisation (NVML) while the job runs."""

    def __init__(self, period_s: float = 0.25):
        super().__init__(daemon=True)
        self.period, self.samples, self._halt = period_s, [], threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml = pynvml
        except Exception:
            self._nvml = None

    def run(self) -> None:
        while self._nvml and not self._halt.is_set():
            self.samples.append(self._nvml.nvmlDeviceGetUtilizationRates(self._h).gpu)
            time.sleep(self.period)

    def stop(self) -> Dict[str, float]:
        self._halt.set()
        if not self.samples:
            return {}
        a = np.asarray(self.samples, dtype=float)
        return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                "p90": float(np.percentile(a, 90)), "samples": int(a.size)}


def label_quality(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ev = Evaluator(iou_threshold=0.5)
    for r in rows:
        packed = {k: np.asarray(r[k])[None] for k in ("num_dets", "det_boxes", "det_scores", "det_classes")}
        ev.add(generate_scenario(int(r["scenario_id"]), seed=int(r["scenario_id"])), unpack(packed)[0])
    res = ev.compute()
    return {"frames": len(rows), "map50_at_deploy_conf": res.map50, "precision": res.precision,
            "recall": res.recall, "per_class_ap50": res.per_class_ap50}


def run_batch(corpus_dir: str, model_path: str, out_dir: str, provider: str = "tensorrt",
              batch_size: int = 16, actors: int = 1, labeler_cpus: float = 1.0,
              decode_batch_size: int = 32, usd_per_hour: Optional[float] = None,
              eval_frames: int = 2000, tag: str = "") -> Dict[str, Any]:
    import ray

    os.makedirs(out_dir, exist_ok=True)
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False,
             logging_level=logging.WARNING)
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = False
    logging.getLogger("ray.data").setLevel(logging.WARNING)
    res = ray.cluster_resources()
    gpu = provider in ("cuda", "tensorrt")

    ds = ray.data.read_parquet(corpus_dir)
    ds = ds.map_batches(decode_jpeg, batch_size=decode_batch_size, batch_format="numpy")
    ds = ds.map_batches(Labeler, batch_size=batch_size, batch_format="numpy", concurrency=actors,
                        num_gpus=(1.0 / actors if actors > 1 else 1) if gpu else 0, num_cpus=labeler_cpus,
                        fn_constructor_kwargs={"model_path": os.path.abspath(model_path), "provider": provider,
                                               "max_batch": batch_size,
                                               "threads": None if gpu else max(1, int(labeler_cpus))})

    sampler = GpuSampler()
    sampler.start()
    t0 = time.perf_counter()
    labels = ds.materialize()                     # executes the streaming pipeline
    wall = time.perf_counter() - t0
    util = sampler.stop()
    n = labels.count()

    name = f"labels{('_' + tag) if tag else ''}"
    labels.write_parquet(os.path.join(out_dir, name))
    with open(os.path.join(out_dir, f"{name}_ray_stats.txt"), "w") as f:
        f.write(str(labels.stats()))

    fps = n / wall
    report = {
        "tag": tag, "provider": provider, "model": model_path, "frames": n, "wall_s": wall, "frames_per_s": fps,
        "batch_size": batch_size, "actors": actors, "labeler_cpus": labeler_cpus,
        "decode_batch_size": decode_batch_size,
        "cluster": {"cpus": res.get("CPU", 0), "gpus": res.get("GPU", 0)},
        "gpu_utilisation_pct": util,
        "hours_per_million_frames": 1e6 / fps / 3600,
        "usd_per_hour": usd_per_hour,
        "usd_per_million_frames": (1e6 / fps / 3600 * usd_per_hour) if usd_per_hour else None,
        "label_quality": label_quality(labels.take(min(eval_frames, n))) if eval_frames else None,
    }
    logger.info("batch %s: %d frames in %.1fs → %.0f frames/s, GPU util %s", tag, n, wall, fps,
                f"{util.get('mean', float('nan')):.0f}%" if util else "n/a")
    with open(os.path.join(out_dir, f"batch{('_' + tag) if tag else ''}.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return report


def batch_markdown(reports: List[Dict[str, Any]]) -> str:
    lines = ["### Offboard batch inference (Ray Data)", "",
             "| run | frames | frames/s | GPU util (mean) | batch | actors | CPUs | h / 1M frames | $ / 1M frames | label mAP@0.5 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        u = r["gpu_utilisation_pct"].get("mean") if r["gpu_utilisation_pct"] else None
        q = r["label_quality"]["map50_at_deploy_conf"] if r["label_quality"] else None
        cost = f"{r['usd_per_million_frames']:.2f}" if r["usd_per_million_frames"] else "—"
        lines.append(f"| {r['tag'] or '-'} | {r['frames']} | {r['frames_per_s']:.0f} | "
                     f"{f'{u:.0f}%' if u is not None else 'n/a'} | {r['batch_size']} | {r['actors']} | "
                     f"{r['cluster']['cpus']:.0f} | {r['hours_per_million_frames']:.2f} | {cost} | "
                     f"{f'{q:.4f}' if q is not None else '—'} |")
    return "\n".join(lines) + "\n"
