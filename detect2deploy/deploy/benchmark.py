"""
benchmark.py
Latency benchmarking with an onboard-inference mindset.

Methodology
  * Cold start reported separately: session/engine build time and the first
    inference (lazy allocation, kernel selection) are what a vehicle pays at
    boot — they are excluded from steady-state numbers, not averaged in.
  * Warmup iterations discarded before measuring.
  * End-to-end at batch 1 (one camera frame → boxes), split into
    preprocess / inference / postprocess, since a fast engine wrapped in
    slow Python pre/post is still a slow perception stage.
  * Tail-focused: p99, max and jitter (p99 − p50). A planner consuming
    detections cares about the worst frame, not the average one.
  * Inference outputs land in host memory, so ORT's run() is synchronous and
    wall-clock timing is valid for GPU EPs too. (With IO binding / device
    outputs you must synchronise the stream before reading the clock.)
  * Batch sweep (inference only) for offboard / batch-inference throughput.
  * Rotating set of distinct frames — NMS cost depends on the scene.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from typing import Any, Dict, List, Sequence

import numpy as np

from detect2deploy.deploy.config import Profile, Target
from detect2deploy.deploy.export import sha256_file
from detect2deploy.deploy.seeds import BENCH_SEED
from detect2deploy.inference.detector import InferenceDetector
from detect2deploy.inference.preprocess import preprocess_batch
from detect2deploy.scenario_generator import generate_scenario

logger = logging.getLogger(__name__)


def summarize(samples_ms: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(samples_ms, dtype=np.float64)
    p50, p99 = np.percentile(a, [50, 99])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p50": float(p50),
        "p90": float(np.percentile(a, 90)),
        "p99": float(p99),
        "max": float(a.max()),
        "jitter_p99_p50": float(p99 - p50),
    }


def host_info() -> Dict[str, Any]:
    import onnxruntime as ort
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
    }


def bench_frames(n: int = 64) -> List[np.ndarray]:
    return [generate_scenario(BENCH_SEED + i, seed=BENCH_SEED + i).frame for i in range(n)]


def benchmark_target(target: Target, frames: Sequence[np.ndarray],
                     bcfg: Dict[str, Any], dcfg: Dict[str, Any]) -> Dict[str, Any]:
    def make_detector():
        return InferenceDetector(conf_threshold=dcfg["conf_threshold"],
                                 iou_threshold=dcfg["iou_threshold"],
                                 intra_op_threads=bcfg.get("intra_op_threads"),
                                 **target.detector_kwargs())

    # Several independent sessions: on this laptop FP32 CPU latency is bimodal
    # *between* sessions (~17 vs ~30 ms p50) while stable within one, so a
    # single-session benchmark can report either mode. Samples are pooled and
    # the per-session p50s are kept so the spread is visible.
    pre, inf, post, total, session_p50s = [], [], [], [], []
    det = None
    for sess in range(bcfg.get("sessions", 1)):
        del det
        t0 = time.perf_counter()
        det = make_detector()
        if sess == 0:
            load_ms = (time.perf_counter() - t0) * 1e3
            _, first = det.predict_timed(frames[0])
        for i in range(bcfg["warmup"]):
            det.predict_timed(frames[i % len(frames)])
        sess_total = []
        for i in range(bcfg["iters"]):
            _, t = det.predict_timed(frames[i % len(frames)])
            pre.append(t.pre_ms)
            inf.append(t.infer_ms)
            post.append(t.post_ms)
            sess_total.append(t.total_ms)
        total += sess_total
        session_p50s.append(float(np.percentile(sess_total, 50)))

    # Batch sweep — inference only. Static-batch targets (CoreML) skip it.
    sweep = {}
    sizes = [1] if target.static_batch else bcfg["batch_sizes"]
    for bs in sizes:
        batch, _ = preprocess_batch(frames[:bs])
        for _ in range(max(3, bcfg["warmup"] // bs)):
            det.runtime.run(batch)
        lat = []
        for _ in range(max(20, bcfg["iters"] // bs)):
            s = time.perf_counter()
            det.runtime.run(batch)
            lat.append((time.perf_counter() - s) * 1e3)
        st = summarize(lat)
        sweep[str(bs)] = {"p50_ms": st["p50"], "p99_ms": st["p99"],
                          "throughput_fps": bs * 1e3 / st["p50"]}

    return {
        "variant": target.variant,
        "provider": target.provider,
        "active_providers": det.runtime.active_providers,
        "provider_active": det.runtime.provider_active,
        "model": target.model_path,
        "model_mb": round(os.path.getsize(target.model_path) / 1e6, 2),
        "model_sha256": sha256_file(target.model_path),
        "cold_start": {"session_load_ms": load_ms, "first_inference_ms": first.total_ms},
        "e2e_batch1": {"total": summarize(total), "pre": summarize(pre),
                       "infer": summarize(inf), "post": summarize(post),
                       "session_p50s": session_p50s},
        "batch_sweep": sweep,
        "samples_total_ms": total,
    }


def run_benchmark(profile: Profile, cfg: Dict[str, Any]) -> Dict[str, Any]:
    bcfg, dcfg = dict(cfg["benchmark"]), cfg["deploy"]
    if profile.intra_op_threads:
        bcfg["intra_op_threads"] = profile.intra_op_threads
    frames = bench_frames(bcfg.get("n_frames", 64))
    report = {"profile": profile.name, "host": host_info(),
              "settings": {**bcfg, "deploy": dcfg}, "targets": {}}
    for t in profile.targets:
        logger.info("  benchmarking %s", t.name)
        report["targets"][t.name] = benchmark_target(t, frames, bcfg, dcfg)
    return report


# ── Reporting ────────────────────────────────────────────────────────────────

def benchmark_markdown(report: Dict[str, Any]) -> str:
    h = report["host"]
    s = report["settings"]
    lines = [f"### Latency — profile `{report['profile']}`", "",
             f"Host: `{h['platform']}`, {h['cpu_count']} cores, onnxruntime {h['onnxruntime']}. "
             f"Batch 1, {s.get('sessions', 1)} independent sessions × ({s['warmup']} warmup + "
             f"{s['iters']} measured iterations), pooled, "
             f"deploy conf {s['deploy']['conf_threshold']}, "
             f"intra-op threads: {s.get('intra_op_threads') or 'ORT default (all cores)'}.", "",
             "| target | EP active | model MB | e2e p50 | e2e p99 | max | jitter | per-session p50 | pre p50 | infer p50 | post p50 | cold start |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in report["targets"].items():
        e = r["e2e_batch1"]
        cold = r["cold_start"]["session_load_ms"] + r["cold_start"]["first_inference_ms"]
        lines.append(
            f"| {name} | {'✅' if r['provider_active'] else '❌'} | {r['model_mb']} | "
            f"{e['total']['p50']:.2f} | {e['total']['p99']:.2f} | {e['total']['max']:.2f} | "
            f"{e['total']['jitter_p99_p50']:.2f} | "
            f"{' / '.join(f'{v:.1f}' for v in e.get('session_p50s', []))} | "
            f"{e['pre']['p50']:.2f} | {e['infer']['p50']:.2f} | "
            f"{e['post']['p50']:.2f} | {cold:.0f} |")
    lines += ["", "All latencies in ms. *cold start* = session load + first inference.", "",
              "**Batch sweep** (inference only, throughput in frames/s at p50)", ""]
    sizes = sorted({int(b) for r in report["targets"].values() for b in r["batch_sweep"]})
    lines += ["| target | " + " | ".join(f"bs={b}" for b in sizes) + " |",
              "|---|" + "---|" * len(sizes)]
    for name, r in report["targets"].items():
        cells = []
        for b in sizes:
            v = r["batch_sweep"].get(str(b))
            cells.append(f"{v['throughput_fps']:.0f} fps ({v['p50_ms']:.1f} ms)" if v else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_latency_cdf(report: Dict[str, Any], save_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.ticker import NullFormatter, ScalarFormatter

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for name, r in report["targets"].items():
        a = np.sort(r["samples_total_ms"])
        ax.plot(a, np.arange(1, a.size + 1) / a.size, label=name, linewidth=1.6)
    ax.axhline(0.99, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([t for t in (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200)
                   if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]])
    ax.text(ax.get_xlim()[1], 0.97, "p99 ", va="top", ha="right", fontsize=8, color="grey")
    ax.set_xlabel("end-to-end latency, batch 1 (ms, log scale)")
    ax.set_ylabel("fraction of frames")
    ax.set_title(f"Latency CDF per deployment target ({report['profile']})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def write_benchmark(report: Dict[str, Any], out_dir: str = "reports") -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"benchmark_{report['profile']}")
    plot_latency_cdf(report, stem + "_cdf.png")
    slim = {**report, "targets": {k: {kk: vv for kk, vv in v.items() if kk != "samples_total_ms"}
                                  for k, v in report["targets"].items()}}
    with open(stem + ".json", "w") as f:
        json.dump(slim, f, indent=2)
    with open(stem + ".md", "w") as f:
        f.write(benchmark_markdown(report))
    return stem + ".json"
