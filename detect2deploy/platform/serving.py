"""
serving.py
Dynamic-batching sweep: how much latency does batching cost, and how much
throughput does it buy, as load rises?

For each batching config the server is (re)started — PyTriton in a
subprocess, or a containerised Triton reloaded through its model-control API
— then the closed-loop load generator runs at each concurrency level.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from detect2deploy.deploy.seeds import BENCH_SEED
from detect2deploy.platform.loadgen import auth_headers, closed_loop, ensure_token, scrape_metrics, wait_ready
from detect2deploy.scenario_generator import generate_scenario

logger = logging.getLogger(__name__)

# (name, max_batch_size, max_queue_delay_us)
CONFIGS: List[Tuple[str, int, int]] = [
    ("no_batching", 1, 0),
    ("dynamic_0us", 8, 0),
    ("dynamic_500us", 8, 500),
    ("dynamic_2000us", 8, 2000),
]
CONCURRENCY = [1, 2, 4, 8, 16]


def bench_frames(n: int = 64) -> np.ndarray:
    return np.stack([generate_scenario(BENCH_SEED + i, seed=BENCH_SEED + i).frame for i in range(n)])


class PyTritonProcess:
    """`python -m detect2deploy.platform serve …` as a managed subprocess."""

    def __init__(self, models: Dict[str, str], provider: str, max_batch: int, queue_delay_us: int,
                 log_path: str, uri: Optional[str] = None):
        # PyTriton needs numpy < 2 → it usually runs from its own environment
        # (scripts/platform/make_serve_env.sh), named by D2D_SERVE_PYTHON.
        python = os.environ.get("D2D_SERVE_PYTHON", sys.executable)
        ensure_token()                       # server and our clients share it via env
        cmd = [python, "-m", "detect2deploy.platform", "serve", "--provider", provider,
               "--max-batch", str(max_batch), "--queue-delay-us", str(queue_delay_us)]
        if uri:
            cmd += ["--uri", uri]
        for name, path in models.items():
            cmd += ["--model", f"{name}={path}"]
        self.log = open(log_path, "a")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        self.log.close()
        time.sleep(2)                       # let the ports free up


def triton_model_config(model: str, max_batch: int, queue_delay_us: int, size: int = 640,
                        max_det: int = 100) -> Dict[str, Any]:
    """Full Triton config for the TensorRT plan (uint8 frames in, EfficientNMS outputs)."""
    cfg = {
        "name": model, "platform": "tensorrt_plan", "max_batch_size": max_batch,
        "input": [{"name": "frames", "data_type": "TYPE_UINT8", "dims": [size, size, 3]}],
        "output": [{"name": "num_dets", "data_type": "TYPE_INT32", "dims": [1]},
                   {"name": "det_boxes", "data_type": "TYPE_FP32", "dims": [max_det, 4]},
                   {"name": "det_scores", "data_type": "TYPE_FP32", "dims": [max_det]},
                   {"name": "det_classes", "data_type": "TYPE_INT32", "dims": [max_det]}],
        "instance_group": [{"kind": "KIND_GPU", "count": 1}],
    }
    if max_batch > 1:
        cfg["dynamic_batching"] = {"max_queue_delay_microseconds": queue_delay_us}
    return cfg


def triton_load_config(model: str, max_batch: int, queue_delay_us: int, url: str) -> None:
    """Containerised Triton (explicit model control): reload with a full config override."""
    import tritonclient.grpc as grpcclient
    c = grpcclient.InferenceServerClient(url=url)
    c.load_model(model, config=json.dumps(triton_model_config(model, max_batch, queue_delay_us)),
                 headers=auth_headers())


def run_sweep(model_path: Optional[str], out_dir: str, provider: str = "tensorrt",
              backend: str = "pytriton", model: str = "d2d_det", duration_s: float = 10.0,
              concurrency: List[int] = CONCURRENCY, configs=CONFIGS,
              url: str = "localhost:8001") -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    frames = bench_frames()
    report: Dict[str, Any] = {"backend": backend, "provider": provider, "model": model_path,
                              "duration_s": duration_s, "configs": []}
    for name, max_batch, delay in configs:
        logger.info("serving config %s (max_batch=%d, queue_delay=%dus)", name, max_batch, delay)
        server = None
        if backend == "pytriton":
            server = PyTritonProcess({model: model_path}, provider, max_batch, delay,
                                     os.path.join(out_dir, "server.log"))
        else:
            triton_load_config(model, max_batch, delay, url)
        try:
            if not wait_ready(model, url=url):
                raise RuntimeError(f"server not ready for config {name} (see server.log)")
            m0 = scrape_metrics()
            levels = []
            for c in concurrency:
                r = asyncio.run(closed_loop(model, frames, c, duration_s=duration_s, url=url))
                logger.info("  c=%-3d %7.1f fps  p50 %6.2f ms  p99 %6.2f ms  avg batch %.2f  queue %.0f us",
                            c, r["throughput_fps"], r["latency_ms"]["p50"], r["latency_ms"]["p99"],
                            r["server"]["avg_batch"], r["server"]["queue_us_per_request"])
                levels.append(r)
            m1 = scrape_metrics()
            gpu = {k: m1.get(k) for k in ("nv_gpu_utilization", "nv_gpu_memory_used_bytes",
                                         "nv_gpu_power_usage") if k in m1}
            report["configs"].append({"name": name, "max_batch": max_batch, "queue_delay_us": delay,
                                      "levels": levels, "gpu_metrics_at_end": gpu,
                                      "metrics_delta": {k: m1[k] - m0.get(k, 0.0) for k in m1
                                                        if k.startswith("nv_inference")}})
        finally:
            if server:
                server.stop()
    with open(os.path.join(out_dir, "serving.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def serving_markdown(rep: Dict[str, Any]) -> str:
    lines = [f"### Serving — dynamic batching sweep ({rep['backend']}, {rep['provider']} EP, "
             f"{rep['duration_s']:.0f} s per level)", "",
             "| config | concurrency | frames/s | p50 ms | p99 ms | avg batch | queue µs/req | infer µs/exec |",
             "|---|---|---|---|---|---|---|---|"]
    for cfg in rep["configs"]:
        for lv in cfg["levels"]:
            s, l = lv["server"], lv["latency_ms"]
            lines.append(f"| {cfg['name']} | {lv['concurrency']} | {lv['throughput_fps']:.0f} | "
                         f"{l['p50']:.2f} | {l['p99']:.2f} | {s['avg_batch']:.2f} | "
                         f"{s['queue_us_per_request']:.0f} | {s['compute_infer_us_per_exec']:.0f} |")
    return "\n".join(lines) + "\n"


def plot_serving(rep: Dict[str, Any], path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for cfg in rep["configs"]:
        xs = [lv["throughput_fps"] for lv in cfg["levels"]]
        ys = [lv["latency_ms"]["p99"] for lv in cfg["levels"]]
        ax.plot(xs, ys, marker="o", label=cfg["name"])
        for lv, x, y in zip(cfg["levels"], xs, ys):
            ax.annotate(str(lv["concurrency"]), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("throughput (frames/s)")
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("Latency vs throughput per batching config (labels = concurrency)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
