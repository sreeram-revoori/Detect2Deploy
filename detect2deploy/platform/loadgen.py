"""
loadgen.py
Closed-loop load generator for a Triton endpoint (PyTriton or containerised).

N concurrent clients each send one frame, wait for the answer, send the next
— the same model as perf_analyzer's concurrency mode. Reported per level:
client-side latency (p50/p99/max) and throughput, plus the server's own view
from Triton's statistics API: time in queue vs compute, and the batch sizes
the dynamic batcher actually formed.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np

from detect2deploy.platform.backend import OUTPUTS
from detect2deploy.platform.graph import FRAME_INPUT

GRPC_URL = "localhost:8001"
METRICS_URL = "http://localhost:8002/metrics"

# Token for Triton's protected endpoints (statistics, model repository, …).
# PyTriton >= 0.7 restricts them by default; the server reads the same env var.
TOKEN_ENV = "D2D_TRITON_TOKEN"
_GRPC_AUTH_HEADER = "triton-grpc-protocol-triton-access-token"


def auth_headers() -> Optional[Dict[str, str]]:
    token = os.environ.get(TOKEN_ENV)
    return {_GRPC_AUTH_HEADER: token} if token else None


def ensure_token() -> str:
    """Create a shared token for this run if none is set (inherited by child processes)."""
    if not os.environ.get(TOKEN_ENV):
        import secrets
        os.environ[TOKEN_ENV] = secrets.token_hex(16)
    return os.environ[TOKEN_ENV]


def _summ(ms: List[float]) -> Dict[str, float]:
    if not ms:
        return {"n": 0}
    a = np.asarray(ms)
    p50, p99 = np.percentile(a, [50, 99])
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(p50), "p90": float(np.percentile(a, 90)),
            "p99": float(p99), "max": float(a.max())}


def _stat_counts(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Triton statistics JSON (int64 fields arrive as strings) → numbers."""
    m = stats["model_stats"][0]
    inf = m.get("inference_stats", {})
    out = {"inference_count": int(m.get("inference_count", 0)),
           "execution_count": int(m.get("execution_count", 0))}
    for k in ("success", "queue", "compute_input", "compute_infer", "compute_output"):
        out[k + "_ns"] = int(inf.get(k, {}).get("ns", 0))
        out[k + "_count"] = int(inf.get(k, {}).get("count", 0))
    out["batch_hist"] = {int(b["batch_size"]): int(b.get("compute_infer", {}).get("count", 0))
                         for b in m.get("batch_stats", [])}
    return out


def server_view(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Difference of two statistics snapshots → per-request / per-execution averages."""
    d = {k: after[k] - before[k] for k in after if k != "batch_hist"}
    reqs, execs = max(1, d["success_count"]), max(1, d["execution_count"])
    hist = {b: after["batch_hist"].get(b, 0) - before["batch_hist"].get(b, 0) for b in after["batch_hist"]}
    return {"requests": d["success_count"], "executions": d["execution_count"],
            "avg_batch": d["inference_count"] / execs,
            "queue_us_per_request": d["queue_ns"] / reqs / 1e3,
            "compute_infer_us_per_exec": d["compute_infer_ns"] / execs / 1e3,
            "compute_io_us_per_exec": (d["compute_input_ns"] + d["compute_output_ns"]) / execs / 1e3,
            "server_us_per_request": d["success_ns"] / reqs / 1e3,
            "batch_hist": {b: n for b, n in sorted(hist.items()) if n}}


async def closed_loop(model: str, frames: np.ndarray, concurrency: int, duration_s: float = 10.0,
                      warmup_s: float = 2.0, url: str = GRPC_URL) -> Dict[str, Any]:
    import tritonclient.grpc.aio as grpcclient

    client = grpcclient.InferenceServerClient(url=url)
    outputs = [grpcclient.InferRequestedOutput(n) for n in OUTPUTS]
    lat: List[float] = []
    t_start = time.perf_counter()
    t_measure, t_end = t_start + warmup_s, t_start + warmup_s + duration_s
    before: Optional[Dict[str, Any]] = None

    async def worker(i: int) -> None:
        k = i
        while True:
            now = time.perf_counter()
            if now >= t_end:
                return
            inp = grpcclient.InferInput(FRAME_INPUT, [1, *frames.shape[1:]], "UINT8")
            inp.set_data_from_numpy(frames[k % len(frames)][None])
            t0 = time.perf_counter()
            await client.infer(model, [inp], outputs=outputs)
            t1 = time.perf_counter()
            if t0 >= t_measure:
                lat.append((t1 - t0) * 1e3)
            k += concurrency

    async def snapshot_after_warmup() -> None:
        nonlocal before
        await asyncio.sleep(warmup_s)
        before = _stat_counts(await client.get_inference_statistics(model_name=model, as_json=True,
                                                                    headers=auth_headers()))

    await asyncio.gather(snapshot_after_warmup(), *(worker(i) for i in range(concurrency)))
    after = _stat_counts(await client.get_inference_statistics(model_name=model, as_json=True,
                                                               headers=auth_headers()))
    await client.close()
    return {"concurrency": concurrency, "duration_s": duration_s, "throughput_fps": len(lat) / duration_s,
            "latency_ms": _summ(lat), "server": server_view(before, after)}


def scrape_metrics(url: str = METRICS_URL) -> Dict[str, float]:
    """Triton's Prometheus endpoint → {metric: value summed over labels}."""
    try:
        text = urllib.request.urlopen(url, timeout=5).read().decode()
    except Exception:
        return {}
    vals: Dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$", line)
        if m and m.group(3) != "NaN":
            vals[m.group(1)] = vals.get(m.group(1), 0.0) + float(m.group(3))
    return vals


def wait_ready(model: str, timeout_s: float = 900, url: str = GRPC_URL) -> bool:
    """Block until the server reports the model ready (TensorRT engine builds can take minutes)."""
    import tritonclient.grpc as grpcclient
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        try:
            c = grpcclient.InferenceServerClient(url=url)
            if c.is_server_ready() and c.is_model_ready(model):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False
