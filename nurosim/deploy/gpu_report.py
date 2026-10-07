"""
gpu_report.py
Collects everything scripts/gpu/run_all.sh produced in one run directory into
SUMMARY.md: environment, C++ benchmark matrix, batch sweep, multi-model
contention, TensorRT accuracy parity, the ONNX Runtime GPU profile, profiler
outputs, and which steps failed.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List


def _load(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _bench_table(rows: List[Dict[str, Any]]) -> List[str]:
    out = ["| config | pre | post | CUDA graph | e2e p50 | e2e p99 | max | jitter | "
           "host pre | H2D | GPU pre | infer | D2H | post | fps |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e, s = r["e2e"], r["stages"]
        g = lambda k: f"{s[k]['p50']:.3f}" if k in s else "—"   # noqa: E731
        out.append(f"| {r['label']} | {r['pre']} | {r['post']} | {'✅' if r['cuda_graph'] else '—'} | "
                   f"{e['p50']:.3f} | {e['p99']:.3f} | {e['max']:.3f} | {e['jitter_p99_p50']:.3f} | "
                   f"{g('host_pre')} | {g('h2d')} | {g('gpu_pre')} | {g('infer')} | {g('d2h')} | "
                   f"{g('post')} | {r['throughput_fps']:.0f} |")
    return out


def build_summary(run_dir: str) -> str:
    lines = [f"# GPU run — `{os.path.basename(os.path.normpath(run_dir))}`", ""]

    env = os.path.join(run_dir, "env.txt")
    benches = [_load(p) for p in sorted(glob.glob(os.path.join(run_dir, "bench_[0-9]*.json")))]
    if benches:
        d = benches[0]["device"]
        lines += [f"**{d['name']}** (sm_{d['compute_capability'].replace('.', '')}, {d['sms']} SMs"
                  f"{', integrated' if d['integrated'] else ''}) · TensorRT {d['tensorrt']} · "
                  f"CUDA runtime {d['cuda_runtime']} / driver {d['cuda_driver']}", ""]
    if os.path.exists(env):
        lines += [f"Full environment: [`env.txt`](env.txt)", ""]

    if benches:
        lines += ["## C++ pipeline latency (batch 1, ms)", "",
                  "Each row changes one thing from the row above. *host pre* is CPU letterbox "
                  "(cpu) or the memcpy into pinned staging (gpu); GPU stages are CUDA-event timed; "
                  "with a CUDA graph the GPU side is one launch, so only end-to-end is split.", ""]
        lines += _bench_table(benches) + [""]

    sweep = [_load(p) for p in sorted(glob.glob(os.path.join(run_dir, "sweep_bs*.json")),
                                      key=lambda p: int(p.split("bs")[-1].split(".")[0]))]
    if sweep:
        lines += ["## Batch sweep (offboard / batch-inference throughput)", "",
                  "| batch | e2e p50 (ms) | e2e p99 (ms) | frames/s |", "|---|---|---|---|"]
        for r in sweep:
            lines.append(f"| {r['batch']} | {r['e2e']['p50']:.2f} | {r['e2e']['p99']:.2f} | "
                         f"{r['throughput_fps']:.0f} |")
        lines.append("")

    ms = os.path.join(run_dir, "multistream.json")
    if os.path.exists(ms):
        m = _load(ms)
        period = 1000.0 / m["rate_a_hz"]
        lines += ["## Two models on one GPU", "",
                  f"A = latency-critical detector at {m['rate_a_hz']:.0f} Hz (deadline {period:.1f} ms), "
                  f"batch 1 · B = background model saturating the GPU at batch {m['batch_b']} · "
                  f"{m['duration_s']:.0f} s per phase.", "",
                  "| phase | A p50 | A p99 | A max | A deadline misses | B frames/s |",
                  "|---|---|---|---|---|---|"]
        for p in m["phases"]:
            a = p["a"]
            miss = f"{p['a_deadline_misses']} / {p['a_frames']}" if p["a_frames"] else "—"
            lines.append(f"| {p['name']} | {a['p50']:.2f} | {a['p99']:.2f} | {a['max']:.2f} | {miss} | "
                         f"{p['b_fps']:.0f} |" if p["a_frames"] else
                         f"| {p['name']} | — | — | — | — | {p['b_fps']:.0f} |")
        lines.append("")

    pc = os.path.join(run_dir, "parity_cpp.md")
    if os.path.exists(pc):
        lines += ["## Accuracy parity of the TensorRT engines", ""] + open(pc).read().splitlines()[2:] + [""]

    ort = [f for f in ("parity_nvidia-trt.md", "benchmark_nvidia-trt.md", "gate_nvidia-trt.md")
           if os.path.exists(os.path.join(run_dir, f))]
    if ort:
        lines += ["## ONNX Runtime CUDA / TensorRT EPs (Python harness, profile `nvidia-trt`)", "",
                  " · ".join(f"[{f}]({f})" for f in ort), ""]

    prof = sorted(glob.glob(os.path.join(run_dir, "*.nsys-rep")))
    if prof:
        lines += ["## Nsight Systems", "",
                  "Open in Nsight Systems: " + ", ".join(f"`{os.path.basename(p)}`" for p in prof)
                  + ". Kernel / NVTX summaries: "
                  + ", ".join(f"[`{os.path.basename(p)}`]({os.path.basename(p)})"
                              for p in sorted(glob.glob(os.path.join(run_dir, "nsys_*.txt")))), ""]

    log = os.path.join(run_dir, "run.log")
    if os.path.exists(log):
        failed = [l.split("✗", 1)[1].strip() for l in open(log) if "✗" in l]
        lines += ["## Run status", "",
                  "All steps succeeded." if not failed else
                  "Failed steps (details in [`run.log`](run.log)):\n" + "\n".join(f"- {f}" for f in failed), ""]
    return "\n".join(lines)


def write_summary(run_dir: str) -> str:
    path = os.path.join(run_dir, "SUMMARY.md")
    with open(path, "w") as f:
        f.write(build_summary(run_dir))
    return path
