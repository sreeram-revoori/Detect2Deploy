#!/usr/bin/env python3
"""
nsys_analyze.py — summarise an Nsight Systems SQLite export of d2d_bench.

    nsys export --type sqlite -o run.sqlite run.nsys-rep
    python scripts/gpu/nsys_analyze.py run.sqlite

Reads the tables directly (no dependency on `nsys stats` report names, which
change between Nsight versions) and reports, per steady-state iteration:
kernel count, GPU busy time vs idle gaps between kernels, the kernels that
dominate (incl. layout-conversion copies), CPU-side launch cost, and — for
CUDA-graph runs — the graph's GPU duration and per-launch CPU cost.
"""

import sqlite3
import statistics as st
import sys
from collections import defaultdict


def analyze(path: str) -> str:
    c = sqlite3.connect(path)
    strings = dict(c.execute("select id, value from StringIds"))
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    out = [f"# {path}", ""]

    nvtx = c.execute("select start, end, text, textId from NVTX_EVENTS where end is not null").fetchall()
    gpu_ranges = sorted((s, e) for s, e, t, tid in nvtx if (t or strings.get(tid)) == "gpu")
    steady = gpu_ranges[len(gpu_ranges) // 4:]            # skip warm-up iterations
    out.append(f"iterations (NVTX 'gpu' ranges): {len(gpu_ranges)}, steady-state analysed: {len(steady)}")

    kernels = c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    graph = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables and \
        c.execute("select count(*) from CUPTI_ACTIVITY_KIND_GRAPH_TRACE").fetchone()[0] > 0

    if graph:
        g = sorted((e - s) / 1e6 for s, e in c.execute("select start, end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE"))
        out.append(f"CUDA graph launches: {len(g)}; graph GPU duration p50 {st.median(g):.3f} ms, "
                   f"p99 {g[max(0, int(0.99 * len(g)) - 1)]:.3f} ms (kernels inside graphs are not traced individually)")
    elif steady:
        busy, span, n = [], [], []
        i = 0
        for s, e in steady:
            while i < len(kernels) and kernels[i][0] < s:
                i += 1
            j = i
            ks = []
            while j < len(kernels) and kernels[j][1] <= e:
                ks.append(kernels[j])
                j += 1
            if ks:
                busy.append(sum(b - a for a, b, _ in ks) / 1e6)
                span.append((ks[-1][1] - ks[0][0]) / 1e6)
                n.append(len(ks))
        gaps = [sp - b for sp, b in zip(span, busy)]
        out.append(f"kernels per iteration: {st.median(n):.0f}; GPU busy {st.median(busy):.3f} ms of a "
                   f"{st.median(span):.3f} ms span → idle gaps {st.median(gaps):.3f} ms "
                   f"({100 * st.median(gaps) / st.median(span):.0f}%)")

        tot, cnt = defaultdict(float), defaultdict(int)
        for a, b, name in kernels:
            k = strings.get(name, str(name))
            tot[k] += (b - a) / 1e6
            cnt[k] += 1
        total = sum(tot.values())
        iters = max(1, len(gpu_ranges))
        out += ["", "top kernels by GPU time:", "| share | avg µs | per iter | kernel |", "|---|---|---|---|"]
        for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:10]:
            out.append(f"| {100 * v / total:.1f}% | {1000 * v / cnt[k]:.1f} | {cnt[k] / iters:.0f} | `{k[:70]}` |")

    api = defaultdict(list)
    for s, e, name in c.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        api[strings.get(name, str(name))].append((e - s) / 1e6)
    launches = sum(len(v) for k, v in api.items() if "LaunchKernel" in k)
    launch_ms = sum(sum(v) for k, v in api.items() if "LaunchKernel" in k)
    iters = max(1, len(gpu_ranges))
    out += ["", f"CPU launch cost per iteration: {launches / iters:.0f} kernel launches, "
                f"{launch_ms / iters:.3f} ms"]
    if "cudaGraphLaunch_v10000" in api:
        gl = api["cudaGraphLaunch_v10000"]
        out.append(f"cudaGraphLaunch per iteration: {st.median(gl):.3f} ms (one call)")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(analyze(p))
