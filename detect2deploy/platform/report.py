"""report.py — SUMMARY.md for a platform run directory."""

from __future__ import annotations

import glob
import json
import os

from detect2deploy.platform.batch import batch_markdown
from detect2deploy.platform.serving import serving_markdown
from detect2deploy.platform.shadow import shadow_markdown


def _load(p):
    with open(p) as f:
        return json.load(f)


def build_summary(run_dir: str) -> str:
    lines = [f"# Platform run — `{os.path.basename(os.path.normpath(run_dir))}`", ""]
    env = os.path.join(run_dir, "env.txt")
    if os.path.exists(env):
        lines += ["Environment: [`env.txt`](env.txt)", ""]

    sv = os.path.join(run_dir, "serving.json")
    if os.path.exists(sv):
        lines += ["## 1. Serving with dynamic batching", "", serving_markdown(_load(sv))]
        if os.path.exists(os.path.join(run_dir, "serving.png")):
            lines += ["![latency vs throughput](serving.png)", ""]

    batches = [_load(p) for p in sorted(glob.glob(os.path.join(run_dir, "batch_*.json")))]
    if batches:
        lines += ["## 2. Offboard batch inference", "", batch_markdown(batches)]
        corpus = os.path.join(run_dir, "corpus.json")
        if os.path.exists(corpus):
            c = _load(corpus)
            lines += [f"Corpus: {c['frames']} frames, {c['shards']} Parquet shards, "
                      f"{c['avg_kb_per_frame']:.0f} KB/frame JPEG (q{c['jpeg_quality']}).", ""]
        lines += ["Ray per-operator stats: " + ", ".join(
            f"[`{os.path.basename(p)}`]({os.path.basename(p)})"
            for p in sorted(glob.glob(os.path.join(run_dir, "*_ray_stats.txt")))), ""]

    reg = os.path.join(run_dir, "registry.json")
    if os.path.exists(reg):
        r = _load(reg)
        lines += ["## 3. Registry and gated promotion", "",
                  "| version | file | target (gate evidence) | gate | aliases |", "|---|---|---|---|---|"]
        for v in r["versions"]:
            lines.append(f"| {v['version']} | `{v['file']}` | {v['target']} | {v['gate']} | "
                         f"{', '.join(v['aliases']) or '—'} |")
        lines += [""] + [f"- {e}" for e in r.get("events", [])] + [""]

    for p in sorted(glob.glob(os.path.join(run_dir, "shadow_*.json"))):
        lines += ["## 4. Shadow comparison", "", shadow_markdown(_load(p))]

    log = os.path.join(run_dir, "run.log")
    if os.path.exists(log):
        failed = [l.split("✗", 1)[1].strip() for l in open(log) if "✗" in l]
        lines += ["## Run status", "", "All steps succeeded." if not failed else
                  "Failed steps (see [`run.log`](run.log)):\n" + "\n".join(f"- {f}" for f in failed), ""]
    return "\n".join(lines)


def write_summary(run_dir: str) -> str:
    path = os.path.join(run_dir, "SUMMARY.md")
    with open(path, "w") as f:
        f.write(build_summary(run_dir))
    return path
