"""
shadow.py
Shadow deployment check: mirror the same frames to the production model and
a candidate, compare outputs and latency, decide promote / hold.

Criteria (fixed up front, not tuned to a result):
    agreement   candidate reproduces >= 99% of production's detections
                (same class, IoU >= 0.5) and adds <= 1% new ones
    latency     candidate p99 <= 1.25 x production p99

Shadow agreement is an aggregate — it complements the per-class release gate,
it doesn't replace it.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Protocol

import numpy as np

from nurosim.deploy.parity import detection_agreement
from nurosim.platform.backend import OUTPUTS, ServingModel, unpack
from nurosim.platform.graph import FRAME_INPUT

MIN_AGREEMENT = 0.99
MAX_P99_RATIO = 1.25


class Endpoint(Protocol):
    name: str
    def infer(self, frames: np.ndarray) -> Dict[str, np.ndarray]: ...


class LocalEndpoint:
    def __init__(self, name: str, model: ServingModel):
        self.name, self.model = name, model

    def infer(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        return self.model.infer(frames)


class TritonEndpoint:
    def __init__(self, name: str, url: str = "localhost:8001"):
        import tritonclient.grpc as grpcclient
        self.name, self._g = name, grpcclient
        self.client = grpcclient.InferenceServerClient(url=url)

    def infer(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        inp = self._g.InferInput(FRAME_INPUT, list(frames.shape), "UINT8")
        inp.set_data_from_numpy(np.ascontiguousarray(frames, dtype=np.uint8))
        res = self.client.infer(self.name, [inp], outputs=[self._g.InferRequestedOutput(n) for n in OUTPUTS])
        return decode_response(res)


def decode_response(res) -> Dict[str, np.ndarray]:
    """InferResult → arrays, with a precise error if the server sent an output
    without data (tritonclient's own error is just a failed reshape)."""
    try:
        return {n: res.as_numpy(n) for n in OUTPUTS}
    except ValueError as e:
        r = res.get_response()
        desc = [(o.name, list(o.shape), o.datatype) for o in r.outputs]
        raise RuntimeError(f"malformed response: outputs={desc}, raw_output_contents="
                           f"{[len(b) for b in r.raw_output_contents]} ({e})") from e


def _p(a: List[float], q: float) -> float:
    return float(np.percentile(a, q)) if a else float("nan")


def shadow_compare(production: Endpoint, candidate: Endpoint, frames: np.ndarray) -> Dict[str, Any]:
    prod_dets, cand_dets, lat_p, lat_c = [], [], [], []
    for i, f in enumerate(frames):
        batch = f[None]
        # alternate the order so neither model systematically sees a warmer cache
        order = (production, candidate) if i % 2 == 0 else (candidate, production)
        for ep in order:
            t0 = time.perf_counter()
            out = ep.infer(batch)
            ms = (time.perf_counter() - t0) * 1e3
            if ep is production:
                prod_dets += unpack(out)
                lat_p.append(ms)
            else:
                cand_dets += unpack(out)
                lat_c.append(ms)

    agree = detection_agreement(prod_dets, cand_dets)
    p99_ratio = _p(lat_c, 99) / _p(lat_p, 99)
    ok_agree = agree["ref_recall"] >= MIN_AGREEMENT and (1 - agree["target_precision"]) <= 1 - MIN_AGREEMENT
    ok_lat = p99_ratio <= MAX_P99_RATIO
    return {
        "production": production.name, "candidate": candidate.name, "frames": len(frames),
        "agreement": agree,
        "latency_ms": {"production": {"p50": _p(lat_p, 50), "p99": _p(lat_p, 99)},
                       "candidate": {"p50": _p(lat_c, 50), "p99": _p(lat_c, 99)}, "p99_ratio": p99_ratio},
        "criteria": {"min_agreement": MIN_AGREEMENT, "max_p99_ratio": MAX_P99_RATIO},
        "decision": "promote" if ok_agree and ok_lat else "hold",
        "reasons": ([] if ok_agree else [f"agreement {agree['ref_recall']:.4f} / extra "
                                         f"{1 - agree['target_precision']:.4f} vs {MIN_AGREEMENT}"])
                   + ([] if ok_lat else [f"p99 ratio {p99_ratio:.2f} > {MAX_P99_RATIO}"]),
    }


def shadow_markdown(r: Dict[str, Any]) -> str:
    a, l = r["agreement"], r["latency_ms"]
    return "\n".join([
        f"### Shadow: `{r['candidate']}` vs production `{r['production']}` ({r['frames']} frames)", "",
        "| production dets reproduced | extra dets | match IoU | prod p50 / p99 ms | cand p50 / p99 ms | decision |",
        "|---|---|---|---|---|---|",
        f"| {a['ref_recall']:.4f} | {1 - a['target_precision']:.4f} | {a['mean_match_iou']:.3f} | "
        f"{l['production']['p50']:.2f} / {l['production']['p99']:.2f} | "
        f"{l['candidate']['p50']:.2f} / {l['candidate']['p99']:.2f} | **{r['decision']}** |",
        "", ("Reasons: " + "; ".join(r["reasons"])) if r["reasons"] else "Both criteria met.", ""])


def write_shadow(r: Dict[str, Any], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"shadow_{r['candidate']}.json")
    with open(path, "w") as f:
        json.dump(r, f, indent=2)
    with open(path[:-5] + ".md", "w") as f:
        f.write(shadow_markdown(r))
    return path
