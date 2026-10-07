"""
gate.py
CI release gate: a model change ships only if every deployment target in the
profile stays inside its accuracy and latency budgets.

Checks per target
  * provider   – the requested execution provider actually ran (no silent
                 CPU fallback)
  * freshness  – the reports were produced from the model files on disk now
  * accuracy   – mAP@0.5 drop vs the reference, worst per-weather drop and
                 worst per-class drop, against configs/deploy.yaml budgets
  * latency    – end-to-end p99 at batch 1 vs the target's p99_budget_ms

Targets marked `expect_fail` are negative controls — known-bad variants kept
in the profile to prove the gate still catches a regression. For them the
accuracy checks pass only if at least one budget is exceeded.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nurosim.deploy.config import Profile
from nurosim.deploy.export import sha256_file


@dataclass
class Check:
    target: str
    check: str
    value: str
    limit: str
    passed: bool


def _load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_gate(cfg: Dict[str, Any], profile: Profile,
             parity: Optional[Dict[str, Any]],
             bench: Optional[Dict[str, Any]]) -> List[Check]:
    ab = cfg["accuracy_budget"]
    checks: List[Check] = []

    for t in profile.targets:
        p = (parity or {}).get("targets", {}).get(t.name)
        b = (bench or {}).get("targets", {}).get(t.name)
        if p is None and b is None:
            checks.append(Check(t.name, "measured", "missing", "parity + benchmark", False))
            continue

        active = all(r["provider_active"] for r in (p, b) if r)
        actual = (p or b)["active_providers"][0]
        checks.append(Check(t.name, "provider", actual, t.provider, active))

        sha = sha256_file(t.model_path) if os.path.exists(t.model_path) else "missing"
        stale = [n for n, r in (("parity", p), ("bench", b)) if r and r.get("model_sha256") != sha]
        checks.append(Check(t.name, "freshness", "stale: " + ", ".join(stale) if stale else sha,
                            "reports match model on disk", not stale))

        if p and t.name != profile.reference:
            d = p["delta_vs_reference"]
            wx, wxd = max(d["weather_map50_drop"].items(), key=lambda kv: kv[1])
            cl, cld = max(d["per_class_ap50_drop"].items(), key=lambda kv: kv[1])
            acc = [
                Check(t.name, "mAP@0.5 drop", f"{d['map50_drop']:+.4f}",
                      f"≤ {ab['max_map50_drop']}", d["map50_drop"] <= ab["max_map50_drop"]),
                Check(t.name, "worst weather drop", f"{wx} {wxd:+.4f}",
                      f"≤ {ab['max_map50_drop_per_weather']}",
                      wxd <= ab["max_map50_drop_per_weather"]),
                Check(t.name, "worst class drop", f"{cl} {cld:+.4f}",
                      f"≤ {ab['max_ap50_drop_per_class']}", cld <= ab["max_ap50_drop_per_class"]),
            ]
            if t.expect_fail:
                caught = not all(c.passed for c in acc)
                checks.append(Check(t.name, "negative control caught",
                                    f"mAP@0.5 drop {d['map50_drop']:+.4f}",
                                    "must exceed accuracy budget", caught))
            else:
                checks += acc
        elif t.name != profile.reference:
            checks.append(Check(t.name, "accuracy", "not measured", "parity report", False))

        if t.p99_budget_ms is not None and not t.expect_fail:
            if b:
                p99 = b["e2e_batch1"]["total"]["p99"]
                checks.append(Check(t.name, "e2e p99 latency", f"{p99:.2f} ms",
                                    f"≤ {t.p99_budget_ms} ms", p99 <= t.p99_budget_ms))
            else:
                checks.append(Check(t.name, "e2e p99 latency", "not measured",
                                    f"≤ {t.p99_budget_ms} ms", False))
    return checks


def gate_markdown(profile: Profile, checks: List[Check]) -> str:
    ok = all(c.passed for c in checks)
    lines = [f"### Release gate — profile `{profile.name}`: "
             f"{'✅ PASS' if ok else '❌ FAIL'}", "",
             "| target | check | value | budget | |", "|---|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c.target} | {c.check} | {c.value} | {c.limit} | "
                     f"{'✅' if c.passed else '❌'} |")
    return "\n".join(lines) + "\n"


def load_reports(profile_name: str, report_dir: str = "reports"):
    return (_load(os.path.join(report_dir, f"parity_{profile_name}.json")),
            _load(os.path.join(report_dir, f"benchmark_{profile_name}.json")))
