"""
config.py
Loads configs/deploy.yaml: model variants, eval/benchmark settings, the
accuracy budget, and per-hardware profiles of deployment targets.

A *target* is (model variant, execution provider, provider options), e.g.
"int8 model on the CPU EP" or "fp32 model on TensorRT with FP16 enabled".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CONFIG = "configs/deploy.yaml"


@dataclass
class Target:
    name: str
    variant: str                      # key into config["models"]
    model_path: str
    provider: str = "cpu"
    provider_options: Dict[str, Any] = field(default_factory=dict)
    static_batch: Optional[int] = None
    p99_budget_ms: Optional[float] = None   # end-to-end, batch 1; None = not gated
    expect_fail: bool = False               # negative control: accuracy gate MUST trip

    def detector_kwargs(self) -> Dict[str, Any]:
        return dict(model_path=self.model_path, provider=self.provider,
                    provider_options=self.provider_options or None,
                    static_batch=self.static_batch)


@dataclass
class Profile:
    name: str
    description: str
    reference: str
    targets: List[Target]
    intra_op_threads: Optional[int] = None  # benchmark thread pool; None = global setting

    def target(self, name: str) -> Target:
        for t in self.targets:
            if t.name == name:
                return t
        raise KeyError(f"target '{name}' not in profile '{self.name}'")


def load_config(path: str = DEFAULT_CONFIG) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_profile(cfg: Dict[str, Any], name: str,
                 only: Optional[List[str]] = None) -> Profile:
    if name not in cfg["profiles"]:
        raise KeyError(f"profile '{name}' not found; have {list(cfg['profiles'])}")
    p = cfg["profiles"][name]
    targets = []
    for tname, t in p["targets"].items():
        if only and tname not in only and tname != p["reference"]:
            continue
        targets.append(Target(
            name=tname,
            variant=t["model"],
            model_path=cfg["models"][t["model"]],
            provider=t.get("provider", "cpu"),
            provider_options=t.get("provider_options") or {},
            static_batch=t.get("static_batch"),
            p99_budget_ms=t.get("p99_budget_ms"),
            expect_fail=bool(t.get("expect_fail", False)),
        ))
    return Profile(name=name, description=p.get("description", ""),
                   reference=p["reference"], targets=targets,
                   intra_op_threads=p.get("intra_op_threads"))
