"""
registry.py
MLflow model registry with gated promotion.

    register  log the artefact + its gate verdict and accuracy as a new version
    promote   point an alias ("production") at a version — refused unless the
              release gate passed *for that exact file* (the gate JSON's
              evidence hash must equal the artefact's hash)
    resolve   alias → local model file, what a server loads at startup

The evidence binding matters in practice: the T4 run showed INT8 calibration
isn't bit-identical across machines, so "INT8 passed on x86" says nothing
about an INT8 file rebuilt on ARM.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional

from detect2deploy.deploy.export import sha256_file

MODEL_NAME = "d2d_det"
DEFAULT_URI = "sqlite:///mlruns/registry.db"


class PromotionBlocked(RuntimeError):
    pass


def _client(uri: str):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from mlflow import MlflowClient
    if uri.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(uri[len("sqlite:///"):]) or ".", exist_ok=True)
    mlflow.set_tracking_uri(uri)
    return mlflow, MlflowClient()


def gate_verdict(gate_json: str, target: str, model_path: str) -> Dict[str, Any]:
    """What the gate evidence says about this file on this target."""
    with open(gate_json) as f:
        g = json.load(f)
    t = g["targets"].get(target)
    sha = sha256_file(model_path)
    if t is None:
        return {"gate": "no-evidence", "reason": f"target '{target}' not in {gate_json}", "sha256": sha}
    if t.get("evidence_sha256") != sha:
        return {"gate": "no-evidence", "sha256": sha,
                "reason": f"gate measured {t.get('evidence_sha256')}, this file is {sha}"}
    failed = [c for c in t["checks"] if not c["passed"]]
    return {"gate": "pass" if t["passed"] else "fail", "sha256": sha, "profile": g["profile"],
            "reason": "; ".join(f"{c['check']} {c['value']} (budget {c['limit']})" for c in failed)}


def register(model_path: str, target: str, gate_json: str, parity_json: Optional[str] = None,
             uri: str = DEFAULT_URI, name: str = MODEL_NAME) -> Dict[str, Any]:
    mlflow, client = _client(uri)
    verdict = gate_verdict(gate_json, target, model_path)
    metrics: Dict[str, float] = {}
    if parity_json and os.path.exists(parity_json):
        with open(parity_json) as f:
            m = json.load(f)["targets"].get(target, {}).get("metrics", {})
        metrics = {"map50": m.get("map50"), "map50_95": m.get("map50_95"),
                   **{f"ap50_{k}": v for k, v in m.get("per_class_ap50", {}).items()}}
        metrics = {k: float(v) for k, v in metrics.items() if v is not None}

    with mlflow.start_run(run_name=f"register-{target}") as run:
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_params({"target": target, "file": os.path.basename(model_path),
                           "sha256": verdict["sha256"]})
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.set_tags({"gate": verdict["gate"], "gate_reason": verdict["reason"][:500]})

    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name, description="Detect2Deploy BEV detector")
    mv = client.create_model_version(
        name, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id,
        tags={"gate": verdict["gate"], "target": target, "sha256": verdict["sha256"],
              "file": os.path.basename(model_path)})
    return {"name": name, "version": mv.version, "target": target, **verdict}


def promote(version: str, alias: str = "production", uri: str = DEFAULT_URI,
            name: str = MODEL_NAME) -> Dict[str, Any]:
    _, client = _client(uri)
    mv = client.get_model_version(name, str(version))
    gate = mv.tags.get("gate")
    if gate != "pass":
        raise PromotionBlocked(f"{name} v{version} ({mv.tags.get('file')} on {mv.tags.get('target')}): "
                               f"gate={gate} — not promoting to '{alias}'")
    client.set_registered_model_alias(name, alias, str(version))
    return {"name": name, "alias": alias, "version": str(version), "target": mv.tags.get("target")}


def resolve(alias: str = "production", uri: str = DEFAULT_URI, name: str = MODEL_NAME,
            dst: Optional[str] = None) -> str:
    mlflow, client = _client(uri)
    mv = client.get_model_version_by_alias(name, alias)
    local = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=dst)
    files = glob.glob(os.path.join(local, "*.onnx")) if os.path.isdir(local) else [local]
    if not files:
        raise FileNotFoundError(f"no model file under {local}")
    return files[0]


def list_versions(uri: str = DEFAULT_URI, name: str = MODEL_NAME) -> List[Dict[str, Any]]:
    _, client = _client(uri)
    aliases: Dict[str, List[str]] = {}
    try:
        for a, v in client.get_registered_model(name).aliases.items():
            aliases.setdefault(str(v), []).append(a)
    except Exception:
        return []
    out = []
    for mv in client.search_model_versions(f"name='{name}'"):
        out.append({"version": mv.version, "target": mv.tags.get("target"), "file": mv.tags.get("file"),
                    "gate": mv.tags.get("gate"), "sha256": mv.tags.get("sha256"),
                    "aliases": aliases.get(str(mv.version), [])})
    return sorted(out, key=lambda r: int(r["version"]))
