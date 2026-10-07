"""Detect2Deploy — AV Simulation & ML Evaluation Framework"""

# Public names are resolved lazily (PEP 562) so that importing any submodule —
# e.g. the serving process, which runs in a lean env — doesn't pull in
# matplotlib / MLflow / Ray through this package's __init__.
_EXPORTS = {
    "generate_scenario": "detect2deploy.scenario_generator",
    "generate_batch": "detect2deploy.scenario_generator",
    "Scenario": "detect2deploy.scenario_generator",
    "BBox": "detect2deploy.scenario_generator",
    "MockDetector": "detect2deploy.perception_model",
    "TimedDetector": "detect2deploy.perception_model",
    "Evaluator": "detect2deploy.metrics",
    "EvalResult": "detect2deploy.metrics",
    "ParallelEvaluator": "detect2deploy.ray_worker",
    "log_to_mlflow": "detect2deploy.tracker",
    "annotate_frame": "detect2deploy.tracker",
}

__version__ = "0.3.0"
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'detect2deploy' has no attribute {name!r}")
