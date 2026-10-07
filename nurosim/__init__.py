"""NuroSim-Lite — AV Simulation & ML Evaluation Framework"""

# Public names are resolved lazily (PEP 562) so that importing any submodule —
# e.g. the serving process, which runs in a lean env — doesn't pull in
# matplotlib / MLflow / Ray through this package's __init__.
_EXPORTS = {
    "generate_scenario": "nurosim.scenario_generator",
    "generate_batch": "nurosim.scenario_generator",
    "Scenario": "nurosim.scenario_generator",
    "BBox": "nurosim.scenario_generator",
    "MockDetector": "nurosim.perception_model",
    "TimedDetector": "nurosim.perception_model",
    "Evaluator": "nurosim.metrics",
    "EvalResult": "nurosim.metrics",
    "ParallelEvaluator": "nurosim.ray_worker",
    "log_to_mlflow": "nurosim.tracker",
    "annotate_frame": "nurosim.tracker",
}

__version__ = "0.3.0"
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'nurosim' has no attribute {name!r}")
