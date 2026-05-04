"""NuroSim-Lite — AV Simulation & ML Evaluation Framework"""

from nurosim.scenario_generator import generate_scenario, generate_batch, Scenario, BBox
from nurosim.perception_model   import MockDetector, TimedDetector
from nurosim.metrics            import Evaluator, EvalResult
from nurosim.ray_worker         import ParallelEvaluator
from nurosim.tracker            import log_to_mlflow, annotate_frame

__version__ = "0.1.0"
__all__ = [
    "generate_scenario", "generate_batch", "Scenario", "BBox",
    "MockDetector", "TimedDetector",
    "Evaluator", "EvalResult",
    "ParallelEvaluator",
    "log_to_mlflow", "annotate_frame",
]
