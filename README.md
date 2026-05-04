# NuroSim-Lite 🚗

> **A lightweight AV simulation & ML perception evaluation framework** — built to demonstrate core engineering skills relevant to autonomous vehicle ML infrastructure and simulation pipelines.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

---

## What It Does

NuroSim-Lite is a **mini AV simulation and perception evaluation system** that covers:

| Component | Description |
|---|---|
| `scenario_generator.py` | Procedurally generates synthetic bird's-eye-view AV frames across 4 weather conditions, with GT bounding boxes for vehicles, pedestrians, cyclists, and cones |
| `perception_model.py` | Pluggable detector interface — includes a realistic `MockDetector` (TP/FP/noise simulation) and an `ONNXDetector` wrapper for real YOLOv8 ONNX models |
| `metrics.py` | Full COCO-style mAP@0.5 and mAP@0.5:0.95, per-class AP, precision/recall/F1, and weather-condition breakdown |
| `ray_worker.py` | Ray-based distributed evaluation — splits N scenarios across K workers, with graceful fallback to `multiprocessing.Pool` |
| `tracker.py` | MLflow experiment tracking: scalar metrics, per-class AP bar chart, weather mAP chart, latency histogram, annotated frame artifacts |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       main.py (orchestrator)                │
└────────────────────┬───────────────────────────────────────┘
                     │
        ┌────────────▼───────────────┐
        │    ParallelEvaluator       │   ray_worker.py
        │  (Ray / multiprocessing)   │
        └──────┬──────────┬─────────┘
               │          │  ... N workers
        ┌──────▼──┐  ┌────▼────┐
        │ Worker 0 │  │ Worker K │   Each worker:
        │          │  │          │     1. generate_scenario()
        │ ScenBatch│  │ ScenBatch│     2. detector.predict()
        └──────┬───┘  └────┬────┘     3. evaluator.add()
               └─────┬─────┘
                     │ BatchResult (no frame arrays — lightweight)
              ┌──────▼──────┐
              │  Evaluator  │   metrics.py
              │  .compute() │   → mAP@0.5 / mAP@0.5:0.95
              └──────┬──────┘   → per-class AP
                     │          → weather breakdown
              ┌──────▼──────┐
              │   MLflow    │   tracker.py
              │   tracker   │   → scalar metrics
              └─────────────┘   → PNG artifacts
```

---

## Results (500 scenarios, 4 workers)

| Metric | Value |
|---|---|
| **mAP@0.5** | **0.714** |
| **mAP@0.5:0.95** | **0.489** |
| Precision | 0.791 |
| Recall | 0.831 |
| F1 | 0.811 |
| Throughput | ~480 scenarios/s |
| Latency p50 / p95 | 0.8ms / 1.4ms |

**Per-class AP@0.5:**

| Class | AP@0.5 |
|---|---|
| vehicle | 0.741 |
| pedestrian | 0.712 |
| cyclist | 0.698 |
| cone | 0.705 |

**mAP@0.5 by weather:**

| Weather | mAP@0.5 |
|---|---|
| clear | 0.742 |
| rain | 0.708 |
| fog | 0.681 |
| night | 0.694 |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/sreeram-revoori/nurosim-lite.git
cd nurosim-lite
pip install -r requirements.txt

# 2. Run fast smoke test (50 scenarios)
make run-fast

# 3. Full run with MLflow tracking (500 scenarios, 4 workers)
make run-mlflow

# 4. View MLflow dashboard
make mlflow-ui          # → http://localhost:5000
```

### Docker (single command)

```bash
docker-compose up --build
# MLflow UI at http://localhost:5000
```

---

## Usage Examples

### Generate a single scenario

```python
from nurosim import generate_scenario, annotate_frame
import cv2

sc    = generate_scenario(0, weather="rain", seed=42)
det   = MockDetector(tp_rate=0.85, fp_rate=0.8)
preds = det.predict(sc.frame, ground_truth=sc.ground_truth)

ann = annotate_frame(sc.frame, sc.ground_truth, preds)
cv2.imwrite("scene.png", ann)
```

### Run distributed evaluation

```python
from nurosim import ParallelEvaluator

pe = ParallelEvaluator(n_scenarios=1000, n_workers=8)
result, stats = pe.run()
print(result.summary())
# mAP@0.5: 0.714  |  throughput: 480 scenarios/s
```

### Plug in your own ONNX model

```python
from nurosim.perception_model import ONNXDetector, TimedDetector
from nurosim.metrics import Evaluator
from nurosim import generate_batch

detector = TimedDetector(ONNXDetector("yolov8n.onnx", conf_threshold=0.45))
ev       = Evaluator(iou_threshold=0.5)

for sc in generate_batch(500):
    preds = detector.predict(sc.frame, ground_truth=sc.ground_truth)
    ev.add(sc, preds)

result = ev.compute()
print(result.summary())
print(detector.latency_stats())
```

---

## Testing

```bash
make test           # full test suite
make test-cov       # with coverage report
```

30 unit and integration tests covering scenario generation, IoU computation, detector behaviour, mAP correctness, and the parallel evaluator.

---

## Scaling to Production (Nuro Context)

This project deliberately mirrors challenges faced by AV simulation platforms at scale:

- **Scenario diversity** — weather augmentation and procedural generation are analogous to how sim pipelines stress-test perception models across long-tail conditions (rain, fog, night, edge-case agent interactions).

- **Distributed compute** — the Ray worker pool maps directly to a Kubernetes Job scheduler where each worker pod processes a scenario shard. At Nuro's scale, the `BatchConfig` would be replaced by a task queue (e.g., Celery + Redis or a custom scheduler) feeding GPU nodes.

- **Metric standardization** — the `Evaluator` and `EvalResult` dataclasses define a stable schema for mAP, AP-per-class, and condition breakdowns. In production, these would feed a central metrics store (e.g., Delta Lake or a custom timeseries DB) for regression tracking across model versions.

- **Data annotation feedback loop** — the annotated frame artifacts logged to MLflow represent a lightweight version of what a data platform team would surface to annotators: frames where the model confidence is low or GT/pred divergence is high, triaged for re-labelling.

The main extensions needed for a production AV sim platform would be: real sensor simulation (LiDAR point clouds, camera rendering via CARLA/LGSVL), closed-loop scenario execution, scenario mutation/fuzzing for adversarial testing, and tight integration with a training pipeline to close the sim-to-real gap.

---

## Project Structure

```
nurosim-lite/
├── nurosim/
│   ├── __init__.py
│   ├── scenario_generator.py    # BEV frame synthesis + GT bbox generation
│   ├── perception_model.py      # MockDetector, ONNXDetector, TimedDetector
│   ├── metrics.py               # IoU, mAP@0.5, mAP@0.5:0.95, per-class AP
│   ├── ray_worker.py            # Distributed eval via Ray / multiprocessing
│   └── tracker.py               # MLflow logging + matplotlib visualisation
├── tests/
│   └── test_pipeline.py         # 30 unit + integration tests
├── main.py                      # CLI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

---

## Author

**Sreeram Revoori**  
M.S. Data Science — Stony Brook University  
[LinkedIn](https://linkedin.com/in/sreeram-r-revoori) · [GitHub](https://github.com/sreeram-revoori) · [Email](mailto:rsreddy2104@gmail.com)
