.PHONY: install install-train test test-cov run run-fast run-model run-mlflow mlflow-ui \
        dataset train export quantize parity bench gate deploy ci-gate \
        docker-build docker-run clean

PY      ?= python
PROFILE ?= m4

# ── Local development ──────────────────────────────────────────────────────────
install:
	pip install -r requirements-extras.txt

install-train:
	pip install -r requirements-train.txt

test:
	$(PY) -m pytest tests/ -v --tb=short

test-cov:
	$(PY) -m pytest tests/ -v --cov=nurosim --cov-report=term-missing

# ── Evaluation pipeline (original) ─────────────────────────────────────────────
run:
	$(PY) main.py --n 500 --workers 4 --save-frames

run-fast:
	$(PY) main.py --n 50 --workers 2 --save-frames

run-model:
	$(PY) main.py --n 500 --workers 4 --save-frames --model models/nurosim_det_int8.onnx

run-mlflow:
	$(PY) main.py --n 500 --workers 4 --save-frames --mlflow

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

# ── Model deployment pipeline ──────────────────────────────────────────────────
dataset:
	$(PY) -m nurosim.deploy dataset

train:
	$(PY) -m nurosim.deploy train

export:
	$(PY) -m nurosim.deploy export

quantize:
	$(PY) -m nurosim.deploy quantize

parity:
	$(PY) -m nurosim.deploy parity --profile $(PROFILE)

bench:
	$(PY) -m nurosim.deploy bench --profile $(PROFILE)

gate:
	$(PY) -m nurosim.deploy gate --profile $(PROFILE)

deploy:            ## quantize → parity → bench → gate
	$(PY) -m nurosim.deploy all --profile $(PROFILE)

ci-gate:
	$(PY) -m nurosim.deploy all --profile ci-cpu

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t nurosim-lite:latest .

docker-run:
	docker-compose up --build

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/ mlruns/ __pycache__ nurosim/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
