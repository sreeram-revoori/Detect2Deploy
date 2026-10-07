.PHONY: install install-train test test-cov run run-fast run-model run-mlflow mlflow-ui \
        dataset train export quantize parity bench gate deploy ci-gate \
        cpp-test gpu-run gpu-docker-build gpu-docker-run docker-build docker-run clean

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
	$(PY) -m pytest tests/ -v --cov=detect2deploy --cov-report=term-missing

# ── Evaluation pipeline (original) ─────────────────────────────────────────────
run:
	$(PY) main.py --n 500 --workers 4 --save-frames

run-fast:
	$(PY) main.py --n 50 --workers 2 --save-frames

run-model:
	$(PY) main.py --n 500 --workers 4 --save-frames --model models/d2d_det_int8.onnx

run-mlflow:
	$(PY) main.py --n 500 --workers 4 --save-frames --mlflow

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

# ── Model deployment pipeline ──────────────────────────────────────────────────
dataset:
	$(PY) -m detect2deploy.deploy dataset

train:
	$(PY) -m detect2deploy.deploy train

export:
	$(PY) -m detect2deploy.deploy export

quantize:
	$(PY) -m detect2deploy.deploy quantize

parity:
	$(PY) -m detect2deploy.deploy parity --profile $(PROFILE)

bench:
	$(PY) -m detect2deploy.deploy bench --profile $(PROFILE)

gate:
	$(PY) -m detect2deploy.deploy gate --profile $(PROFILE)

deploy:            ## quantize → parity → bench → gate
	$(PY) -m detect2deploy.deploy all --profile $(PROFILE)

ci-gate:
	$(PY) -m detect2deploy.deploy all --profile ci-cpu

# ── GPU / TensorRT (Tier 2) ───────────────────────────────────────────────────
cpp-test:          ## host-only C++ build + tests (no CUDA needed)
	cmake -S gpu -B build/cpp-host -DD2D_HOST_ONLY=ON && cmake --build build/cpp-host -j4
	ctest --test-dir build/cpp-host --output-on-failure --no-tests=error

gpu-run:           ## everything, on a machine with an NVIDIA GPU (or Jetson)
	bash scripts/gpu/run_all.sh

gpu-docker-build:
	docker build -f gpu/Dockerfile -t d2d-gpu .

gpu-docker-run:
	docker run --rm --gpus all -v "$(PWD)/reports:/workspace/detect2deploy/reports" \
	           -v "$(PWD)/build:/workspace/detect2deploy/build" d2d-gpu

# ── AI platform (Tier 3) ──────────────────────────────────────────────────────
install-platform:
	pip install -r requirements-platform.txt

platform-smoke:    ## real Triton (PyTriton) on CPU + load + shadow; Linux only
	$(PY) -m detect2deploy.platform smoke

platform-colab:
	bash scripts/platform/colab_run.sh

platform-vm:
	bash scripts/platform/vm_run.sh

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t detect2deploy:latest .

docker-run:
	docker-compose up --build

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/ mlruns/ __pycache__ detect2deploy/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
