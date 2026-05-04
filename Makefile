.PHONY: install test run run-fast run-mlflow docker-build docker-run clean

# ── Local development ──────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=nurosim --cov-report=term-missing

# ── Run pipeline ───────────────────────────────────────────────────────────────
run:
	python main.py --n 500 --workers 4 --save-frames

run-fast:
	python main.py --n 50 --workers 2 --save-frames

run-mlflow:
	python main.py --n 500 --workers 4 --save-frames --mlflow

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t nurosim-lite:latest .

docker-run:
	docker-compose up --build

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/ mlruns/ __pycache__ nurosim/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
