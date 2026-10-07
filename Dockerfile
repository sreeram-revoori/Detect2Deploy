FROM python:3.11-slim

LABEL maintainer="Sreeram Revoori <rsreddy2104@gmail.com>"
LABEL description="Detect2Deploy — AV Simulation & ML Evaluation Framework"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt requirements-extras.txt ./
RUN pip install --no-cache-dir -r requirements-extras.txt

# ── Source ───────────────────────────────────────────────────────────────────
COPY . .

# ── Output volume ─────────────────────────────────────────────────────────────
VOLUME ["/app/outputs", "/app/mlruns"]

# ── Default: run 500-scenario eval with 4 workers ────────────────────────────
CMD ["python", "main.py", "--n", "500", "--workers", "4", "--save-frames"]
