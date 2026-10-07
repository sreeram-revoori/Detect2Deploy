# Tier 3 — AI platform: serving and offboard inference at scale

Tiers 1–2 make one model correct and fast on one GPU. Tier 3 is the platform around
it: serve it to many clients, run it over large offboard datasets, and control which
model version is allowed into production.

| piece | what it does | code |
|---|---|---|
| **Raw-frame serving graph** | Preprocessing folded into the ONNX graph: clients send uint8 camera frames (1.2 MB, 4x less than float32) and there is no preprocessing service to keep in sync. Bit-identical to the Tier-1 path. | `detect2deploy/platform/graph.py` |
| **Serving with dynamic batching** | Triton Inference Server — via PyTriton (no Docker, works on Colab) or the Triton container serving TensorRT plans. A closed-loop gRPC load generator sweeps batching configs × concurrency and records client latency plus Triton's own queue / compute split and the batch sizes actually formed. | `server.py`, `loadgen.py`, `serving.py` |
| **Offboard batch inference** | Ray Data: JPEG corpus in Parquet shards → CPU decode tasks → GPU actor pool (model loaded once) → labels in Parquet. Reports frames/s, sampled GPU utilisation (is the GPU busy or starved by decode?), Ray per-operator stats, $ per million frames, and label mAP vs ground truth. | `corpus.py`, `batch.py` |
| **Gated model registry** | MLflow registry. `promote` refuses any version whose Tier-1 gate didn't pass **for that exact file**: the gate JSON records the hash of the model it measured, and the registry compares it with the artefact. Servers load `registry:production`. | `registry.py`, `detect2deploy/deploy/gate.py` |
| **Shadow comparison** | Mirror the same frames to production and a candidate; promote only if the candidate reproduces ≥ 99 % of production's detections and its p99 is ≤ 1.25x. | `shadow.py` |
| **Monitoring (VM)** | Prometheus scraping Triton; a provisioned Grafana dashboard: request rate, avg / p99 latency, queue vs compute time, batch size formed, GPU utilisation / power / memory. | `infra/vm/` |

## Google Colab (T4 GPU runtime)

```python
!git clone -q https://github.com/sreeram-revoori/Detect2Deploy.git
%cd /content/Detect2Deploy
!bash scripts/gpu/colab_setup.sh            # TensorRT + onnxruntime-gpu (same as Tier 2)
!bash scripts/platform/colab_run.sh         # ~25 min; QUICK=1 for ~8 min
```

Results: `reports/platform/<gpu>/SUMMARY.md`. The PyTriton server runs in its own Python 3.12
environment (`build/serve-venv`, made by `scripts/platform/make_serve_env.sh`): PyTriton ≥ 0.5.8
requires numpy < 2, while Colab is Python 3.13 / numpy 2 — and 0.5.7, the last release without
the cap, returned empty output tensors under numpy 2 (caught by the CI smoke test). Colab gives 2 vCPUs, so client, server and
decode share two cores — the batch job will show how decode-bound that is.

## Cloud GPU VM (Docker + NVIDIA Container Toolkit)

Any Linux VM with an NVIDIA GPU, a recent driver, Docker and the NVIDIA Container Toolkit
(e.g. GCP `g2-standard-8` with an L4, or a GCP/AWS deep-learning image, which ship all three).

```bash
git clone https://github.com/sreeram-revoori/Detect2Deploy.git && cd Detect2Deploy
USD_PER_HOUR=0.85 bash scripts/platform/vm_run.sh      # the VM's hourly price → $ / 1M frames
```

What it does differently from Colab:

* **Real Triton container** serving TensorRT plans with EfficientNMS inside — the whole
  model, including preprocessing and NMS, is one engine. Batching configs are applied
  through Triton's model-control API instead of restarting a process.
* **Fresh gate evidence on that GPU**: it runs the Tier-1/2 `nvidia-trt` profile there,
  registers the models against it, and builds the production plan **from the registered
  ONNX** with `--strongly-typed`, so the deployed precision is exactly the registered graph's.
* **Prometheus + Grafana** stay up after the run. Ports bind to localhost only; view them
  through a tunnel instead of opening firewall rules:

  ```bash
  ssh -L 3000:localhost:3000 -L 9090:localhost:9090 <vm>    # Grafana → http://localhost:3000
  ```

* Default corpus 100 k frames (`CORPUS_N`), labelled with 2 CPUs per GPU actor.

Triton (`tritonserver:25.01`) and the plan builder (`tensorrt:25.01`) must come from the
same NGC release so the TensorRT versions match (10.8); change both together.

## Tested

* Locally (macOS): serving graph bit-exact vs Tier 1; ServingModel detections identical to
  `InferenceDetector`; Ray Data job end to end on CPU; registry gating (pass / fail /
  evidence-for-a-different-file); shadow decisions.
* CI (Linux, CPU): the same tests plus `python -m detect2deploy.platform smoke` — a real Triton
  server via PyTriton serving two models, driven over gRPC (the dynamic batcher formed batches
  of 1–3 from 4 clients), statistics read through the access token, and a shadow comparison.
  Getting there surfaced two serving-environment issues, both fixed: PyTriton needs numpy < 2
  (0.5.7, which pip falls back to under numpy 2, returns empty output tensors), and PyTriton
  0.7 protects the statistics / model-repository endpoints with an access token by default.
* Not yet: a GPU run (Colab or VM). The VM path (Docker Triton, Prometheus, Grafana) has not
  been executed at all yet; compose / dashboard files are only validated syntactically.
