"""
detect2deploy.platform — Tier 3: serving and offboard inference at scale.

  graph.py      raw-frame (uint8 NHWC) frontend folded into the model graph
  backend.py    ServingModel: uint8 frames → padded detections (ORT, any EP)
  server.py     PyTriton server with dynamic batching (Colab / any Linux GPU)
  loadgen.py    closed-loop gRPC load generator + Triton server statistics
  serving.py    dynamic-batching sweep (PyTriton or a real Triton server)
  corpus.py     JPEG frame corpus as Parquet shards
  batch.py      Ray Data offboard labelling job, GPU utilisation, cost / 1M frames
  registry.py   MLflow model registry; promotion requires gate evidence
  shadow.py     production-vs-candidate shadow comparison
  report.py     SUMMARY.md for a platform run

Run `python -m detect2deploy.platform --help`.
"""
