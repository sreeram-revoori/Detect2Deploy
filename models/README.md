# Model card — `d2d_det`

| | |
|---|---|
| Architecture | YOLOv8n (3.0 M params), 4 classes: vehicle, pedestrian, cyclist, cone |
| Input | `images` — float32 `[batch, 3, 640, 640]`, RGB, [0, 1], letterboxed (pad 114) |
| Output | `output0` — float32 `[batch, 8, 8400]` = (cx, cy, w, h, 4 class scores) per anchor, pre-NMS |
| Training data | 3,000 synthetic Detect2Deploy BEV frames (seeds 0–2,999), uniform over clear / rain / fog / night |
| Validation | 300 frames (seeds 50,000+): mAP@0.5 **0.961**, mAP@0.5:0.95 **0.779** |
| Per-class val AP@0.5 | vehicle 0.995 · pedestrian 0.960 · cyclist 0.995 · cone 0.895 |
| Training | from scratch (`yolov8n.yaml`, no COCO weights), 40 epochs, batch 16, Apple M4 (MPS), ~100 min |
| Export | ONNX opset 17, dynamic batch, H/W pinned to 640, anchor grid constant-folded (onnxslim) |

## Artefacts

Only the FP32 reference is committed. Every other variant is rebuilt from it by
`python -m detect2deploy.deploy quantize` (CI does exactly this), so a variant can never
silently drift from the model it claims to come from.

| file | how it's made | size |
|---|---|---|
| `d2d_det_fp32.onnx` | export of the best checkpoint (sha256 `4446d7299936…`) | 12.2 MB |
| `d2d_det_fp16.onnx` | all ops FP16, FP32 I/O | 6.1 MB |
| `d2d_det_fp16_mixed.onnx` | FP16, Detect-head tail (24 nodes) kept FP32 | 6.2 MB |
| `d2d_det_int8.onnx` | INT8 QDQ, per-channel symmetric, 128 weather-stratified calibration frames, head tail FP32 | 3.5 MB |
| `d2d_det_int8_trt.onnx` | as `int8` but conv biases stay float — TensorRT's parser rejects INT32 bias `DequantizeLinear`; identical accuracy and CPU latency | 3.4 MB |
| `d2d_det_int8_full.onnx` | INT8 QDQ, every op quantised — **known broken**, kept as the gate's negative control | 3.4 MB |

## Known limitations

* Cones render at 4–5 px. They are the weakest class (val AP@0.5:0.95 0.43) and
  the first to regress under reduced precision.
* Synthetic, top-down, axis-aligned boxes only — this model exists to exercise the
  deployment pipeline, not as a perception model.
* Ultralytics weights are AGPL-3.0.

To retrain: `make dataset train export` (needs `requirements-train.txt`).
