# GPU run — `tesla-t4`

**Tesla T4** (sm_75, 40 SMs) · TensorRT 10.14.1 · CUDA runtime 13000 / driver 13000

Full environment: [`env.txt`](env.txt)

## C++ pipeline latency (batch 1, ms)

Each row changes one thing from the row above. *host pre* is CPU letterbox (cpu) or the memcpy into pinned staging (gpu); GPU stages are CUDA-event timed; with a CUDA graph the GPU side is one launch, so only end-to-end is split.

| config | pre | post | CUDA graph | e2e p50 | e2e p99 | max | jitter | host pre | H2D | GPU pre | infer | D2H | post | fps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP32 · CPU pre · CPU NMS | cpu | cpu_nms | — | 5.834 | 6.579 | 8.162 | 0.744 | 0.971 | 0.409 | 0.003 | 4.293 | 0.028 | 0.086 | 171 |
| FP16 · CPU pre · CPU NMS | cpu | cpu_nms | — | 3.647 | 5.529 | 6.434 | 1.883 | 1.748 | 0.410 | 0.003 | 1.316 | 0.028 | 0.104 | 274 |
| FP16 · GPU pre · CPU NMS | gpu | cpu_nms | — | 1.969 | 2.064 | 2.175 | 0.095 | 0.145 | 0.110 | 0.030 | 1.556 | 0.029 | 0.086 | 508 |
| FP16 · GPU pre · EfficientNMS | gpu | efficient_nms | — | 1.960 | 2.035 | 2.159 | 0.076 | 0.143 | 0.109 | 0.032 | 1.648 | 0.011 | 0.001 | 510 |
| FP16 · GPU pre · EfficientNMS · graph | gpu | efficient_nms | ✅ | 1.980 | 2.038 | 2.094 | 0.058 | 0.140 | — | — | — | — | 0.002 | 505 |
| FP16+FP32 tail · GPU pre · NMS · graph | gpu | efficient_nms | ✅ | 2.019 | 2.434 | 3.970 | 0.414 | 0.143 | — | — | — | — | 0.002 | 495 |
| INT8 · GPU pre · NMS · graph | gpu | efficient_nms | ✅ | 1.915 | 1.983 | 2.679 | 0.068 | 0.146 | — | — | — | — | 0.002 | 522 |

## Batch sweep (offboard / batch-inference throughput)

| batch | e2e p50 (ms) | e2e p99 (ms) | frames/s |
|---|---|---|---|
| 1 | 1.95 | 2.02 | 513 |
| 2 | 3.44 | 3.60 | 582 |
| 4 | 6.60 | 6.88 | 606 |
| 8 | 12.74 | 13.49 | 628 |

## Two models on one GPU

A = latency-critical detector at 30 Hz (deadline 33.3 ms), batch 1 · B = background model saturating the GPU at batch 8 · 8 s per phase.

| phase | A p50 | A p99 | A max | A deadline misses | B frames/s |
|---|---|---|---|---|---|
| a_solo | 3.60 | 3.75 | 3.79 | 0 / 240 | 0 |
| b_solo | — | — | — | — | 647 |
| concurrent_equal_priority | 6.18 | 11.88 | 12.01 | 0 / 240 | 618 |
| concurrent_a_high_priority | 4.79 | 6.51 | 7.20 | 0 / 240 | 610 |

## Accuracy parity of the TensorRT engines

| engine | mAP@0.5 | Δ | mAP@.5:.95 | Δ | vehicle AP | pedestrian AP | cyclist AP | cone AP | worst weather Δ | ref dets reproduced | budget |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ort-fp32-cpu** | 0.9656 | — | 0.8552 | — | 0.990 | 0.970 | 0.990 | 0.912 | — | — | — |
| fp32_nms_eval | 0.9656 | +0.0000 | 0.8554 | +0.0002 | 0.990 | 0.970 | 0.990 | 0.912 | clear +0.0000 | 1.000 | ✅ |
| fp16_nms_eval | 0.9627 | -0.0029 | 0.8501 | -0.0051 | 0.990 | 0.970 | 0.990 | 0.901 | rain -0.0082 | 1.000 | ✅ |
| fp16_mixed_nms_eval | 0.9656 | -0.0000 | 0.8558 | +0.0006 | 0.990 | 0.970 | 0.990 | 0.912 | clear -0.0001 | 1.000 | ✅ |
| int8_nms_eval | 0.9608 | -0.0047 | 0.8222 | -0.0330 | 0.990 | 0.970 | 0.990 | 0.893 | fog -0.0178 | 0.987 | ✅ |
| fp16_cpunms_eval | 0.9656 | +0.0000 | 0.8562 | +0.0010 | 0.990 | 0.970 | 0.990 | 0.912 | clear -0.0000 | 1.000 | ✅ |

## ONNX Runtime CUDA / TensorRT EPs (Python harness, profile `nvidia-trt`)

[parity_nvidia-trt.md](parity_nvidia-trt.md) · [benchmark_nvidia-trt.md](benchmark_nvidia-trt.md) · [gate_nvidia-trt.md](gate_nvidia-trt.md)

### Release gate — profile `nvidia-trt`: ❌ FAIL

| target | check | value | budget | |
|---|---|---|---|---|
| fp32-cpu | provider | CPUExecutionProvider | cpu | ✅ |
| fp32-cpu | freshness | 4446d7299936 | reports match model on disk | ✅ |
| fp32-cuda | provider | CUDAExecutionProvider | cuda | ✅ |
| fp32-cuda | freshness | 4446d7299936 | reports match model on disk | ✅ |
| fp32-cuda | mAP@0.5 drop | +0.0000 | ≤ 0.01 | ✅ |
| fp32-cuda | worst weather drop | clear +0.0000 | ≤ 0.02 | ✅ |
| fp32-cuda | worst class drop | vehicle +0.0000 | ≤ 0.03 | ✅ |
| fp16-trt | provider | TensorrtExecutionProvider | tensorrt | ✅ |
| fp16-trt | freshness | 4446d7299936 | reports match model on disk | ✅ |
| fp16-trt | mAP@0.5 drop | -0.0000 | ≤ 0.01 | ✅ |
| fp16-trt | worst weather drop | clear +0.0000 | ≤ 0.02 | ✅ |
| fp16-trt | worst class drop | vehicle +0.0000 | ≤ 0.03 | ✅ |
| int8-trt | provider | TensorrtExecutionProvider | tensorrt | ✅ |
| int8-trt | freshness | fb14f0cd56f4 | reports match model on disk | ✅ |
| int8-trt | mAP@0.5 drop | +0.0113 | ≤ 0.01 | ❌ |
| int8-trt | worst weather drop | fog +0.0199 | ≤ 0.02 | ✅ |
| int8-trt | worst class drop | cone +0.0449 | ≤ 0.03 | ❌ |
| int8_tail32-trt | provider | TensorrtExecutionProvider | tensorrt | ✅ |
| int8_tail32-trt | freshness | fb14f0cd56f4 | reports match model on disk | ✅ |
| int8_tail32-trt | mAP@0.5 drop | +0.0113 | ≤ 0.01 | ❌ |
| int8_tail32-trt | worst weather drop | fog +0.0199 | ≤ 0.02 | ✅ |
| int8_tail32-trt | worst class drop | cone +0.0449 | ≤ 0.03 | ❌ |
| fp16_mixed-trt | provider | TensorrtExecutionProvider | tensorrt | ✅ |
| fp16_mixed-trt | freshness | b45e802ade19 | reports match model on disk | ✅ |
| fp16_mixed-trt | mAP@0.5 drop | +0.0000 | ≤ 0.01 | ✅ |
| fp16_mixed-trt | worst weather drop | clear +0.0000 | ≤ 0.02 | ✅ |
| fp16_mixed-trt | worst class drop | vehicle +0.0000 | ≤ 0.03 | ✅ |

## Nsight Systems

Open in Nsight Systems: `nsys_fp16_nms.nsys-rep`, `nsys_fp16_nms_graph.nsys-rep`. Kernel / NVTX summaries: [`nsys_fp16_nms.txt`](nsys_fp16_nms.txt), [`nsys_fp16_nms_graph.txt`](nsys_fp16_nms_graph.txt)

## Gate verdicts

- ORT gate (nvidia-trt profile): FAIL — a model missed its budget (see the gate report)
- - ORT gate (nvidia-trt profile): FAIL — a model missed its budget (see the gate report)

## Run status

All steps succeeded.
