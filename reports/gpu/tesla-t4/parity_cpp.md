### TensorRT C++ path — accuracy parity (300 held-out scenarios, reference ONNX Runtime FP32 CPU)

| engine | mAP@0.5 | Δ | mAP@.5:.95 | Δ | vehicle AP | pedestrian AP | cyclist AP | cone AP | worst weather Δ | ref dets reproduced | budget |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ort-fp32-cpu** | 0.9656 | — | 0.8552 | — | 0.990 | 0.970 | 0.990 | 0.912 | — | — | — |
| fp32_nms_eval | 0.9656 | +0.0000 | 0.8554 | +0.0002 | 0.990 | 0.970 | 0.990 | 0.912 | clear +0.0000 | 1.000 | ✅ |
| fp16_nms_eval | 0.9627 | -0.0029 | 0.8501 | -0.0051 | 0.990 | 0.970 | 0.990 | 0.901 | rain -0.0082 | 1.000 | ✅ |
| fp16_mixed_nms_eval | 0.9656 | -0.0000 | 0.8558 | +0.0006 | 0.990 | 0.970 | 0.990 | 0.912 | clear -0.0001 | 1.000 | ✅ |
| int8_nms_eval | 0.9608 | -0.0047 | 0.8222 | -0.0330 | 0.990 | 0.970 | 0.990 | 0.893 | fog -0.0178 | 0.987 | ✅ |
| fp16_cpunms_eval | 0.9656 | +0.0000 | 0.8562 | +0.0010 | 0.990 | 0.970 | 0.990 | 0.912 | clear -0.0000 | 1.000 | ✅ |
