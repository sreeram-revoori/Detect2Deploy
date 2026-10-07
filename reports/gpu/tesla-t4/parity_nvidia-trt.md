### Accuracy parity — profile `nvidia-trt` (300 held-out scenarios, reference `fp32-cpu`)

| target | EP active | mAP@0.5 | Δ | mAP@.5:.95 | Δ | worst class Δ | worst weather Δ |
|---|---|---|---|---|---|---|---|
| **fp32-cpu** | ✅ | 0.9656 | — | 0.8552 | — | — | — |
| fp32-cuda | ✅ | 0.9656 | +0.0000 | 0.8552 | +0.0000 | vehicle +0.0000 | clear +0.0000 |
| fp16-trt | ✅ | 0.9656 | +0.0000 | 0.8557 | +0.0005 | vehicle +0.0000 | clear -0.0000 |
| int8-trt | ✅ | 0.9543 | -0.0113 | 0.8153 | -0.0399 | cone -0.0449 | fog -0.0199 |
| int8_tail32-trt | ✅ | 0.9543 | -0.0113 | 0.8172 | -0.0380 | cone -0.0449 | fog -0.0199 |
| fp16_mixed-trt | ✅ | 0.9656 | +0.0000 | 0.8552 | +0.0000 | vehicle +0.0000 | clear +0.0000 |

**AP@0.5 per class**

| target | vehicle | pedestrian | cyclist | cone |
|---|---|---|---|---|
| fp32-cpu | 0.990 | 0.970 | 0.990 | 0.912 |
| fp32-cuda | 0.990 | 0.970 | 0.990 | 0.912 |
| fp16-trt | 0.990 | 0.970 | 0.990 | 0.912 |
| int8-trt | 0.990 | 0.970 | 0.990 | 0.867 |
| int8_tail32-trt | 0.990 | 0.970 | 0.990 | 0.867 |
| fp16_mixed-trt | 0.990 | 0.970 | 0.990 | 0.912 |

**mAP@0.5 per weather**

| target | clear | rain | fog | night |
|---|---|---|---|---|
| fp32-cpu | 0.972 | 0.965 | 0.962 | 0.966 |
| fp32-cuda | 0.972 | 0.965 | 0.962 | 0.966 |
| fp16-trt | 0.972 | 0.965 | 0.962 | 0.966 |
| int8-trt | 0.968 | 0.954 | 0.942 | 0.950 |
| int8_tail32-trt | 0.968 | 0.954 | 0.942 | 0.950 |
| fp16_mixed-trt | 0.972 | 0.965 | 0.962 | 0.966 |

**Output parity vs `fp32-cpu`** (detections at conf ≥ 0.25; tensors over 16 frames)

| target | ref dets reproduced | extra dets | match IoU | |Δconf| | box MAE px | box max px | score MAE | class agree |
|---|---|---|---|---|---|---|---|---|
| fp32-cuda | 1.000 | 0.000 | 1.000 | 0.0000 | 0.000 | 0.00 | 0.00000 | 1.0000 |
| fp16-trt | 1.000 | 0.000 | 1.000 | 0.0003 | 0.008 | 0.06 | 0.00000 | 1.0000 |
| int8-trt | 0.988 | 0.000 | 0.979 | 0.0178 | 0.159 | 1.83 | 0.00004 | 0.9988 |
| int8_tail32-trt | 0.988 | 0.000 | 0.979 | 0.0178 | 0.159 | 1.84 | 0.00004 | 0.9988 |
| fp16_mixed-trt | 1.000 | 0.000 | 1.000 | 0.0000 | 0.001 | 0.01 | 0.00000 | 1.0000 |
