### Accuracy parity — profile `m4` (300 held-out scenarios, reference `fp32-cpu`)

| target | EP active | mAP@0.5 | Δ | mAP@.5:.95 | Δ | worst class Δ | worst weather Δ |
|---|---|---|---|---|---|---|---|
| **fp32-cpu** | ✅ | 0.9651 | — | 0.8537 | — | — | — |
| fp16-cpu | ✅ | 0.9658 | +0.0007 | 0.8489 | -0.0048 | vehicle +0.0000 | rain +0.0000 |
| int8-cpu | ✅ | 0.9585 | -0.0066 | 0.8095 | -0.0442 | cone -0.0264 | rain -0.0154 |
| int8_full-cpu | ✅ | 0.0000 | -0.9651 | 0.0000 | -0.8537 | vehicle -0.9901 | clear -0.9707 |
| fp32-coreml | ✅ | 0.9651 | +0.0000 | 0.8537 | +0.0000 | vehicle +0.0000 | clear +0.0000 |
| fp16-coreml | ✅ | 0.9574 | -0.0077 | 0.8296 | -0.0241 | cone -0.0306 | rain -0.0192 |
| fp16_mixed-coreml | ✅ | 0.9652 | +0.0001 | 0.8533 | -0.0004 | vehicle +0.0000 | night -0.0000 |

**AP@0.5 per class**

| target | vehicle | pedestrian | cyclist | cone |
|---|---|---|---|---|
| fp32-cpu | 0.990 | 0.970 | 0.990 | 0.910 |
| fp16-cpu | 0.990 | 0.970 | 0.990 | 0.913 |
| int8-cpu | 0.990 | 0.970 | 0.990 | 0.884 |
| int8_full-cpu | 0.000 | 0.000 | 0.000 | 0.000 |
| fp32-coreml | 0.990 | 0.970 | 0.990 | 0.910 |
| fp16-coreml | 0.990 | 0.970 | 0.990 | 0.879 |
| fp16_mixed-coreml | 0.990 | 0.970 | 0.990 | 0.910 |

**mAP@0.5 per weather**

| target | clear | rain | fog | night |
|---|---|---|---|---|
| fp32-cpu | 0.971 | 0.965 | 0.957 | 0.966 |
| fp16-cpu | 0.974 | 0.965 | 0.957 | 0.966 |
| int8-cpu | 0.970 | 0.949 | 0.952 | 0.957 |
| int8_full-cpu | 0.000 | 0.000 | 0.000 | 0.000 |
| fp32-coreml | 0.971 | 0.965 | 0.957 | 0.966 |
| fp16-coreml | 0.969 | 0.945 | 0.951 | 0.963 |
| fp16_mixed-coreml | 0.971 | 0.965 | 0.957 | 0.966 |

**Output parity vs `fp32-cpu`** (detections at conf ≥ 0.25; tensors over 16 frames)

| target | ref dets reproduced | extra dets | match IoU | |Δconf| | box MAE px | box max px | score MAE | class agree |
|---|---|---|---|---|---|---|---|---|
| fp16-cpu | 1.000 | 0.000 | 0.993 | 0.0001 | 0.029 | 0.25 | 0.00000 | 1.0000 |
| int8-cpu | 0.986 | 0.001 | 0.976 | 0.0305 | 0.149 | 1.70 | 0.00005 | 1.0000 |
| int8_full-cpu | 0.000 | 0.000 | 0.000 | 0.0000 | 2.719 | 10.75 | 0.00136 | 0.9039 |
| fp32-coreml | 1.000 | 0.000 | 1.000 | 0.0000 | 0.000 | 0.00 | 0.00000 | 1.0000 |
| fp16-coreml | 0.999 | 0.000 | 0.987 | 0.0013 | 0.071 | 0.46 | 0.00001 | 1.0000 |
| fp16_mixed-coreml | 1.000 | 0.000 | 0.998 | 0.0006 | 0.004 | 0.05 | 0.00000 | 1.0000 |
