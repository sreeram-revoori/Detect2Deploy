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
