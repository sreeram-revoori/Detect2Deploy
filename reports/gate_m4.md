### Release gate — profile `m4`: ❌ FAIL

| target | check | value | budget | |
|---|---|---|---|---|
| fp32-cpu | provider | CPUExecutionProvider | cpu | ✅ |
| fp32-cpu | freshness | 4446d7299936 | reports match model on disk | ✅ |
| fp32-cpu | e2e p99 latency | 21.39 ms | ≤ 40 ms | ✅ |
| fp16-cpu | provider | CPUExecutionProvider | cpu | ✅ |
| fp16-cpu | freshness | e3721133721e | reports match model on disk | ✅ |
| fp16-cpu | mAP@0.5 drop | -0.0007 | ≤ 0.01 | ✅ |
| fp16-cpu | worst weather drop | rain +0.0000 | ≤ 0.02 | ✅ |
| fp16-cpu | worst class drop | vehicle +0.0000 | ≤ 0.03 | ✅ |
| fp16-cpu | e2e p99 latency | 20.86 ms | ≤ 40 ms | ✅ |
| int8-cpu | provider | CPUExecutionProvider | cpu | ✅ |
| int8-cpu | freshness | 6cd5fe89c272 | reports match model on disk | ✅ |
| int8-cpu | mAP@0.5 drop | +0.0066 | ≤ 0.01 | ✅ |
| int8-cpu | worst weather drop | rain +0.0154 | ≤ 0.02 | ✅ |
| int8-cpu | worst class drop | cone +0.0264 | ≤ 0.03 | ✅ |
| int8-cpu | e2e p99 latency | 16.80 ms | ≤ 30 ms | ✅ |
| int8_full-cpu | provider | CPUExecutionProvider | cpu | ✅ |
| int8_full-cpu | freshness | f8f479e7f728 | reports match model on disk | ✅ |
| int8_full-cpu | negative control caught | mAP@0.5 drop +0.9651 | must exceed accuracy budget | ✅ |
| fp32-coreml | provider | CoreMLExecutionProvider | coreml | ✅ |
| fp32-coreml | freshness | 4446d7299936 | reports match model on disk | ✅ |
| fp32-coreml | mAP@0.5 drop | +0.0000 | ≤ 0.01 | ✅ |
| fp32-coreml | worst weather drop | clear +0.0000 | ≤ 0.02 | ✅ |
| fp32-coreml | worst class drop | vehicle +0.0000 | ≤ 0.03 | ✅ |
| fp32-coreml | e2e p99 latency | 7.07 ms | ≤ 20 ms | ✅ |
| fp16-coreml | provider | CoreMLExecutionProvider | coreml | ✅ |
| fp16-coreml | freshness | e3721133721e | reports match model on disk | ✅ |
| fp16-coreml | mAP@0.5 drop | +0.0077 | ≤ 0.01 | ✅ |
| fp16-coreml | worst weather drop | rain +0.0192 | ≤ 0.02 | ✅ |
| fp16-coreml | worst class drop | cone +0.0306 | ≤ 0.03 | ❌ |
| fp16-coreml | e2e p99 latency | 3.10 ms | ≤ 10 ms | ✅ |
| fp16_mixed-coreml | provider | CoreMLExecutionProvider | coreml | ✅ |
| fp16_mixed-coreml | freshness | b45e802ade19 | reports match model on disk | ✅ |
| fp16_mixed-coreml | mAP@0.5 drop | -0.0001 | ≤ 0.01 | ✅ |
| fp16_mixed-coreml | worst weather drop | night +0.0000 | ≤ 0.02 | ✅ |
| fp16_mixed-coreml | worst class drop | vehicle +0.0000 | ≤ 0.03 | ✅ |
| fp16_mixed-coreml | e2e p99 latency | 7.08 ms | ≤ 10 ms | ✅ |
