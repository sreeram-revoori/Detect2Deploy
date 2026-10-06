### Latency — profile `m4`

Host: `macOS-26.6.2-arm64-arm-64bit`, 10 cores, onnxruntime 1.30.0. Batch 1, 3 independent sessions × (30 warmup + 200 measured iterations), pooled, deploy conf 0.25, intra-op threads: 4.

| target | EP active | model MB | e2e p50 | e2e p99 | max | jitter | per-session p50 | pre p50 | infer p50 | post p50 | cold start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32-cpu | ✅ | 12.21 | 18.83 | 21.39 | 33.42 | 2.56 | 18.4 / 19.2 / 19.0 | 1.58 | 16.90 | 0.29 | 34 |
| fp16-cpu | ✅ | 6.15 | 20.02 | 20.86 | 30.25 | 0.85 | 20.0 / 20.0 / 20.1 | 1.75 | 17.98 | 0.28 | 45 |
| int8-cpu | ✅ | 3.46 | 15.97 | 16.80 | 16.89 | 0.83 | 16.0 / 15.9 / 16.1 | 1.77 | 13.85 | 0.28 | 55 |
| int8_full-cpu | ✅ | 3.4 | 15.37 | 16.22 | 16.71 | 0.86 | 15.4 / 15.4 / 15.4 | 1.76 | 13.47 | 0.11 | 54 |
| fp32-coreml | ✅ | 12.21 | 6.90 | 7.07 | 7.18 | 0.17 | 6.9 / 6.9 / 6.9 | 1.36 | 5.33 | 0.20 | 1704 |
| fp16-coreml | ✅ | 6.15 | 2.96 | 3.10 | 3.12 | 0.15 | 3.0 / 3.0 / 3.0 | 1.37 | 1.41 | 0.18 | 552 |
| fp16_mixed-coreml | ✅ | 6.21 | 4.39 | 7.08 | 7.16 | 2.69 | 4.4 / 4.4 / 4.4 | 1.36 | 2.82 | 0.20 | 501 |

All latencies in ms. *cold start* = session load + first inference.

**Batch sweep** (inference only, throughput in frames/s at p50)

| target | bs=1 | bs=2 | bs=4 | bs=8 |
|---|---|---|---|---|
| fp32-cpu | 57 fps (17.4 ms) | 17 fps (115.2 ms) | 32 fps (126.8 ms) | 31 fps (257.2 ms) |
| fp16-cpu | 54 fps (18.4 ms) | 16 fps (122.4 ms) | 30 fps (131.7 ms) | 30 fps (264.5 ms) |
| int8-cpu | 71 fps (14.1 ms) | 76 fps (26.4 ms) | 76 fps (52.3 ms) | 77 fps (104.2 ms) |
| int8_full-cpu | 75 fps (13.4 ms) | 78 fps (25.5 ms) | 80 fps (50.2 ms) | 80 fps (100.0 ms) |
| fp32-coreml | 199 fps (5.0 ms) | — | — | — |
| fp16-coreml | 843 fps (1.2 ms) | — | — | — |
| fp16_mixed-coreml | 374 fps (2.7 ms) | — | — | — |
