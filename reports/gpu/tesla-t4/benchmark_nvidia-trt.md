### Latency — profile `nvidia-trt`

Host: `Linux-6.6.122+-x86_64-with-glibc2.39`, 2 cores, onnxruntime 1.30.0. Batch 1, 3 independent sessions × (30 warmup + 200 measured iterations), pooled, deploy conf 0.25, intra-op threads: ORT default (all cores).

| target | EP active | model MB | e2e p50 | e2e p99 | max | jitter | per-session p50 | pre p50 | infer p50 | post p50 | cold start |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32-cpu | ✅ | 12.21 | 102.48 | 173.21 | 185.81 | 70.74 | 102.4 / 102.8 / 102.0 | 5.51 | 95.87 | 1.07 | 181 |
| fp32-cuda | ✅ | 12.21 | 11.96 | 18.95 | 22.09 | 6.99 | 12.2 / 11.9 / 11.8 | 4.69 | 6.43 | 0.83 | 1050 |
| fp16-trt | ✅ | 12.21 | 9.77 | 13.80 | 15.07 | 4.02 | 9.7 / 9.8 / 9.8 | 4.64 | 4.32 | 0.81 | 1892 |
| int8-trt | ✅ | 3.4 | 10.85 | 16.27 | 25.04 | 5.42 | 10.9 / 10.7 / 11.1 | 4.68 | 5.30 | 0.86 | 2148 |
| int8_tail32-trt | ✅ | 3.4 | 10.82 | 16.95 | 20.31 | 6.12 | 10.7 / 10.7 / 14.4 | 4.71 | 5.28 | 0.89 | 1690 |
| fp16_mixed-trt | ✅ | 6.21 | 10.08 | 13.60 | 15.70 | 3.52 | 9.9 / 10.2 / 10.2 | 4.61 | 4.72 | 0.81 | 138 |

All latencies in ms. *cold start* = session load + first inference.

**Batch sweep** (inference only, throughput in frames/s at p50)

| target | bs=1 | bs=2 | bs=4 | bs=8 |
|---|---|---|---|---|
| fp32-cpu | 10 fps (95.6 ms) | 10 fps (199.9 ms) | 10 fps (408.0 ms) | 10 fps (837.7 ms) |
| fp32-cuda | 144 fps (7.0 ms) | 161 fps (12.4 ms) | 158 fps (25.3 ms) | 184 fps (43.4 ms) |
| fp16-trt | 384 fps (2.6 ms) | — | — | — |
| int8-trt | 252 fps (4.0 ms) | — | — | — |
| int8_tail32-trt | 237 fps (4.2 ms) | — | — | — |
| fp16_mixed-trt | 213 fps (4.7 ms) | — | — | — |
