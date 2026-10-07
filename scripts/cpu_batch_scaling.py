#!/usr/bin/env python3
"""
cpu_batch_scaling.py
Inference-only latency vs batch size on the ONNX Runtime CPU EP, with and
without MLAS's KleidiAI kernels (Arm SME/SME2, e.g. Apple M4).

    python scripts/cpu_batch_scaling.py models/d2d_det_fp32.onnx models/d2d_det_int8.onnx
"""

import sys
import time

import numpy as np
import onnxruntime as ort


def median_ms(sess, x, warmup=5, iters=20):
    for _ in range(warmup):
        sess.run(None, {"images": x})
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, {"images": x})
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def main(paths):
    x = np.random.default_rng(0).random((4, 3, 640, 640), dtype=np.float32)
    print(f"onnxruntime {ort.__version__}")
    print(f"{'model':<28} {'kleidiai':<9} {'bs=1':>9} {'bs=2':>9} {'bs=4':>9}   ms/frame @bs4")
    for path in paths:
        for disable in ("0", "1"):
            so = ort.SessionOptions()
            so.add_session_config_entry("mlas.disable_kleidiai", disable)
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            r = {bs: median_ms(sess, x[:bs]) for bs in (1, 2, 4)}
            print(f"{path.split('/')[-1]:<28} {'off' if disable == '1' else 'on':<9} "
                  f"{r[1]:>8.1f} {r[2]:>8.1f} {r[4]:>8.1f}   {r[4] / 4:>8.1f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["models/d2d_det_fp32.onnx", "models/d2d_det_int8.onnx"])
