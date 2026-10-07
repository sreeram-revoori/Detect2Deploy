"""
server.py
Serve ServingModels through Triton Inference Server via PyTriton.

PyTriton runs a real tritonserver (same HTTP/gRPC API, dynamic batcher,
statistics and Prometheus metrics as a containerised Triton) with the model
executing in this Python process — so it works where Docker doesn't, e.g.
Google Colab. The Triton deployment for a VM is in platform/vm/.

    python -m nurosim.platform serve --model nurosim_det=build/platform/serve_fp16_mixed.onnx \\
        --provider tensorrt --max-batch 8 --queue-delay-us 500
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np

from nurosim.platform.backend import MAX_DET, ServingModel
from nurosim.platform.graph import FRAME_INPUT
from nurosim.platform.loadgen import TOKEN_ENV

logger = logging.getLogger(__name__)

HTTP_PORT, GRPC_PORT, METRICS_PORT = 8000, 8001, 8002


def serve(models: Dict[str, str], provider: str = "cpu", max_batch: int = 8,
          queue_delay_us: int = 500, preferred_batch: Optional[List[int]] = None,
          size: int = 640) -> None:
    from pytriton.decorators import batch
    from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig

    # Load (and on TensorRT, build/deserialize) every model before accepting traffic
    loaded = {name: ServingModel(path, provider=provider, max_batch=max_batch)
              for name, path in models.items()}
    for name, m in loaded.items():
        logger.info("loaded %s ← %s on %s", name, m.model_path, m.runtime.active_providers[0])

    inputs = [Tensor(name=FRAME_INPUT, dtype=np.uint8, shape=(size, size, 3))]
    outputs = [Tensor(name="num_dets", dtype=np.int32, shape=(1,)),
               Tensor(name="det_boxes", dtype=np.float32, shape=(MAX_DET, 4)),
               Tensor(name="det_scores", dtype=np.float32, shape=(MAX_DET,)),
               Tensor(name="det_classes", dtype=np.int32, shape=(MAX_DET,))]
    # max_batch=1 means "no batching": each execution takes exactly one request
    batcher = DynamicBatcher(max_queue_delay_microseconds=queue_delay_us if max_batch > 1 else 0,
                             preferred_batch_size=preferred_batch if max_batch > 1 else None)

    config = TritonConfig(http_port=HTTP_PORT, grpc_port=GRPC_PORT, metrics_port=METRICS_PORT)
    kwargs = {}
    try:
        # PyTriton >= 0.7 protects statistics / model-repository / shared-memory
        # endpoints with an access token; share ours with the clients via env.
        from pytriton.triton import TritonSecurityConfig
        token = os.environ.get(TOKEN_ENV)
        if token:
            kwargs["security_config"] = TritonSecurityConfig(access_token=token)
    except ImportError:
        pass
    with Triton(config=config, **kwargs) as triton:
        for name, model in loaded.items():
            def make_fn(m: ServingModel):
                @batch
                def infer_fn(**inputs):
                    return m.infer(inputs[FRAME_INPUT])
                return infer_fn

            triton.bind(model_name=name, infer_func=make_fn(model), inputs=inputs, outputs=outputs,
                        config=ModelConfig(max_batch_size=max_batch, batcher=batcher))
        logger.info("serving %s on grpc :%d (max_batch=%d, queue_delay=%dus)",
                    list(loaded), GRPC_PORT, max_batch, queue_delay_us)
        triton.serve()
