"""
nurosim.inference
Production-style inference path: letterbox preprocessing, ONNX Runtime
sessions across execution providers, vectorised YOLOv8 decode + NMS.
"""

from nurosim.inference.preprocess  import letterbox, preprocess_batch, LetterboxMeta
from nurosim.inference.postprocess import decode_single, decode_batch, nms
from nurosim.inference.runtime     import OrtRuntime, PROVIDERS, ProviderUnavailableError
from nurosim.inference.detector    import InferenceDetector, StageTiming

__all__ = [
    "letterbox", "preprocess_batch", "LetterboxMeta",
    "decode_single", "decode_batch", "nms",
    "OrtRuntime", "PROVIDERS", "ProviderUnavailableError",
    "InferenceDetector", "StageTiming",
]
