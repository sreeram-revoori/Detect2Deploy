"""
detect2deploy.inference
Production-style inference path: letterbox preprocessing, ONNX Runtime
sessions across execution providers, vectorised YOLOv8 decode + NMS.
"""

from detect2deploy.inference.preprocess  import letterbox, preprocess_batch, LetterboxMeta
from detect2deploy.inference.postprocess import decode_single, decode_batch, nms
from detect2deploy.inference.runtime     import OrtRuntime, PROVIDERS, ProviderUnavailableError
from detect2deploy.inference.detector    import InferenceDetector, StageTiming

__all__ = [
    "letterbox", "preprocess_batch", "LetterboxMeta",
    "decode_single", "decode_batch", "nms",
    "OrtRuntime", "PROVIDERS", "ProviderUnavailableError",
    "InferenceDetector", "StageTiming",
]
