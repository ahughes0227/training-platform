"""DINOv3 trainer and inference APIs.

Heavy ML and cloud dependencies are imported only by operations that need them,
so configuration, certification lookup, and offline development work without a
CUDA installation.
"""

from .inference import load_inference_bundle, predict_crop
from .runtime import classify_failure, lookup_certified_runtime

__all__ = [
    "classify_failure",
    "load_inference_bundle",
    "lookup_certified_runtime",
    "predict_crop",
]
