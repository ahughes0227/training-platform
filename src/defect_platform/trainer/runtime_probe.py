"""Small Vertex handshake entrypoint. Probe target is provided by env variables."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import tempfile

logger = logging.getLogger(__name__)


def main() -> None:
    from ..telemetry import bind_context, configure_logging
    configure_logging()
    bind_context(run_id=os.getenv("DEFECT_PLATFORM_RUN_ID", "vertex-handshake"),
                 image_digest=os.getenv("DEFECT_PLATFORM_IMAGE_DIGEST", ""),
                 step="vertex_handshake")
    import torch
    from google.cloud import storage

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Vertex GPU handshake requires an available CUDA GPU")
    logger.info("vertex_gpu_available", extra={"stage": "vertex_gpu_available"})
    # Exercise allocation, arithmetic and synchronization on the assigned GPU.
    values = torch.arange(64, dtype=torch.float32, device="cuda").reshape(8, 8)
    product = values @ values.T
    torch.cuda.synchronize()
    gpu_tensor_operation = bool(product.is_cuda and torch.isfinite(product).all().item())
    if not gpu_tensor_operation:
        raise RuntimeError("GPU tensor operation failed")
    logger.info("vertex_gpu_tensor_operation_passed", extra={"stage": "vertex_gpu_tensor_operation"})

    uri = os.environ["DEFECT_PLATFORM_GCS_PROBE_URI"]
    bucket_name, _, prefix = uri[5:].partition("/")
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(prefix.rstrip("/") + "/handshake.txt")
    payload = b"defect-platform-vertex-handshake"
    blob.upload_from_string(payload)
    downloaded = blob.download_as_bytes()
    blob.delete()
    if downloaded != payload:
        raise RuntimeError("GCS handshake checksum mismatch")
    logger.info("vertex_gcs_read_write_passed", extra={"stage": "vertex_gcs_read_write"})
    result = {"gpu_count": torch.cuda.device_count(),
              "gpu_name": torch.cuda.get_device_name(0),
              "gpu_tensor_operation": gpu_tensor_operation,
              "python_version": platform.python_version(),
              "pytorch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "gcs_read": True, "gcs_write": True,
              "gcs_sha256": hashlib.sha256(downloaded).hexdigest(),
              "image_digest": os.environ["DEFECT_PLATFORM_IMAGE_DIGEST"]}
    result_uri = os.environ["DEFECT_PLATFORM_GCS_RESULT_URI"]
    result_bytes = (json.dumps(result, sort_keys=True) + "\n").encode("utf-8")
    result_bucket_name, _, result_key = result_uri[5:].partition("/")
    result_blob = client.bucket(result_bucket_name).blob(result_key)
    # GCS object replacement is atomic on successful finalize; upload a complete
    # result document and never expose a partially written certification record.
    with tempfile.NamedTemporaryFile(prefix="vertex-handshake-", suffix=".json") as temp:
        temp.write(result_bytes)
        temp.flush()
        result_blob.upload_from_filename(temp.name, content_type="application/json")
    logger.info("vertex_handshake_result_written", extra={"stage": "vertex_handshake_complete"})
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
