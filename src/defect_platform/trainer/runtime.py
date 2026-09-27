"""Runtime validation, certification lookup and actionable failure ownership."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from ..contracts import CertifiedRuntime, ValidationResults, VertexJobConfig


class RuntimeValidationError(RuntimeError):
    """A runtime certification stage failed with a stable stage identifier."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class FailureClassification:
    code: str
    owner: str
    retryable: bool
    explanation: str


def lookup_certified_runtime(runtimes: Iterable[CertifiedRuntime], image_digest: str,
                             runtime_id: str | None = None) -> CertifiedRuntime:
    """Return only a fully validated runtime matching the exact immutable digest."""
    if "@sha256:" not in image_digest or len(image_digest.rsplit(":", 1)[-1]) != 64:
        raise ValueError("image_digest must be an immutable registry digest")
    matches = [r for r in runtimes if r.image_digest == image_digest
               and (runtime_id is None or r.runtime_id == runtime_id)]
    certified = [r for r in matches if r.certified and all(r.validation.model_dump().values())]
    if len(certified) != 1:
        raise LookupError("no unique certified runtime exists for this image digest")
    return certified[0]


def certify_runtime(*, runtime_id: str, source_commit: str, image_tag: str,
                    image_digest: str, runtime_version: str, python_version: str,
                    pytorch_version: str, cuda_version: str,
                    trainer_check: Callable[[], bool],
                    container_gpu_check: Callable[[], bool],
                    vertex_gpu_check: Callable[[], bool],
                    gcs_read_check: Callable[[], bool],
                    gcs_write_check: Callable[[], bool]) -> CertifiedRuntime:
    """Run every certification gate and bind successful evidence to one digest.

    Callbacks are explicit so the orchestrator can provide real environment and
    cloud probes. No build or push operation is performed by this function.
    """
    if "@sha256:" not in image_digest or len(image_digest.rsplit(":", 1)[-1]) != 64:
        raise RuntimeValidationError("image", "certification requires an immutable sha256 image digest")
    probes = [
        ("trainer", trainer_check), ("container_gpu", container_gpu_check),
        ("vertex_gpu", vertex_gpu_check), ("gcs_read", gcs_read_check),
        ("gcs_write", gcs_write_check),
    ]
    outcomes = {}
    for stage, probe in probes:
        try:
            outcomes[stage] = bool(probe())
        except Exception as exc:
            raise RuntimeValidationError(stage, f"{stage} check failed: {exc}") from exc
        if not outcomes[stage]:
            raise RuntimeValidationError(stage, f"{stage} check returned false")
    return CertifiedRuntime(
        runtime_id=runtime_id, source_commit=source_commit, image_tag=image_tag,
        image_digest=image_digest, runtime_version=runtime_version,
        python_version=python_version, pytorch_version=pytorch_version,
        cuda_version=cuda_version, validation=ValidationResults(**outcomes),
        certified=True, certified_at=datetime.now(UTC),
    )


def vertex_gpu_handshake(config: VertexJobConfig, *, image_uri: str,
                         staging_bucket_uri: str, submit_and_wait: Callable[[dict], dict],
                         result_uri: str | None = None,
                         run_id: str | None = None,
                         timeout_seconds: int = 900) -> dict:
    """Submit a bounded GPU handshake job and verify GPU and GCS access.

    ``submit_and_wait`` is the cloud adapter supplied by the control plane. The
    payload is designed as a short job that reports device metadata and performs
    a staging-bucket write/read checksum roundtrip.
    """
    if not image_uri or "@sha256:" not in image_uri:
        raise RuntimeValidationError("vertex_gpu", "handshake image must use the candidate digest")
    result_uri = result_uri or staging_bucket_uri.rstrip("/") + "/handshake/result.json"
    payload = {
        "project": config.project, "region": config.region,
        "machine_type": config.machine_type, "accelerator_type": config.accelerator_type,
        "accelerator_count": config.accelerator_count, "service_account": config.service_account,
        "staging_uri": config.staging_uri, "network": config.network,
        "image_uri": image_uri, "gcs_probe_uri": staging_bucket_uri,
        "timeout_seconds": timeout_seconds,
        "environment": {"DEFECT_PLATFORM_GCS_PROBE_URI": staging_bucket_uri,
                        "DEFECT_PLATFORM_GCS_RESULT_URI": result_uri,
                        "DEFECT_PLATFORM_RUN_ID": run_id or "runtime-handshake",
                        "DEFECT_PLATFORM_IMAGE_DIGEST": image_uri.split("@", 1)[1]},
        "command": ["python", "-m", "defect_platform.trainer.runtime_probe"],
    }
    result = submit_and_wait(payload)
    for field in ("gpu_count", "gpu_name", "gpu_tensor_operation", "python_version",
                  "pytorch_version", "cuda_version", "gcs_read", "gcs_write", "image_digest"):
        if field not in result:
            raise RuntimeValidationError("vertex_gpu", f"handshake result missing {field}")
    if int(result["gpu_count"]) < config.accelerator_count:
        raise RuntimeValidationError("vertex_gpu", "Vertex job exposed fewer GPUs than requested")
    if not result["gpu_tensor_operation"]:
        raise RuntimeValidationError("vertex_gpu", "GPU tensor operation did not complete")
    if result["image_digest"] != image_uri.split("@", 1)[1]:
        raise RuntimeValidationError("vertex_gpu", "handshake ran a different image digest")
    if not result["gcs_read"] or not result["gcs_write"]:
        raise RuntimeValidationError("gcs", "Vertex service account could not complete GCS roundtrip")
    return result


def classify_failure(stage: str, message: str) -> FailureClassification:
    """Map runtime failures to the layer that can make the next useful fix."""
    text = message.lower()
    if stage in {"dataset", "data", "webdataset"}:
        return FailureClassification("DATA_INVALID", "dataset", False, message)
    if stage in {"trainer", "training"} and any(k in text for k in ("out of memory", "cuda", "torch")):
        return FailureClassification("TRAINER_RESOURCE", "trainer", True, message)
    if stage in {"vertex", "vertex_gpu", "gcs"}:
        retryable = any(k in text for k in ("timeout", "unavailable", "temporar", "429", "503"))
        owner = "cloud_configuration" if any(k in text for k in ("permission", "quota", "not found", "denied")) else "gcp_runtime"
        return FailureClassification("GCP_HANDSHAKE", owner, retryable, message)
    if stage in {"mlflow", "registry"}:
        retryable = any(k in text for k in ("timeout", "unavailable", "temporar", "connection reset"))
        return FailureClassification("MLFLOW_REGISTRATION", "mlflow", retryable, message)
    return FailureClassification("UNKNOWN_RUNTIME", "platform", False, message)


def run_synthetic_validation(output_dir: str) -> dict:
    """Exercise optimizer, save, reload, prediction and metrics with a tiny CNN.

    This validates the training image's PyTorch/CUDA path independently of
    downloading DINOv3 weights. It is deterministic and useful in CI and Vertex.
    """
    import json
    from pathlib import Path

    import torch
    from torch import nn
    torch.manual_seed(5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 8), nn.GELU(), nn.Linear(8, 2)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.rand(12, 3, 16, 16, device=device)
    y = torch.tensor([0, 1] * 6, device=device)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(x), y); loss.backward(); optimizer.step()
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "synthetic-validation.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    reloaded = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 8), nn.GELU(), nn.Linear(8, 2)).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    reloaded.load_state_dict(saved["state_dict"]); reloaded.eval()
    with torch.inference_mode():
        logits = reloaded(x)
    result = {"success": logits.shape == (12, 2), "device": device,
              "loss": float(loss.item()), "checkpoint": str(checkpoint)}
    (target / "synthetic-validation.json").write_text(json.dumps(result, indent=2))
    if not result["success"]:
        raise RuntimeError("synthetic trainer validation did not return expected logits")
    return result
