"""Ordered, explicit release of one exact GPU trainer image digest."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import typer
import yaml
from pydantic import BaseModel, ConfigDict, Field

from defect_platform.contracts import CertifiedRuntime, ValidationResults, VertexJobConfig
from defect_platform.control.vertex import VertexAdapter
from defect_platform.trainer.runtime import vertex_gpu_handshake


runtime_app = typer.Typer(help="Certify and inspect immutable trainer runtimes.", no_args_is_help=True)


class RuntimeReleaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime_id: str
    runtime_version: str
    source_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    image_tag: str
    training_base_image_digest: str
    job: VertexJobConfig
    gcs_probe_uri: str


class CertificationFailure(RuntimeError):
    def __init__(self, code: str, command: list[str] | None, message: str):
        super().__init__(message)
        self.code = code
        self.command = command


def _run(command: list[str], *, code: str, runner: Callable = subprocess.run) -> str:
    try:
        result = runner(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (getattr(exc, "stderr", None) or str(exc)).strip()
        raise CertificationFailure(code, command, detail[-4000:]) from exc
    return result.stdout


def _source_check(expected_commit: str, *, runner: Callable) -> None:
    actual = _run(["git", "rev-parse", "HEAD"], code="SOURCE_VALIDATION_FAILED", runner=runner).strip()
    if actual != expected_commit:
        raise CertificationFailure("SOURCE_VALIDATION_FAILED", None, "source commit does not match HEAD")
    dirty = _run(["git", "status", "--porcelain"], code="SOURCE_VALIDATION_FAILED", runner=runner)
    if dirty.strip():
        raise CertificationFailure("SOURCE_VALIDATION_FAILED", None, "commit source changes before certification")


def certify_image(
    request: RuntimeReleaseRequest,
    *,
    runner: Callable = subprocess.run,
    handshake: Callable[[dict], dict] | None = None,
) -> CertifiedRuntime:
    """Stop at the first failing gate; experiments never call this path."""
    if "@sha256:" not in request.training_base_image_digest:
        raise CertificationFailure("IMAGE_BUILD_FAILED", None, "training base image must be pinned by digest")
    if not request.gcs_probe_uri.startswith("gs://"):
        raise CertificationFailure("VERTEX_RUNTIME_VALIDATION_FAILED", None, "probe URI must be in GCS")
    _source_check(request.source_commit, runner=runner)
    _run([sys.executable, "-m", "defect_platform.trainer.validate"],
         code="TRAINER_VALIDATION_FAILED", runner=runner)
    _run(["docker", "build", "-f", "infra/docker/trainer.Dockerfile",
          "--build-arg", f"TRAINING_BASE_IMAGE={request.training_base_image_digest}",
          "-t", request.image_tag, "."], code="IMAGE_BUILD_FAILED", runner=runner)
    _run(["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "python",
          request.image_tag, "-m", "defect_platform.trainer.validate", "--require-gpu"],
         code="CONTAINER_VALIDATION_FAILED", runner=runner)
    pushed = _run(["docker", "push", request.image_tag],
                  code="IMAGE_PUSH_FAILED", runner=runner)
    match = re.search(r"\bdigest:\s*(sha256:[a-f0-9]{64})\b", pushed)
    if not match:
        raise CertificationFailure("IMAGE_PUSH_FAILED", ["docker", "push", request.image_tag],
                                   "registry did not return an immutable digest")
    repository = request.image_tag.rsplit(":", 1)[0]
    image_digest = f"{repository}@{match.group(1)}"
    submit_and_wait = handshake or VertexAdapter().submit_handshake_and_wait
    try:
        evidence = vertex_gpu_handshake(
            request.job, image_uri=image_digest, staging_bucket_uri=request.gcs_probe_uri,
            submit_and_wait=submit_and_wait,
            result_uri=request.gcs_probe_uri.rstrip("/") + f"/handshakes/{request.runtime_id}-{match.group(1)[7:19]}.json",
        )
    except Exception as exc:
        raise CertificationFailure("VERTEX_RUNTIME_VALIDATION_FAILED", None, str(exc)) from exc
    return CertifiedRuntime(
        runtime_id=request.runtime_id,
        source_commit=request.source_commit,
        image_tag=request.image_tag,
        image_digest=image_digest,
        runtime_version=request.runtime_version,
        python_version=str(evidence["python_version"]),
        pytorch_version=str(evidence["pytorch_version"]),
        cuda_version=str(evidence["cuda_version"]),
        validation=ValidationResults(trainer=True, container_gpu=True, vertex_gpu=True,
                                     gcs_read=True, gcs_write=True),
        certified=True,
        certified_at=datetime.now(timezone.utc),
    )


@runtime_app.command("certify")
def certify(config: Path, output: Path | None = typer.Option(None, "--output", "-o")) -> None:
    """Validate trainer, GPU container, registry digest, then Vertex handshake."""
    try:
        request = RuntimeReleaseRequest.model_validate(yaml.safe_load(config.read_text()))
        result = certify_image(request)
        destination = output or Path("certifications") / f"{result.runtime_id}.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(result.model_dump(mode="json"), stream, sort_keys=False)
    except CertificationFailure as exc:
        typer.echo(json.dumps({"code": exc.code, "message": str(exc), "command": exc.command}), err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps({"runtime_id": result.runtime_id, "source_commit": result.source_commit,
                           "image_digest": result.image_digest,
                           "validation": result.validation.model_dump(), "certified": result.certified,
                           "record": str(destination)}, indent=2))


@runtime_app.command("show")
def show(config: Path) -> None:
    """Read a previously certified immutable runtime record."""
    runtime = CertifiedRuntime.model_validate(yaml.safe_load(config.read_text()))
    typer.echo(json.dumps(runtime.model_dump(mode="json"), indent=2))
