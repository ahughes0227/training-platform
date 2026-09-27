from __future__ import annotations

import subprocess

import pytest

from defect_platform.contracts import VertexJobConfig
from defect_platform.runtime_release import CertificationFailure, RuntimeReleaseRequest, certify_image


def request() -> RuntimeReleaseRequest:
    return RuntimeReleaseRequest(
        runtime_id="runtime-1", runtime_version="1", source_commit="a" * 40,
        image_tag="registry.example/trainer:v1",
        training_base_image_digest="registry.example/base@sha256:" + "b" * 64,
        gcs_probe_uri="gs://artifacts/probe",
        job=VertexJobConfig(project="p", region="us-central1", machine_type="n1-standard-4",
                            accelerator_type="NVIDIA_TESLA_T4", accelerator_count=1,
                            service_account="trainer@example.iam.gserviceaccount.com",
                            staging_uri="gs://artifacts/staging", max_run_hours=1,
                            estimated_hourly_usd=1, max_run_cost_usd=2),
    )


def test_certification_gate_order_and_exact_digest():
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            stdout = "a" * 40 + "\n"
        elif command[:3] == ["git", "status", "--porcelain"]:
            stdout = ""
        elif command[:2] == ["docker", "push"]:
            stdout = "v1: digest: sha256:" + "c" * 64 + " size: 1234"
        else:
            stdout = "ok"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    def handshake(payload):
        assert payload["image_uri"] == "registry.example/trainer@sha256:" + "c" * 64
        return {"gpu_count": 1, "gpu_name": "T4", "gpu_tensor_operation": True,
                "python_version": "3.12", "pytorch_version": "2.5", "cuda_version": "12.4",
                "gcs_read": True, "gcs_write": True, "image_digest": "sha256:" + "c" * 64}

    certified = certify_image(request(), runner=runner, handshake=handshake)
    assert certified.certified
    assert certified.image_digest == "registry.example/trainer@sha256:" + "c" * 64
    assert [command[:2] for command in commands if command[0] == "docker"] == [
        ["docker", "build"], ["docker", "run"], ["docker", "push"]]


def test_failure_stops_before_image_push_or_vertex():
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "a" * 40 + "\n", "")
        if command[:2] == ["docker", "run"]:
            raise subprocess.CalledProcessError(1, command, stderr="CUDA unavailable")
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(CertificationFailure) as error:
        certify_image(request(), runner=runner, handshake=lambda payload: (_ for _ in ()).throw(AssertionError()))
    assert error.value.code == "CONTAINER_VALIDATION_FAILED"
    assert not any(command[:2] == ["docker", "push"] for command in commands)
