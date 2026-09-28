from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from defect_platform.infrastructure_contract import (
    InfrastructureCapabilities,
    load_capabilities_from_env,
    serialize_capabilities,
)


def capabilities(now):
    return InfrastructureCapabilities(capability_id="acceptance-1", deployment_revision="a" * 64,
        project="project", region="us-central1", observed_at=now-timedelta(minutes=1),
        expires_at=now+timedelta(hours=1), readiness="ready", dataset_root_uri="gs://datasets",
        artifact_root_uri="gs://artifacts", trainer_service_account="trainer@project.iam.gserviceaccount.com",
        control_url="https://control.run.app", mlflow_uri="https://mlflow.run.app",
        certified_sources={"repo/trainer@sha256:"+"b"*64: "c"*40},
        evidence_uris=["gs://artifacts/acceptance/vertex-handshake.json"])


def inputs():
    return (SimpleNamespace(project="project", region="us-central1",
                service_account="trainer@project.iam.gserviceaccount.com", staging_uri="gs://artifacts/staging"),
            SimpleNamespace(image_digest="repo/trainer@sha256:"+"b"*64, source_commit="c"*40,
                certified=True, validation=SimpleNamespace(model_dump=lambda: {"gpu": True})),
            SimpleNamespace(root_uri="gs://datasets/valve/ds-1"))


def test_observed_infrastructure_contract_accepts_matching_revision():
    now = datetime.now(UTC)
    capabilities(now).validate_training(*inputs(), now=now)


@pytest.mark.parametrize("field,value", [("region", "europe-west4"), ("project", "other"),
    ("service_account", "attacker@project.iam.gserviceaccount.com"), ("staging_uri", "gs://artifacts-evil/x")])
def test_infrastructure_rejects_unobserved_job_identity(field, value):
    now = datetime.now(UTC)
    job, runtime, dataset = inputs()
    setattr(job, field, value)
    with pytest.raises(ValueError):
        capabilities(now).validate_training(job, runtime, dataset, now=now)


def test_infrastructure_rejects_stale_unready_and_changed_runtime():
    now = datetime.now(UTC)
    value = capabilities(now)
    for current in (now-timedelta(hours=1), now+timedelta(hours=2)):
        with pytest.raises(ValueError):
            value.validate_training(*inputs(), now=current)
    value.readiness = "unready"
    with pytest.raises(ValueError): value.validate_training(*inputs(), now=now)
    value.readiness = "ready"
    job, runtime, dataset = inputs()
    runtime.source_commit = "d" * 40
    with pytest.raises(ValueError, match="runtime revision"):
        value.validate_training(job, runtime, dataset, now=now)


def test_capability_file_is_pinned_and_missing_config_remains_unconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv("DEFECT_INFRA_CAPABILITIES_FILE", raising=False)
    monkeypatch.delenv("DEFECT_INFRA_CAPABILITIES_SHA256", raising=False)
    assert load_capabilities_from_env() is None
    value = capabilities(datetime.now(UTC))
    path = tmp_path / "capabilities.json"
    path.write_bytes(serialize_capabilities(value))
    monkeypatch.setenv("DEFECT_INFRA_CAPABILITIES_FILE", str(path))
    monkeypatch.setenv("DEFECT_INFRA_CAPABILITIES_SHA256", value.sha256)
    assert load_capabilities_from_env() == value
    monkeypatch.setenv("DEFECT_INFRA_CAPABILITIES_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="fingerprint"):
        load_capabilities_from_env()
