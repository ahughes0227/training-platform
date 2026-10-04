from __future__ import annotations

from datetime import datetime, timezone

import yaml
from typer.testing import CliRunner

from defect_platform import cli
from defect_platform.catalog_store import DirectoryCatalogStore
from defect_platform.contracts import (
    CertifiedRuntime,
    DatasetVersion,
    RunRecord,
    RunState,
    ValidationResults,
    VertexJobConfig,
)
from defect_platform.control.setup import SetupDraft
from defect_platform.semantics import ClassCatalog, ClassDefinition


def test_guided_setup_builds_project_and_submits_to_durable_control_api(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEFECT_CONTROL_SERVICE_URL", "http://localhost:8080")
    monkeypatch.setenv("DEFECT_CLASS_CATALOG_ROOT", str(tmp_path / "catalogs"))
    monkeypatch.setenv("DEFECT_LITELLM_MODEL", "fixture-model")
    monkeypatch.setenv("DEFECT_DATASET_OUTPUT_URI", str(tmp_path / "datasets"))
    monkeypatch.setenv("DEFECT_DINOV3_WEIGHTS_URI", "gs://weights/dino")
    monkeypatch.setenv("DEFECT_DINOV3_WEIGHTS_SHA256", "a" * 64)
    runtime = CertifiedRuntime(
        runtime_id="runtime-1",
        source_commit="b" * 40,
        image_tag="trainer:v1",
        image_digest="trainer@sha256:" + "c" * 64,
        runtime_version="1",
        python_version="3.13",
        pytorch_version="2.5",
        cuda_version="12.4",
        validation=ValidationResults(
            trainer=True, container_gpu=True, vertex_gpu=True, gcs_read=True, gcs_write=True
        ),
        certified=True,
        certified_at=datetime.now(timezone.utc),
    )
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(yaml.safe_dump(runtime.model_dump(mode="json")))
    monkeypatch.setenv("DEFECT_DEFAULT_RUNTIME_RECORD", str(runtime_path))
    job = VertexJobConfig(
        project="project",
        region="us-central1",
        machine_type="g2-standard-8",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        service_account="trainer@example.com",
        staging_uri="gs://artifacts/staging",
        max_run_hours=1,
        estimated_hourly_usd=1,
        max_run_cost_usd=2,
    )
    job_path = tmp_path / "vertex.yaml"
    job_path.write_text(yaml.safe_dump(job.model_dump(mode="json")))
    monkeypatch.setenv("DEFECT_DEFAULT_VERTEX_CONFIG", str(job_path))
    catalog = ClassCatalog(
        catalog_id="valve-v1",
        object_slug="valve",
        review_status="reviewed",
        reviewed_by="fixture-reviewer",
        reviewed_at=datetime.now(timezone.utc),
        classes=[
            ClassDefinition(class_id="crack", label="crack", definition="A crack"),
            ClassDefinition(class_id="dent", label="dent", definition="A dent"),
        ],
    )
    DirectoryCatalogStore(tmp_path / "catalogs").publish(catalog)
    draft = SetupDraft.model_validate(
        {
            "object": {
                "slug": "valve",
                "display_name": "Valve",
                "classes": ["crack", "dent"],
                "class_catalog": catalog.model_dump(mode="json"),
            },
            "sources": [{"kind": "csv", "location": "labels.csv"}],
            "experiment": {"epochs": 2},
        }
    )
    monkeypatch.setattr(cli, "propose_setup", lambda *args, **kwargs: draft)
    dataset = DatasetVersion(
        version_id="ds-1",
        object_slug="valve",
        root_uri="gs://datasets/valve/ds-1",
        manifest_uri="gs://datasets/valve/ds-1/manifest.json",
        shard_uris={"train": [], "validation": [], "test": []},
        sample_counts={},
        sha256="d" * 64,
        source_snapshot_uri="gs://datasets/valve/ds-1/source-snapshot.json",
    )

    def fake_build(config, object_config, output, interactive_review):
        assert interactive_review
        output.write_text(yaml.safe_dump(dataset.model_dump(mode="json")))

    monkeypatch.setattr(cli, "dataset_build", fake_build)

    class FakeClient:
        def __init__(self, url):
            assert url == "http://localhost:8080"

        def submit(self, payload):
            assert payload["classes"] == ["crack", "dent"]
            assert payload["experiment"]["dataset_version_id"] == "ds-1"
            assert payload["experiment"]["catalog_sha256"] == catalog.sha256
            return RunRecord(
                run_id="run-1",
                object_slug="valve",
                experiment_id="exp-1",
                dataset_version_id="ds-1",
                runtime_id="runtime-1",
                state=RunState.SUBMITTED,
                created_at=datetime.now(timezone.utc),
                output_uri="gs://artifacts/runs/run-1",
            ), True

    monkeypatch.setattr(cli, "ControlAPIClient", FakeClient)
    result = CliRunner().invoke(cli.app, ["train", "guided", "Valve with labeled crop CSV"])
    assert result.exit_code == 0, result.output
    assert "Run ID: run-1" in result.output
    assert (tmp_path / "projects/valve/run-request.yaml").exists()
    assert (tmp_path / "projects/valve/runs/run-1.yaml").exists()
