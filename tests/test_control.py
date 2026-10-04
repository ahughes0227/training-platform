from __future__ import annotations

from datetime import UTC, datetime

import pytest
from PIL import Image

from defect_platform.contracts import (
    CertifiedRuntime,
    DatasetSpec,
    DatasetVersion,
    ExperimentConfig,
    LabelSource,
    ModelSpec,
    ObjectSpec,
    OperatorProfile,
    RunState,
    SourceKind,
    ValidationResults,
    VertexJobConfig,
)
from defect_platform.control.controller import RunController, VertexAiplatformSubmitter
from defect_platform.control.store import SQLiteRunStore
from defect_platform.dataset import LabelRow, build_dataset
from defect_platform.semantics import ClassCatalog, ClassDefinition, fingerprint


def fixtures(tmp_path):
    catalog_spec = ClassCatalog(
        catalog_id="panel-v1",
        object_slug="panel",
        version=1,
        review_status="reviewed",
        reviewed_by="fixture-reviewer",
        reviewed_at=datetime.now(UTC),
        classes=[
            ClassDefinition(class_id="crack", label="crack", definition="A crack"),
            ClassDefinition(class_id="dent", label="dent", definition="A dent"),
        ],
    )
    object_spec = ObjectSpec(
        slug="panel", display_name="Panel", classes=["crack", "dent"], class_catalog=catalog_spec
    )
    rows = []
    for index in range(12):
        image = Image.new("RGB", (24, 24))
        image.putdata(
            [
                (
                    (index * 37 + x * 17 + y * 7) % 256,
                    (index * 11 + x * 23 + y * 19) % 256,
                    (index * 29 + x * 13 + y * 31) % 256,
                )
                for y in range(24)
                for x in range(24)
            ]
        )
        path = tmp_path / f"sample-{index}.png"
        image.save(path)
        rows.append(
            LabelRow(
                str(path), object_spec.classes[index % 2], source="fixture", row_number=index + 1
            )
        )
    dataset = build_dataset(
        DatasetSpec(
            object_slug="panel",
            sources=[LabelSource(kind=SourceKind.CSV, location="fixture.csv")],
            output_uri=str(tmp_path / "datasets"),
        ),
        object_spec,
        rows=rows,
        near_duplicate_distance=0,
    )
    digest = "sha256:" + "a" * 64
    runtime = CertifiedRuntime(
        runtime_id="runtime-1",
        source_commit="a" * 40,
        image_tag="train:v1",
        image_digest="us-docker.pkg.dev/p/r/train@" + digest,
        runtime_version="1",
        python_version="3.12",
        pytorch_version="2.6",
        cuda_version="12.4",
        validation=ValidationResults(
            trainer=True, container_gpu=True, vertex_gpu=True, gcs_read=True, gcs_write=True
        ),
        certified=True,
        certified_at=datetime.now(UTC),
    )
    experiment = ExperimentConfig(
        experiment_id="exp-1",
        object_slug="panel",
        dataset_version_id=dataset.version_id,
        runtime_id="runtime-1",
        model=ModelSpec(weights_uri="gs://weights/dino.safetensors", weights_sha256="b" * 64),
    )
    job = VertexJobConfig(
        project="project",
        region="us-central1",
        machine_type="g2-standard-8",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        service_account="train@project.iam.gserviceaccount.com",
        staging_uri="gs://bucket/staging",
        max_run_hours=2,
        estimated_hourly_usd=10,
        max_run_cost_usd=25,
    )

    class Catalog:
        def get_certified(self, runtime_id: str) -> CertifiedRuntime | None:
            assert runtime_id == runtime.runtime_id
            return runtime

        def get(self, version_id: str) -> DatasetVersion | None:
            assert version_id == dataset.version_id
            return dataset

    class TrustedCatalogs:
        def get_catalog(self, object_slug: str, catalog_sha256: str):
            assert object_slug == catalog_spec.object_slug
            return catalog_spec if catalog_sha256 == catalog_spec.sha256 else None

    return runtime, experiment, dataset, job, Catalog(), TrustedCatalogs()


class FakeWorkflow:
    def __init__(self, store):
        self.store, self.calls = store, []

    def start(self, *, run_id, payload):
        assert self.store.get(run_id) is not None
        self.calls.append((run_id, payload))
        return f"execution/{run_id}"


class FakeVertex:
    def __init__(self):
        self.calls = []

    def submit(self, **kwargs):
        self.calls.append(kwargs)
        return f"customJobs/{kwargs['run_id']}"


def test_submit_persists_before_start_and_idempotent_retry_starts_once(tmp_path):
    runtime, experiment, _dataset, job, catalog, catalogs = fixtures(tmp_path)
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    workflow = FakeWorkflow(store)
    controller = RunController(
        store=store, runtimes=catalog, datasets=catalog, workflows=workflow, catalogs=catalogs
    )

    run, created = controller.submit(
        experiment=experiment, job=job, idempotency_key="request-1", classes=["crack", "dent"]
    )
    retry, retry_created = controller.submit(
        experiment=experiment, job=job, idempotency_key="request-1", classes=["crack", "dent"]
    )

    assert created and not retry_created
    assert retry.run_id == run.run_id
    assert run.state == RunState.SUBMITTED
    assert run.output_uri == f"gs://bucket/staging/runs/{run.run_id}"
    assert len(workflow.calls) == 1
    assert run.runtime_image_digest == runtime.image_digest
    assert run.runtime_source_commit == runtime.source_commit
    assert retry.runtime_image_digest == runtime.image_digest
    payload = store.get_payload(run.run_id)
    assert payload is not None
    assert payload["runtime"]["image_digest"] == runtime.image_digest


def test_preflight_blocks_over_budget_before_creating_run(tmp_path):
    _runtime, experiment, _dataset, job, catalog, catalogs = fixtures(tmp_path)
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    controller = RunController(
        store=store, runtimes=catalog, datasets=catalog, catalogs=catalogs, vertex=FakeVertex()
    )
    expensive = job.model_copy(update={"max_run_cost_usd": 19})

    with pytest.raises(ValueError, match="exceeds configured run cap"):
        controller.submit(experiment=experiment, job=expensive, idempotency_key="too-expensive")
    assert store.list() == []


def test_normal_experiment_reuses_certified_digest_and_failure_has_owner(tmp_path):
    runtime, experiment, _dataset, job, catalog, catalogs = fixtures(tmp_path)
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    vertex = FakeVertex()
    controller = RunController(
        store=store, runtimes=catalog, datasets=catalog, vertex=vertex, catalogs=catalogs
    )

    run, _ = controller.submit(
        experiment=experiment, job=job, idempotency_key="direct-run", classes=["crack", "dent"]
    )

    assert vertex.calls[0]["runtime"].image_digest == runtime.image_digest
    assert run.state == RunState.SUBMITTED
    with pytest.raises(ValueError, match="invalid run transition"):
        controller.update_state(run.run_id, RunState.PENDING)
    failed = controller.update_state(
        run.run_id, RunState.FAILED, failure=RuntimeError("accelerator quota exceeded")
    )
    assert failed.failure_code == "VERTEX_CAPACITY"
    assert failed.state == RunState.FAILED


def test_idempotency_key_rejects_changed_config(tmp_path):
    _runtime, experiment, _dataset, job, catalog, catalogs = fixtures(tmp_path)
    controller = RunController(
        store=SQLiteRunStore(tmp_path / "runs.sqlite"),
        runtimes=catalog,
        datasets=catalog,
        vertex=FakeVertex(),
        catalogs=catalogs,
    )
    controller.submit(
        experiment=experiment, job=job, idempotency_key="same-key", classes=["crack", "dent"]
    )
    changed = experiment.model_copy(update={"learning_rate": 0.002})
    with pytest.raises(ValueError, match="different request"):
        controller.submit(
            experiment=changed, job=job, idempotency_key="same-key", classes=["crack", "dent"]
        )


def test_vertex_adapter_waits_and_validates_gcs_handshake_result():
    class Job:
        state = "JOB_STATE_SUCCEEDED"

        def __init__(self, **kwargs):
            self.spec = kwargs["worker_pool_specs"]

        def run(self, **kwargs):
            self.run_args = kwargs

    class Blob:
        def download_as_text(self):
            return (
                '{"gpu_count":1,"gcs_read":true,"gcs_write":true,"gpu_tensor_operation":true,"image_digest":"sha256:'
                + "a" * 64
                + '","cuda":"12.4"}'
            )

    class Bucket:
        def blob(self, name):
            assert name == "results/result.json"
            return Blob()

    class Storage:
        def bucket(self, name):
            assert name == "probe-bucket"
            return Bucket()

    from defect_platform.control.vertex import VertexAdapter

    payload = {
        "project": "p",
        "region": "us-central1",
        "machine_type": "g2-standard-8",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 1,
        "service_account": "sa",
        "staging_uri": "gs://bucket/staging",
        "image_uri": "img@sha256:" + "a" * 64,
        "gcs_probe_uri": "gs://probe-bucket/probe",
        "timeout_seconds": 120,
        "environment": {
            "DEFECT_PLATFORM_GCS_RESULT_URI": "gs://probe-bucket/results/result.json",
            "DEFECT_PLATFORM_IMAGE_DIGEST": "sha256:" + "a" * 64,
        },
        "command": ["python", "-m", "defect_platform.trainer.runtime_probe"],
    }
    adapter = VertexAdapter(custom_job_factory=Job, storage_client=Storage())
    result = adapter.submit_handshake_and_wait(payload)
    assert result["cuda"] == "12.4"


def test_vertex_submission_returns_server_resource_and_recovers_retry(tmp_path, monkeypatch):
    runtime, experiment, dataset, job, _, _ = fixtures(tmp_path)
    from defect_platform.control import controller as module

    uploads = []
    monkeypatch.setattr(
        module,
        "_upload_vertex_request",
        lambda *args: uploads.append(args) or "gs://bucket/request.json",
    )

    class Client:
        created = None
        calls = 0

        def list_custom_jobs(self, request):
            return [self.created] if self.created else []

        def create_custom_job(self, parent, custom_job):
            self.calls += 1
            assert (
                custom_job.job_spec.worker_pool_specs[0].container_spec.image_uri
                == runtime.image_digest
            )
            assert custom_job.job_spec.scheduling.timeout.seconds == int(job.max_run_hours * 3600)
            self.created = custom_job
            self.created.name = parent + "/customJobs/123"
            return self.created

    client = Client()
    submitter = VertexAiplatformSubmitter(client_factory=lambda location: client)
    assert dataset.semantic_sha256 is not None
    dataset_semantic_sha256 = dataset.semantic_sha256

    def submit_once() -> str:
        return submitter.submit(
            run_id="run-1",
            experiment=experiment,
            dataset=dataset,
            runtime=runtime,
            job=job,
            output_uri="gs://bucket/runs/run-1",
            classes=["crack", "dent"],
            semantic_refs={
                "dataset_semantic_sha256": dataset_semantic_sha256,
                "catalog_sha256": "e" * 64,
            },
        )

    assert submit_once() == "projects/project/locations/us-central1/customJobs/123"
    assert submit_once() == "projects/project/locations/us-central1/customJobs/123"
    assert client.calls == 1
    assert len(uploads) == 1
    assert uploads[0][2]["semantic_refs"] == {
        "dataset_semantic_sha256": dataset_semantic_sha256,
        "catalog_sha256": "e" * 64,
    }
    assert uploads[0][2]["runtime"]["image_digest"] == runtime.image_digest
    assert uploads[0][2]["runtime"]["source_commit"] == runtime.source_commit


def test_control_http_submission_and_workflow_callbacks(tmp_path):
    from fastapi.testclient import TestClient

    from defect_platform.control.api import create_app

    runtime, experiment, dataset, job, catalog, catalogs = fixtures(tmp_path)
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    workflow = FakeWorkflow(store)
    controller = RunController(
        store=store,
        runtimes=catalog,
        datasets=catalog,
        workflows=workflow,
        vertex=FakeVertex(),
        catalogs=catalogs,
    )
    client = TestClient(create_app(controller))
    schema = client.get("/schemas/urn:defect-platform:intake-patch:v1")
    assert schema.status_code == 200
    assert schema.json()["schema_id"] == "urn:defect-platform:intake-patch:v1"
    body = {
        "experiment": experiment.model_dump(mode="json"),
        "job": job.model_dump(mode="json"),
        "dataset": dataset.model_dump(mode="json"),
        "runtime": runtime.model_dump(mode="json"),
        "classes": ["crack", "dent"],
        "idempotency_key": "http-1",
    }
    response = client.post("/runs", json=body)
    assert response.status_code == 202, response.text
    run_id = response.json()["run"]["run_id"]
    assert client.post("/runs", json=body).json()["created"] is False
    submitted = client.post(f"/runs/{run_id}/vertex-submit")
    assert submitted.status_code == 200
    name = submitted.json()["vertex_job_name"]
    running = client.post(
        f"/runs/{run_id}/events", json={"state": "running", "vertex_job_name": name}
    )
    assert running.status_code == 200, running.text
    done = client.post(
        f"/runs/{run_id}/events", json={"state": "succeeded", "vertex_job_name": name}
    )
    assert done.status_code == 200, done.text
    assert client.get(f"/runs/{run_id}").json()["state"] == "succeeded"


def test_reference_submission_rejects_expired_operator_profile(tmp_path):
    from fastapi.testclient import TestClient

    from defect_platform.catalog_store import DirectoryReferenceCatalogStore
    from defect_platform.control.api import create_app

    runtime, experiment, dataset, job, catalog, catalogs = fixtures(tmp_path)
    profile = OperatorProfile(
        profile_id="operator",
        version="1",
        display_name="Expired operator",
        owner="ops",
        expires_at=datetime.now(UTC),
    )
    references = DirectoryReferenceCatalogStore(tmp_path / "references")
    references.publish_reference("dataset_version", dataset.version_id, "1", dataset)
    references.publish_reference("certified_runtime", runtime.runtime_id, "1", runtime)
    profile_digest = fingerprint(profile.model_dump(mode="json"))
    references.publish_reference("operator_profile", profile.profile_id, profile.version, profile)

    controller = RunController(
        store=SQLiteRunStore(tmp_path / "runs.sqlite"),
        runtimes=catalog,
        datasets=catalog,
        vertex=FakeVertex(),
        catalogs=catalogs,
    )
    client = TestClient(create_app(controller, reference_store=references))
    body = {
        "experiment": experiment.model_dump(mode="json"),
        "job": job.model_dump(mode="json"),
        "dataset_ref": {
            "kind": "dataset_version",
            "id": dataset.version_id,
            "sha256": dataset.sha256,
        },
        "runtime_ref": {
            "kind": "certified_runtime",
            "id": runtime.runtime_id,
            "sha256": fingerprint(runtime.model_dump(mode="json")),
        },
        "operator_profile_ref": {
            "id": profile.profile_id,
            "version": profile.version,
            "sha256": profile_digest,
        },
        "idempotency_key": "expired-profile",
        "classes": ["crack", "dent"],
    }
    response = client.post("/runs/reference", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "OPERATOR_PROFILE_EXPIRED"
