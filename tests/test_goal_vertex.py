"""Goal runs on Vertex AI: the CustomJob spec, waiting, resuming and failures."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("google.cloud.aiplatform_v1")

from defect_platform.contracts import (
    CertifiedRuntime,
    DatasetVersion,
    ExperimentConfig,
    ModelSpec,
    ValidationResults,
)
from defect_platform.control.goal_vertex import VertexExecutor, VertexGoalConfig

DIGEST = "us-central1-docker.pkg.dev/p/r/defect-trainer@sha256:" + "d" * 64


def _config(**updates) -> VertexGoalConfig:
    values = {"project": "p", "region": "us-central1", "service_account": "trainer@p.iam",
              "staging_uri": "gs://bucket/goals", "image_digest": DIGEST,
              "allow_uncertified_image": True, "poll_seconds": 1, **updates}
    return VertexGoalConfig.model_validate(values)


def _dataset(root: str = "gs://bucket/datasets/v1") -> DatasetVersion:
    return DatasetVersion(version_id="v1", object_slug="panel", root_uri=root,
                          manifest_uri=f"{root}/manifest.json", shard_uris={},
                          sample_counts={"train": 1, "validation": 1, "test": 1}, sha256="c" * 64,
                          source_snapshot_uri=f"{root}/source.json")


def _experiment(weights: str = "gs://bucket/backbone") -> ExperimentConfig:
    return ExperimentConfig(experiment_id="panel-run01", object_slug="panel",
                            dataset_version_id="v1", runtime_id="dev",
                            model=ModelSpec(weights_uri=weights, weights_sha256="a" * 64))


class FakeJobs:
    def __init__(self, states, existing=()):
        self.states = list(states)
        self.existing = list(existing)
        self.created = []

    def list_custom_jobs(self, request):
        return self.existing

    def create_custom_job(self, parent, custom_job):
        self.created.append((parent, custom_job))
        return SimpleNamespace(name="projects/p/locations/us-central1/customJobs/1")

    def get_custom_job(self, name):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return SimpleNamespace(state=SimpleNamespace(name=state),
                               error=SimpleNamespace(message="out of memory"))


def _executor(jobs, config=None, uploads=None):
    uploads = uploads if uploads is not None else []

    def upload(staging, run_id, request):
        uploads.append(request)
        return f"{staging}/requests/{run_id}.json"

    def download(uri, destination):
        (destination / "model").mkdir(parents=True, exist_ok=True)
        (destination / "evaluation.json").write_text(json.dumps({"output_uri": uri}))

    return VertexExecutor(config or _config(), _dataset(), ["ok", "scratch"], attempt="a1",
                          client=jobs,
                          sleep=lambda _: None, upload_request=upload, download=download,
                          semantic_refs={"dataset_semantic_sha256": "e" * 64,
                                         "catalog_sha256": "f" * 64})


def test_cpu_job_runs_waits_and_copies_results_back(tmp_path):
    jobs = FakeJobs(["JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED"])
    uploads = []
    executor = _executor(jobs, uploads=uploads)
    report = executor.run(_experiment(), tmp_path / "run")

    assert report["output_uri"] == "gs://bucket/goals/a1/runs/panel-run01"
    assert report["vertex_job_name"].endswith("customJobs/1")
    assert (tmp_path / "run" / "evaluation.json").exists()
    spec = jobs.created[0][1].job_spec.worker_pool_specs[0]
    assert spec.machine_spec.machine_type == "n1-standard-4"
    assert not spec.machine_spec.accelerator_count
    assert spec.container_spec.image_uri == DIGEST
    # An uncertified development image never claims a runtime in the request.
    assert "runtime" not in uploads[0]
    assert executor.releasable is False
    assert "uncertified" in executor.description


def test_certified_runtime_is_sent_with_semantic_refs(tmp_path):
    runtime = CertifiedRuntime(
        runtime_id="rt-1", source_commit="a" * 40, image_tag="trainer:v1", image_digest=DIGEST,
        runtime_version="1", python_version="3.12", pytorch_version="2.6", cuda_version="12.4",
        validation=ValidationResults(trainer=True, container_gpu=True, vertex_gpu=True,
                                     gcs_read=True, gcs_write=True),
        certified=True, certified_at=datetime.now(UTC))
    uploads = []
    executor = _executor(FakeJobs(["JOB_STATE_SUCCEEDED"]),
                         _config(runtime=runtime.model_dump(mode="json"),
                                 allow_uncertified_image=False,
                                 accelerator_type="NVIDIA_L4", accelerator_count=1),
                         uploads=uploads)
    executor.run(_experiment(), tmp_path / "run")
    assert uploads[0]["runtime"]["runtime_id"] == "rt-1"
    assert uploads[0]["semantic_refs"]["catalog_sha256"] == "f" * 64
    assert executor.releasable is True


def test_a_failed_job_raises_with_its_vertex_error(tmp_path):
    executor = _executor(FakeJobs(["JOB_STATE_FAILED"]))
    with pytest.raises(RuntimeError, match="JOB_STATE_FAILED: out of memory"):
        executor.run(_experiment(), tmp_path / "run")


def test_a_resumed_goal_reattaches_to_its_running_job(tmp_path):
    live = SimpleNamespace(name="projects/p/locations/us-central1/customJobs/9",
                           display_name="defect-goal-panel-run01-a1",
                           state=SimpleNamespace(name="JOB_STATE_RUNNING"))
    jobs = FakeJobs(["JOB_STATE_SUCCEEDED"], existing=[live])
    report = _executor(jobs).run(_experiment(), tmp_path / "run")
    assert jobs.created == []
    assert report["vertex_job_name"].endswith("customJobs/9")


def test_vertex_runs_need_data_and_weights_in_gcs(tmp_path):
    with pytest.raises(ValueError, match="dataset in GCS"):
        VertexExecutor(_config(), _dataset("/local/v1"), ["ok", "scratch"], attempt="a1")
    with pytest.raises(ValueError, match="weights in GCS"):
        _executor(FakeJobs(["JOB_STATE_SUCCEEDED"])).run(_experiment("/local/w"), Path(tmp_path))


def test_config_requires_certification_or_an_explicit_development_image():
    with pytest.raises(ValueError, match="certified runtime record"):
        _config(allow_uncertified_image=False)
    with pytest.raises(ValueError, match="set together"):
        _config(accelerator_type="NVIDIA_T4")
    with pytest.raises(ValueError):
        _config(image_digest="us-docker.pkg.dev/p/r/trainer:latest")


def test_each_goal_attempt_stages_and_names_its_jobs_apart(tmp_path):
    from defect_platform.control.goal_vertex import attempt_key

    first, again = attempt_key("1" * 64, _dataset()), attempt_key("1" * 64, _dataset())
    rebuilt = attempt_key("1" * 64, _dataset().model_copy(update={"sha256": "9" * 64}))
    assert first == again != rebuilt

    # A finished job from an earlier attempt with the same run id is not reused.
    old = SimpleNamespace(name="projects/p/locations/us-central1/customJobs/3",
                          display_name="defect-goal-panel-run01-a0",
                          state=SimpleNamespace(name="JOB_STATE_SUCCEEDED"))
    jobs = FakeJobs(["JOB_STATE_SUCCEEDED"], existing=[old])
    uploads = []
    report = _executor(jobs, uploads=uploads).run(_experiment(), tmp_path / "run")
    assert report["vertex_job_name"].endswith("customJobs/1")
    custom_job = jobs.created[0][1]
    assert custom_job.display_name == "defect-goal-panel-run01-a1"
    assert custom_job.labels["defect-goal-attempt"] == "a1"
    assert uploads[0]["output_uri"].startswith("gs://bucket/goals/a1/")
