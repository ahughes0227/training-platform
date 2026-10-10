"""A Vertex goal run with the real trainer entrypoint, against an in-memory GCS.

The fake job client runs ``python -m defect_platform.trainer.runner`` in process
with the job's own arguments, so every gs:// read and write the container makes
(request, dataset, shards, weights, outputs) goes through the same code paths.
The container has no gsutil, so nothing may shell out to it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

for module in ("torch", "transformers", "webdataset", "PIL", "google.cloud.aiplatform_v1"):
    pytest.importorskip(module)


class _Blob:
    def __init__(self, store: dict, bucket: str, name: str):
        self.store, self.bucket_name, self.name = store, bucket, name
        self.metadata = None

    @property
    def _key(self):
        return (self.bucket_name, self.name)

    def exists(self, client=None):
        return self._key in self.store

    def reload(self, client=None):
        self.metadata = self.store[self._key][1]

    def _put(self, data: bytes, if_generation_match=None):
        from google.api_core.exceptions import PreconditionFailed

        if if_generation_match == 0 and self.exists():
            raise PreconditionFailed("exists")
        self.store[self._key] = (data, self.metadata)

    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        self._put(data.encode() if isinstance(data, str) else data, if_generation_match)

    def upload_from_filename(self, filename, content_type=None, if_generation_match=None):
        self._put(Path(filename).read_bytes(), if_generation_match)

    def download_as_bytes(self):
        if not self.exists():
            from google.api_core.exceptions import NotFound

            raise NotFound(self.name)
        return self.store[self._key][0]

    def download_as_text(self):
        return self.download_as_bytes().decode()

    def download_to_filename(self, filename):
        Path(filename).write_bytes(self.download_as_bytes())

    def open(self, mode="rb"):
        import io

        return io.BytesIO(self.download_as_bytes())


class _Client:
    store: ClassVar[dict] = {}

    def __init__(self, *args, **kwargs):
        pass

    def bucket(self, name):
        return SimpleNamespace(name=name, blob=lambda key: _Blob(self.store, name, key))

    def list_blobs(self, bucket, prefix=""):
        return [_Blob(self.store, b, key) for (b, key) in sorted(self.store)
                if b == bucket and key.startswith(prefix)]


@pytest.fixture
def fake_gcs(monkeypatch):
    import google.cloud

    _Client.store = {}
    storage = ModuleType("google.cloud.storage")
    storage.Client = _Client
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage)
    monkeypatch.setattr(google.cloud, "storage", storage, raising=False)
    # The trainer image has no gsutil; fail loudly if anything tries to use it.
    monkeypatch.setenv("PATH", "/nonexistent")
    return _Client.store


class _RunnerJobs:
    """A Vertex job client whose jobs run the trainer entrypoint right away."""

    def __init__(self):
        self.state = "JOB_STATE_PENDING"

    def list_custom_jobs(self, request):
        return []

    def create_custom_job(self, parent, custom_job):
        from defect_platform.trainer import runner

        container = custom_job.job_spec.worker_pool_specs[0].container_spec
        assert list(container.command) == ["python", "-m", "defect_platform.trainer.runner"]
        argv = sys.argv
        sys.argv = ["runner", *container.args]
        try:
            runner.main()
            self.state = "JOB_STATE_SUCCEEDED"
        finally:
            sys.argv = argv
        return SimpleNamespace(name="projects/p/locations/us-central1/customJobs/1")

    def get_custom_job(self, name):
        return SimpleNamespace(state=SimpleNamespace(name=self.state),
                               error=SimpleNamespace(message=""))


def test_demo_baseline_trains_from_gcs_like_the_vertex_container(tmp_path, monkeypatch, fake_gcs):
    from defect_platform.control.goal_commands import load_goal_file
    from defect_platform.control.goal_demo import write_demo
    from defect_platform.control.goal_vertex import VertexExecutor, VertexGoalConfig, attempt_key
    from defect_platform.dataset import load_dataset_semantics

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    goal_file = write_demo(tmp_path / "demo", gcs_prefix="gs://bucket/goals/demo")
    goal, experiment, dataset = load_goal_file(goal_file)
    assert all(uri.startswith("gs://") for uris in dataset.shard_uris.values() for uri in uris)
    config = VertexGoalConfig(project="p", region="us-central1", service_account="sa@p.iam",
                              staging_uri="gs://bucket/goals",
                              image_digest="r/defect-trainer@sha256:" + "d" * 64,
                              allow_uncertified_image=True, poll_seconds=1)
    classes = load_dataset_semantics(dataset).catalog.labels
    executor = VertexExecutor(config, dataset, classes, attempt=attempt_key(goal.sha256, dataset),
                              client=_RunnerJobs(), sleep=lambda _: None)

    report = executor.run(experiment.model_copy(update={"experiment_id": "demo-run01"}),
                          tmp_path / "run")

    assert report["vertex_job_name"].endswith("customJobs/1")
    assert (tmp_path / "run" / "evaluation.json").exists()
    assert (tmp_path / "run" / "model" / "model.pt").exists()
