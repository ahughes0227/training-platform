from __future__ import annotations

from datetime import UTC, datetime

import pytest

from defect_platform.contracts import RunRecord, RunState
from defect_platform.control.controller import MLflowTracker


def test_raw_bundle_registers_once_with_matching_run_id(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = "sqlite:///" + str(tmp_path / "tracking.sqlite3")
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(
        "defect-classification", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    with mlflow.start_run(
        experiment_id=mlflow.get_experiment_by_name("defect-classification").experiment_id
    ) as active:
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "model.pt").write_bytes(b"fixture")
        mlflow.log_artifacts(str(bundle), artifact_path="model")
    run = RunRecord(
        run_id="platform-run-1",
        object_slug="fixture",
        experiment_id="exp-1",
        dataset_version_id="ds-1",
        runtime_id="rt-1",
        state=RunState.SUCCEEDED,
        created_at=datetime.now(UTC),
        mlflow_run_id=active.info.run_id,
    )
    tracker = MLflowTracker(uri)
    first = tracker.register_candidate(run)
    second = tracker.register_candidate(run)
    assert first == second == "models:/defect-fixture/1"
    version = mlflow.MlflowClient().get_model_version("defect-fixture", "1")
    assert version.run_id == active.info.run_id
    assert version.source.endswith("/model")
