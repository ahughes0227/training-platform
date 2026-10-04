from datetime import UTC, datetime

from defect_platform.contracts import RunRecord, RunState
from defect_platform.control.batch import expand_matrix
from defect_platform.control.store import SQLiteRunStore


def test_expand_matrix_is_deterministic_and_preserves_value_order():
    matrix = {"job.machine_type": ["small", "large"], "experiment.epochs": [1, 2]}
    assert expand_matrix(matrix) == [
        {"experiment.epochs": 1, "job.machine_type": "small"},
        {"experiment.epochs": 1, "job.machine_type": "large"},
        {"experiment.epochs": 2, "job.machine_type": "small"},
        {"experiment.epochs": 2, "job.machine_type": "large"},
    ]


def test_sqlite_events_are_idempotent_and_cursored(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    run = RunRecord(
        run_id="run-1",
        object_slug="panel",
        experiment_id="exp-1",
        dataset_version_id="data-1",
        runtime_id="runtime-1",
        state=RunState.PENDING,
        created_at=datetime.now(UTC),
    )
    store.create(run, "fingerprint", {"idempotency_key": "request-1"})
    first = store.append_event(run.run_id, "run.created", {"state": "pending"}, event_id="event-1")
    duplicate = store.append_event(run.run_id, "run.created", {"ignored": True}, event_id="event-1")
    store.append_event(run.run_id, "run.state", {"state": "running"}, event_id="event-2")
    events, cursor = store.read_events(run.run_id, cursor=1, limit=1)
    assert first.sequence == duplicate.sequence == 2
    assert events[0].event_id == "event-1"
    assert cursor == 2
    status = store.compact(run.run_id)
    assert status is not None
    assert status["state"] == "pending"
