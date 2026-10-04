"""Bounded property tests for the control-plane invariants.

These tests intentionally exercise the small, deterministic control helpers rather
than making model or cloud calls.  Hypothesis is an optional development
dependency in this checkout, so the module is skipped when it is not installed.
"""

from __future__ import annotations

import hashlib
import itertools
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from defect_platform.catalog_store import DirectoryReferenceCatalogStore
from defect_platform.contracts import Preset, RunRecord, RunState
from defect_platform.control.batch import expand_matrix
from defect_platform.control.setup import (
    apply_json_patch,
    new_intake_session,
)
from defect_platform.control.store import (
    IntakeSessionConflictError,
    SQLiteIntakeSessionStore,
    SQLiteRunStore,
)
from defect_platform.semantics import fingerprint


@settings(max_examples=40, deadline=None)
@given(note=st.text(max_size=80), marker=st.text(max_size=80))
def test_json_patch_is_atomic_when_a_later_operation_fails(note: str, marker: str):
    """A failed RFC 6902 sequence must not leak its successful prefix."""

    original = {"notes": "before", "experiment": {"marker": "old"}}
    patch = [
        {"op": "replace", "path": "/notes", "value": note},
        {"op": "replace", "path": "/experiment/marker", "value": marker},
        # This operation is guaranteed to fail after the two valid operations.
        {"op": "test", "path": "/notes", "value": object()},
    ]

    with pytest.raises(ValueError):
        apply_json_patch(original, patch)
    assert original == {"notes": "before", "experiment": {"marker": "old"}}


@settings(max_examples=40, deadline=None)
@given(value=st.text(max_size=80))
def test_json_patch_replace_round_trips_generated_values(value: str):
    document = {"notes": "old"}
    result = apply_json_patch(document, [{"op": "replace", "path": "/notes", "value": value}])
    assert result == {"notes": value}
    assert document == {"notes": "old"}


@settings(max_examples=30, deadline=None)
@given(ttl_seconds=st.integers(min_value=1, max_value=300))
def test_intake_revision_fence_rejects_stale_writes(ttl_seconds: int):
    """A session accepts one revision and rejects the same revision twice."""

    now = datetime.now(UTC)
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteIntakeSessionStore(Path(directory) / "intake.sqlite")
        session = new_intake_session(f"session-{ttl_seconds}", ttl_seconds=ttl_seconds, now=now)
        store.create(session)
        changed = session.model_copy(
            update={"revision": 1, "updated_at": now + timedelta(seconds=1)}
        )
        assert store.update(changed, expected_revision=0).revision == 1

        with pytest.raises(IntakeSessionConflictError) as error:
            store.update(changed, expected_revision=0)
        assert error.value.current is not None
        assert error.value.current.revision == 1


@settings(max_examples=25, deadline=None)
@given(parameters=st.dictionaries(st.text(min_size=1, max_size=12), st.integers(-5, 5), max_size=3))
def test_reference_store_binds_record_content_to_checksum(parameters: dict[str, int]):
    """The catalog accepts the content digest and rejects a substituted digest."""

    preset = Preset(
        preset_id="baseline",
        version="v1",
        display_name="Baseline",
        parameters=parameters,
    )
    digest = fingerprint(preset.model_dump(mode="json"))
    wrong_digest = hashlib.sha256((digest + "-wrong").encode()).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        store = DirectoryReferenceCatalogStore(directory)

        with pytest.raises(ValueError, match="checksum"):
            store.publish_reference("preset", "baseline", "v1", preset, sha256=wrong_digest)

        store.publish_reference("preset", "baseline", "v1", preset, sha256=digest)
        loaded = store.get_reference("preset", "baseline", "v1", digest, Preset)
        assert loaded is not None
        assert loaded.model_dump(mode="json") == preset.model_dump(mode="json")


matrix_values = st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=3, unique=True)


@settings(max_examples=40, deadline=None)
@given(
    matrix=st.dictionaries(
        st.sampled_from(["experiment.epochs", "experiment.batch_size", "job.machine_type"]),
        matrix_values,
        min_size=1,
        max_size=3,
    )
)
def test_matrix_expansion_is_order_independent_and_cartesian(matrix: dict[str, list[int]]):
    """Dimension insertion order cannot alter a deterministic batch plan."""

    expanded = expand_matrix(matrix)
    reversed_matrix = dict(reversed(list(matrix.items())))
    assert expanded == expand_matrix(reversed_matrix)
    assert len(expanded) == len(list(itertools.product(*(matrix[key] for key in sorted(matrix)))))
    assert all(set(item) == set(matrix) for item in expanded)


@settings(max_examples=30, deadline=None)
@given(payloads=st.lists(st.integers(-10, 10), max_size=12))
def test_event_sequences_are_monotonic_and_cursor_replayable(payloads: list[int]):
    """Cursors page the append-only stream without duplicates or regressions."""

    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteRunStore(Path(directory) / "events.sqlite")
        run_id = "run-" + hashlib.sha256(repr(payloads).encode()).hexdigest()[:16]
        run = RunRecord(
            run_id=run_id,
            object_slug="panel",
            experiment_id="exp-1",
            dataset_version_id="data-1",
            runtime_id="runtime-1",
            state=RunState.PENDING,
            created_at=datetime.now(UTC),
        )
        store.create(run, "fingerprint-" + run_id, {"idempotency_key": "request-" + run_id})
        for index, value in enumerate(payloads):
            store.append_event(run_id, "run.state", {"value": value}, event_id=f"event-{index}")

        cursor = 0
        seen: list[int] = []
        while True:
            events, next_cursor = store.read_events(run_id, cursor=cursor, limit=2)
            seen.extend(event.sequence for event in events if event.event_type == "run.state")
            if next_cursor is None:
                break
            assert next_cursor > cursor
            cursor = next_cursor

        assert seen == list(range(2, len(payloads) + 2))
