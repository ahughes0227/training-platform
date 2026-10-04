from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from defect_platform.control.setup import (
    apply_json_patch,
    new_intake_session,
)
from defect_platform.control.store import (
    IntakeSessionConflictError,
    SQLiteIntakeSessionStore,
)


def test_json_patch_updates_nested_setup_document():
    value = apply_json_patch(
        {"notes": "", "items": []},
        [
            {"op": "replace", "path": "/notes", "value": "operator note"},
            {"op": "add", "path": "/items/-", "value": "source.csv"},
        ],
    )
    assert value == {"notes": "operator note", "items": ["source.csv"]}


def test_sqlite_intake_session_revision_fence_and_ttl(tmp_path):
    now = datetime.now(UTC)
    store = SQLiteIntakeSessionStore(tmp_path / "state.sqlite")
    session = new_intake_session("session-1", now=now)
    store.create(session)
    changed = session.model_copy(update={"revision": 1, "updated_at": now + timedelta(seconds=1)})
    store.update(changed, expected_revision=0)
    with pytest.raises(IntakeSessionConflictError):
        store.update(changed.model_copy(update={"revision": 2}), expected_revision=0)
    current = store.get("session-1")
    assert current is not None
    assert current.revision == 1
    expired = new_intake_session("expired", now=now, ttl_seconds=1)
    store.create(expired)
    assert store.get("expired", now=now + timedelta(seconds=2)) is None
