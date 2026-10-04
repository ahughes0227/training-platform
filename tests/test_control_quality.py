from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from defect_platform.catalog_store import DirectoryReferenceCatalogStore
from defect_platform.contracts import DatasetVersion, Reference, RunRecord, RunState
from defect_platform.control.routing import ModelRoutingPolicy
from defect_platform.control.setup import (
    IntakeCoordinator,
    IntakePatch,
    SetupDraft,
    propose_setup_turn,
)
from defect_platform.control.store import SQLiteIntakeSessionStore, SQLiteRunStore
from defect_platform.telemetry import TokenBudget, TokenBudgetExceeded


class _Response:
    def __init__(self, content: str = '{"patch": []}', *, prompt: int = 0, completion: int = 0):
        self.choices = [
            type("Choice", (), {"message": type("Message", (), {"content": content})()})()
        ]
        self.usage = {"prompt_tokens": prompt, "completion_tokens": completion}


def test_routing_uses_operation_and_explicit_model_precedence(tmp_path):
    selected: list[str] = []

    def completion(**kwargs):
        selected.append(kwargs["model"])
        return _Response()

    coordinator = IntakeCoordinator(
        SQLiteIntakeSessionStore(tmp_path / "intake.sqlite"),
        routing_policy=ModelRoutingPolicy(small_model="small", large_model="large"),
        completion=completion,
    )
    response = coordinator.start("initial")
    coordinator.turn(
        response.session.session_id,
        "choose architecture",
        expected_revision=1,
        operation="architecture",
    )
    coordinator.turn(
        response.session.session_id,
        "use this model",
        expected_revision=2,
        operation="review",
        model="explicit",
    )
    assert selected == ["small", "large", "explicit"]


def test_prompt_budget_is_checked_before_provider_call():
    calls = 0

    def completion(**_kwargs):
        nonlocal calls
        calls += 1
        return _Response()

    with pytest.raises(TokenBudgetExceeded, match="prompt token budget"):
        propose_setup_turn(
            SetupDraft(),
            "answer",
            model="small",
            completion=completion,
            include_schema=True,
            budget=TokenBudget(max_prompt_tokens=1),
        )
    assert calls == 0


def test_completion_budget_uses_provider_usage():
    with pytest.raises(TokenBudgetExceeded, match="completion token budget"):
        propose_setup_turn(
            SetupDraft(),
            "answer",
            model="small",
            completion=lambda **_kwargs: _Response(completion=3),
            budget=TokenBudget(max_completion_tokens=2),
        )


def test_reference_lookup_accepts_optional_version_and_domain_digest(tmp_path):
    dataset = DatasetVersion(
        version_id="dataset-v1",
        object_slug="panel",
        root_uri="gs://data/root",
        manifest_uri="gs://data/manifest.json",
        shard_uris={"train": ["gs://data/a"]},
        sample_counts={"train": 1},
        sha256="a" * 64,
        source_snapshot_uri="gs://data/source",
    )
    store = DirectoryReferenceCatalogStore(tmp_path)
    store.publish_reference("dataset_version", dataset.version_id, "v1", dataset)
    reference = Reference(kind="dataset_version", id=dataset.version_id, sha256=dataset.sha256)
    resolved = store.get_reference(
        "dataset_version", dataset.version_id, None, reference.sha256, DatasetVersion
    )
    assert resolved == dataset


def test_update_with_event_rolls_back_state_when_event_persist_fails(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    run = RunRecord(
        run_id="run-1",
        object_slug="panel",
        experiment_id="exp-1",
        dataset_version_id="dataset-v1",
        runtime_id="runtime-v1",
        state=RunState.PENDING,
        created_at=datetime.now(UTC),
    )
    store.create(run, "fingerprint", {"idempotency_key": "request-1"})
    with pytest.raises(TypeError, match="JSON serializable"):
        store.update_with_event(
            run.model_copy(update={"state": RunState.RUNNING}), "run.state", {"bad": object()}
        )
    stored = store.get(run.run_id)
    assert stored is not None
    assert stored.state is RunState.PENDING
    events, _ = store.read_events(run.run_id)
    assert [event.event_type for event in events] == ["run.created"]


def test_patch_schema_is_validated_against_registry_shape():
    patch = IntakePatch.model_validate_json(json.dumps({"patch": []}))
    assert patch.patch == []
    with pytest.raises(ValueError):
        IntakePatch.model_validate_json(json.dumps({"patch": [], "unexpected": True}))
