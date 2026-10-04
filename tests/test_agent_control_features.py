from __future__ import annotations

import json

from defect_platform.catalog_store import DirectoryReferenceCatalogStore
from defect_platform.contracts import ComponentDefinition, ComponentRef
from defect_platform.control.defaults import resolve_defaults
from defect_platform.control.presets import BUILTIN_PRESET_IDS, resolve_preset
from defect_platform.control.setup import (
    INTAKE_PATCH_SCHEMA_REF,
    IntakePatch,
    SetupDraft,
    intake_patch_schema_checksum,
    propose_setup_turn,
    setup_schema_checksum,
)


def test_stateful_turn_contains_only_delta_context():
    captured = {}

    class Message:
        content = '{"patch": [], "clarifying_questions": ["Provide labels"]}'

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    def completion(**kwargs):
        captured.update(kwargs)
        return Response()

    propose_setup_turn(
        SetupDraft(),
        "labels.csv",
        model="small",
        completion=completion,
        session_id="s-1",
        revision=3,
        missing_fields=["sources"],
    )
    payload = json.loads(captured["messages"][1]["content"])
    assert payload == {
        "session_id": "s-1",
        "revision": 3,
        "missing_fields": ["sources"],
        "latest_answers": "labels.csv",
        "schema_ref": "urn:defect-platform:setup-draft:v1",
        "schema_sha256": setup_schema_checksum(),
        "response_schema_ref": INTAKE_PATCH_SCHEMA_REF,
        "response_schema_sha256": intake_patch_schema_checksum(),
    }
    assert "draft" not in payload
    assert "model_json_schema" not in captured["messages"][0]["content"]


def test_first_turn_exposes_referenceable_response_schema():
    captured = {}

    class Message:
        content = '{"patch": []}'

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    def completion(**kwargs):
        captured.update(kwargs)
        return Response()

    propose_setup_turn(
        SetupDraft(), "start", model="small", completion=completion, include_schema=True
    )
    payload = json.loads(captured["messages"][1]["content"])
    assert payload["response_schema_ref"] == INTAKE_PATCH_SCHEMA_REF
    assert payload["response_schema_sha256"] == intake_patch_schema_checksum()
    assert payload["response_schema"] == IntakePatch.model_json_schema()


def test_reference_catalog_is_create_only_and_checksum_bound(tmp_path):
    value = ComponentDefinition(
        component_id="prep",
        display_name="Prep",
        owner="team",
        kind="preprocessing",
        version="1",
        reference=ComponentRef(id="prep", version="1", sha256="a" * 64),
    )
    store = DirectoryReferenceCatalogStore(tmp_path)
    path = store.publish_reference("preprocessing", "prep", "1", value)
    loaded = store.get_reference(
        "preprocessing",
        "prep",
        "1",
        __import__("defect_platform.semantics", fromlist=["fingerprint"]).fingerprint(value),
        ComponentDefinition,
    )
    assert loaded == value
    assert path.endswith(".json")


def test_presets_and_defaults_are_deterministic():
    assert set(BUILTIN_PRESET_IDS) == {
        "baseline_classifier",
        "fine_tune_backbone",
        "low_cost_gpu",
        "production_ray",
    }
    assert resolve_preset("fine_tune_backbone")["unfreeze_last_n"] == 1
    assert resolve_defaults({"epochs": 20})["epochs"] == 20
