from datetime import UTC, datetime
from typing import cast

import pytest

from defect_platform.contracts import (
    ComponentCatalog,
    ComponentDefinition,
    ComponentRef,
    ErrorOwner,
    ErrorRecord,
    ErrorRef,
    ExperimentMatrix,
    ExperimentVariant,
    JsonPatchError,
    LifecycleState,
    OperatorProfile,
    Preset,
    PresetRef,
    SchemaRegistry,
    StatusRecord,
    TelemetryEvent,
    TelemetrySeverity,
    apply_json_patch,
)

SHA = "a" * 64


def test_references_bind_catalog_preset_and_matrix_identity():
    component = ComponentDefinition(
        component_id="trainer",
        display_name="Trainer",
        owner="trainer-engineer",
        kind="trainer",
        version="1",
        reference=ComponentRef(id="trainer", version="1", sha256=SHA),
    )
    catalog = ComponentCatalog(catalog_id="platform", version="1", components=(component,))
    assert catalog.components[0].reference.key == "component:trainer:1"
    preset = Preset(
        preset_id="small",
        version="1",
        display_name="Small",
        preset_ref=PresetRef(id="small", version="1", sha256=SHA),
    )
    assert preset.preset_ref is not None
    assert preset.preset_ref.key == "preset:small:1"
    matrix = ExperimentMatrix(
        matrix_id="m1",
        version="1",
        object_slug="valve",
        variants=(ExperimentVariant(variant_id="v1", preset_ref=preset.preset_ref),),
    )
    assert matrix.variants[0].variant_id == "v1"


def test_json_patch_supports_nested_arrays_move_copy_and_atomic_failure():
    original = {"labels": ["a", "b"], "meta": {"old": 1}}
    patched = apply_json_patch(
        original,
        [
            {"op": "add", "path": "/labels/-", "value": None},
            {"op": "move", "from": "/meta/old", "path": "/meta/new"},
            {"op": "copy", "from": "/meta/new", "path": "/meta/copy"},
            {"op": "test", "path": "/labels/0", "value": "a"},
        ],
    )
    assert patched == {"labels": ["a", "b", None], "meta": {"new": 1, "copy": 1}}
    assert original == {"labels": ["a", "b"], "meta": {"old": 1}}
    with pytest.raises(JsonPatchError):
        apply_json_patch(original, [{"op": "replace", "path": "/missing", "value": 1}])


def test_status_error_and_telemetry_preserve_failure_ownership():
    error = ErrorRecord(
        error_id="e1",
        code="GPU_CAPACITY",
        owner=ErrorOwner.INFRASTRUCTURE,
        message="capacity unavailable",
        stage="dispatch",
        retryable=True,
        occurred_at=datetime.now(UTC),
        error_ref=ErrorRef(id="e1", sha256=SHA),
    )
    status = StatusRecord(
        status_id="run-1",
        state=LifecycleState.FAILED,
        observed_at=datetime.now(UTC),
        error_ref=error.error_ref,
    )
    event = TelemetryEvent(
        event_id="t1",
        occurred_at=datetime.now(UTC),
        component="control",
        event="run.failed",
        severity=TelemetrySeverity.ERROR,
        error=error,
    )
    assert status.error_ref is not None
    assert status.error_ref.id == "e1"
    assert event.error is not None
    assert event.error.owner is ErrorOwner.INFRASTRUCTURE
    with pytest.raises(ValueError, match="requires an error reference"):
        StatusRecord(status_id="run-2", state=LifecycleState.FAILED, observed_at=datetime.now(UTC))


def test_schema_registry_rejects_changed_versions_and_validates_models():
    registry = SchemaRegistry()
    entry = registry.register("operator-profile", "1", OperatorProfile)
    registered = registry.get("operator-profile", "1")
    assert registered is not None
    assert registered.sha256 == entry.sha256
    value = registry.validate(
        "operator-profile",
        "1",
        {"profile_id": "operator", "version": "1", "display_name": "Operator", "owner": "ops"},
    )
    profile = cast(OperatorProfile, value)
    assert profile.profile_id == "operator"
    with pytest.raises(ValueError, match="different content"):
        registry.register("operator-profile", "1", ComponentCatalog)
