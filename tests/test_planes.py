from __future__ import annotations

import pytest

from defect_platform.planes import (
    HandoffState,
    Plane,
    PlaneAgentContext,
    SQLiteHandoffLedger,
    ValidationIssue,
    gate_handoff,
    make_handoff,
)


def context(plane: Plane, *, agent_id: str | None = None) -> PlaneAgentContext:
    return PlaneAgentContext(agent_id=agent_id or f"{plane.value}-agent", plane=plane,
                             capabilities=frozenset({"receive_handoff"}))


def test_accepts_valid_handoff_and_records_append_only_decision(tmp_path):
    ledger = SQLiteHandoffLedger(tmp_path / "handoffs.sqlite")
    envelope = make_handoff(source_plane=Plane.DATA, target_plane=Plane.TRAINING,
                            object_type="dataset_version", payload={"version_id": "v1"})
    decision = gate_handoff(envelope, Plane.TRAINING, context(Plane.TRAINING), ledger=ledger)
    assert decision.state is HandoffState.ACCEPTED
    assert decision.accepted_fingerprint == envelope.fingerprint
    assert ledger.decisions(envelope.handoff_id) == [decision]

    with pytest.raises(ValueError, match="already recorded"):
        ledger.append(envelope, decision)


def test_rejects_wrong_target_and_missing_receive_capability(tmp_path):
    ledger = SQLiteHandoffLedger(tmp_path / "handoffs.sqlite")
    envelope = make_handoff(source_plane=Plane.DATA, target_plane=Plane.TRAINING,
                            object_type="dataset_version", payload={"version_id": "v1"})
    receiver = PlaneAgentContext(agent_id="analysis-agent", plane=Plane.ANALYSIS)
    decision = gate_handoff(envelope, Plane.ANALYSIS, receiver, ledger=ledger)
    assert decision.state is HandoffState.REJECTED
    assert {issue.code for issue in decision.issues} == {"TARGET_MISMATCH", "CAPABILITY_DENIED"}
    assert decision.required_action


def test_validator_feedback_is_structured_and_owner_bound():
    envelope = make_handoff(source_plane=Plane.TRAINING, target_plane=Plane.ANALYSIS,
                            object_type="model_result", payload={"metric": 0.1})

    def validate(payload):
        assert payload["metric"] == 0.1
        return (ValidationIssue(code="METRIC_BELOW_POLICY", layer="meaning",
                                message="validation MCC is below the configured threshold",
                                owner=Plane.ANALYSIS),)

    decision = gate_handoff(envelope, Plane.ANALYSIS, context(Plane.ANALYSIS), validator=validate)
    assert decision.state is HandoffState.REJECTED
    assert decision.issues[0].owner is Plane.ANALYSIS
    assert decision.issues[0].layer == "meaning"


def test_boundary_rechecks_nested_payload_integrity():
    envelope = make_handoff(source_plane=Plane.DATA, target_plane=Plane.TRAINING,
                            object_type="dataset_version", payload={"version_id": "v1"})
    envelope.payload["version_id"] = "tampered"
    decision = gate_handoff(envelope, Plane.TRAINING, context(Plane.TRAINING))
    assert decision.state is HandoffState.REJECTED
    assert decision.issues[0].code == "PAYLOAD_MUTATED"


def test_repair_is_new_revision_and_keeps_boundary():
    original = make_handoff(source_plane=Plane.DATA, target_plane=Plane.TRAINING,
                            object_type="dataset_version", payload={"version_id": "v1"})
    repaired = make_handoff(source_plane=Plane.DATA, target_plane=Plane.TRAINING,
                            object_type="dataset_version", payload={"version_id": "v2"},
                            revision=2, parent=original)
    assert repaired.revision == 2
    assert repaired.parent_fingerprint == original.fingerprint
    assert repaired.fingerprint != original.fingerprint

    with pytest.raises(ValueError, match="original plane boundary"):
        make_handoff(source_plane=Plane.DATA, target_plane=Plane.ANALYSIS,
                     object_type="dataset_version", payload={"version_id": "v3"},
                     revision=2, parent=original)


def test_plane_context_isolated_snapshots_do_not_share_mutable_context():
    data = context(Plane.DATA).with_context(dataset="v1")
    training = context(Plane.TRAINING).with_context(experiment="e1")
    assert data.context == {"dataset": "v1"}
    assert training.context == {"experiment": "e1"}
    assert "experiment" not in data.context
    assert data.without_context("dataset").context == {}
    assert data.context == {"dataset": "v1"}
