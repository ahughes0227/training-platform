from __future__ import annotations

import json

import pytest

from defect_platform.analysis import (
    AnalysisAgent,
    AnalysisPolicy,
    MlflowRunReader,
    evidence_from_report,
    validate_data_request,
    validate_experiment_proposal,
)
from defect_platform.contracts import ExperimentConfig, ModelSpec
from defect_platform.planes import (
    HandoffState,
    Plane,
    PlaneAgentContext,
    SQLiteHandoffLedger,
    gate_handoff,
    make_handoff,
)

CLASSES = ["ok", "scratch", "dent"]


def experiment(**overrides) -> ExperimentConfig:
    values = {"experiment_id": "exp-1", "object_slug": "bracket", "dataset_version_id": "ds-v1",
              "runtime_id": "rt-1",
              "model": ModelSpec(weights_uri="gs://w", weights_sha256="a" * 64)}
    return ExperimentConfig(**{**values, **overrides})


def report(recalls=(0.95, 0.9, 0.9), supports=(100, 100, 100), review_rate=0.05,
           run_id="run-1") -> dict:
    per_class = {name: {"precision": 0.9, "recall": recall, "f1": 0.9, "support": float(support)}
                 for name, recall, support in zip(CLASSES, recalls, supports)}
    confusion = []
    for i, (recall, support) in enumerate(zip(recalls, supports)):
        hits = round(recall * support)
        row = [0, 0, 0]
        row[i] = hits
        row[(i + 1) % 3] = support - hits
        confusion.append(row)
    return {"classes": CLASSES, "best_epoch": 3, "mlflow_run_id": run_id,
            "validation": {"mcc": 0.8, "macro_f1": 0.85, "accuracy": 0.9,
                           "per_class": per_class, "confusion": confusion},
            "test": {"mcc": 0.1, "macro_f1": 0.1, "accuracy": 0.1, "per_class": {},
                     "confusion": []},
            "training_config": {"optimizer": "adamw", "loss": "cross_entropy",
                                "focal_gamma": 2.0, "horizontal_flip_probability": 0.5,
                                "class_weights": {}, "seed": 42, "max_review_error_rate": None},
            "abstention": {"confidence_threshold": 0.7, "review_rate": review_rate,
                           "accepted_error_rate": 0.02, "validation_count": 300}}


def agent(plane=Plane.ANALYSIS, capabilities=frozenset({"propose_handoff"}), **policy):
    context = PlaneAgentContext(agent_id="analysis-agent", plane=plane, capabilities=capabilities)
    return AnalysisAgent(context, AnalysisPolicy(**policy))


def receiver(plane: Plane) -> PlaneAgentContext:
    return PlaneAgentContext(agent_id=f"{plane.value}-agent", plane=plane,
                             capabilities=frozenset({"receive_handoff"}))


def test_healthy_run_needs_no_action():
    result = agent().analyze(evidence_from_report(report(), experiment()))
    assert result.kind == "no_action"
    assert result.handoff is None


def test_weak_class_yields_gated_experiment_proposal(tmp_path):
    evidence = evidence_from_report(report(recalls=(0.95, 0.5, 0.9)), experiment())
    result = agent().analyze(evidence)
    assert result.kind == "experiment_proposal"
    assert result.weak_classes == ("scratch",)
    envelope = result.handoff
    assert (envelope.source_plane, envelope.target_plane) == (Plane.ANALYSIS, Plane.CONTROL)
    assert envelope.evidence_fingerprints == (evidence.report_sha256,)

    proposed = ExperimentConfig.model_validate(envelope.payload["experiment"])
    assert proposed.class_weights == {"ok": 1.0, "scratch": 1.6, "dent": 1.0}
    assert proposed.loss == "cross_entropy"
    assert proposed.experiment_id.startswith("exp-1-a") and proposed.experiment_id != "exp-1"
    assert (proposed.dataset_version_id, proposed.runtime_id) == ("ds-v1", "rt-1")
    assert set(envelope.payload["changes"]) == {"class_weights"}

    ledger = SQLiteHandoffLedger(tmp_path / "handoffs.sqlite")
    decision = gate_handoff(envelope, Plane.CONTROL, receiver(Plane.CONTROL),
                            validator=validate_experiment_proposal, ledger=ledger)
    assert decision.state is HandoffState.ACCEPTED
    assert ledger.decisions(envelope.handoff_id) == [decision]


def test_proposal_is_deterministic_for_the_same_evidence():
    evidence = evidence_from_report(report(recalls=(0.95, 0.5, 0.9)), experiment())
    first, second = agent().analyze(evidence), agent().analyze(evidence)
    assert first.handoff.payload == second.handoff.payload


def test_several_weak_classes_switch_to_focal_and_weights_are_capped():
    logged = report(recalls=(0.95, 0.1, 0.6))
    logged["training_config"]["class_weights"] = {"scratch": 9.0}
    evidence = evidence_from_report(logged, experiment(class_weights={"scratch": 9.0}))
    proposed = ExperimentConfig.model_validate(
        agent().analyze(evidence).handoff.payload["experiment"])
    assert proposed.loss == "focal"
    assert proposed.class_weights == {"ok": 1.0, "scratch": 10.0, "dent": 1.3333}


def test_high_review_rate_unfreezes_one_more_block_until_cap():
    evidence = evidence_from_report(report(review_rate=0.4), experiment())
    result = agent().analyze(evidence)
    assert result.kind == "experiment_proposal"
    assert set(result.handoff.payload["changes"]) == {"model.unfreeze_last_n"}

    capped = experiment(model=ModelSpec(weights_uri="gs://w", weights_sha256="a" * 64,
                                        unfreeze_last_n=4))
    result = agent().analyze(evidence.model_copy(update={"experiment": capped}))
    assert result.kind == "no_action"
    assert "human review" in result.rationale[-1]


def test_low_support_weak_class_requests_data_instead_of_retraining(tmp_path):
    evidence = evidence_from_report(report(recalls=(0.95, 0.5, 0.9), supports=(100, 10, 100)),
                                    experiment())
    result = agent().analyze(evidence)
    assert result.kind == "data_request"
    envelope = result.handoff
    assert envelope.target_plane is Plane.DATA
    need = envelope.payload["needs"][0]
    assert (need["class_name"], need["reason"]) == ("scratch", "insufficient_support")
    assert need["confused_with"] == ["dent"]
    decision = gate_handoff(envelope, Plane.DATA, receiver(Plane.DATA),
                            validator=validate_data_request,
                            ledger=SQLiteHandoffLedger(tmp_path / "h.sqlite"))
    assert decision.state is HandoffState.ACCEPTED


def test_suspected_label_errors_request_relabel_with_sample_ids():
    flagged = [{"sample_id": f"s{i}", "true_class": "scratch", "predicted_class": "ok",
                "confidence": 0.95, "reason": "suspected_label_error"} for i in range(12)]
    flagged.append({"sample_id": "d1", "true_class": "dent", "predicted_class": "ok",
                    "confidence": 0.4, "reason": "low_confidence"})
    evidence = evidence_from_report(report(recalls=(0.95, 0.5, 0.9)), experiment(),
                                    flagged_cases=flagged)
    result = agent(max_flagged_samples_per_request=5).analyze(evidence)
    assert result.kind == "data_request"
    need = result.handoff.payload["needs"][0]
    assert need["reason"] == "suspected_label_noise"
    assert need["sample_ids"] == ["s0", "s1", "s10", "s11", "s2"]


def test_control_rejects_proposal_that_changes_dataset_or_runtime():
    evidence = evidence_from_report(report(recalls=(0.95, 0.5, 0.9)), experiment())
    payload = dict(agent().analyze(evidence).handoff.payload)
    payload["experiment"] = {**payload["experiment"], "dataset_version_id": "ds-v2"}
    forged = make_handoff(source_plane=Plane.ANALYSIS, target_plane=Plane.CONTROL,
                          object_type="experiment_proposal", payload=payload)
    decision = gate_handoff(forged, Plane.CONTROL, receiver(Plane.CONTROL),
                            validator=validate_experiment_proposal)
    assert decision.state is HandoffState.REJECTED
    assert {issue.code for issue in decision.issues} == {"PROPOSAL_OUT_OF_SCOPE",
                                                          "PROPOSAL_CHANGES_MISMATCH"}
    assert all(issue.owner is Plane.ANALYSIS for issue in decision.issues)


def test_control_rejects_malformed_or_empty_proposals():
    assert validate_experiment_proposal({"experiment": {}})[0].code == "PROPOSAL_INVALID"
    parent = experiment().model_dump(mode="json")
    issues = validate_experiment_proposal({
        "parent_experiment": parent, "experiment": parent, "source_mlflow_run_id": "run-1",
        "changes": {"epochs": {"from": 10, "to": 20}}, "rationale": ["x"]})
    assert {issue.code for issue in issues} == {"PROPOSAL_ID_REUSED", "PROPOSAL_CHANGES_MISMATCH",
                                                "PROPOSAL_EMPTY"}


def test_agent_requires_analysis_context_and_capability():
    with pytest.raises(PermissionError, match="analysis-plane"):
        agent(plane=Plane.CONTROL)
    with pytest.raises(PermissionError, match="propose_handoff"):
        agent(capabilities=frozenset()).analyze(evidence_from_report(report(), experiment()))


def test_evidence_rejects_mismatched_experiment_and_unknown_classes():
    with pytest.raises(ValueError, match="loss"):
        evidence_from_report(report(), experiment(loss="focal"))
    with pytest.raises(ValueError, match="unknown class"):
        evidence_from_report(report(), experiment(), flagged_cases=[{
            "sample_id": "x", "true_class": "crack", "predicted_class": "ok",
            "confidence": 0.5, "reason": "misclassified"}])


def test_evidence_ignores_test_split():
    evidence = evidence_from_report(report(), experiment())
    assert "test" not in evidence.model_dump()
    assert evidence.validation.mcc == 0.8


def test_mlflow_reader_downloads_evaluation_and_optional_flagged_cases(tmp_path):
    (tmp_path / "evaluation.json").write_text(json.dumps(report(run_id="run-9")))
    (tmp_path / "flagged_cases.json").write_text(json.dumps({"cases": [
        {"sample_id": "s1", "true_class": "dent", "predicted_class": "ok",
         "confidence": 0.3, "reason": "low_confidence"}]}))
    calls = []

    def download(run_id, path):
        calls.append((run_id, path))
        target = tmp_path / path
        return target if target.exists() else None

    evidence = MlflowRunReader(download=download).read("run-9", experiment())
    assert calls == [("run-9", "evaluation.json"), ("run-9", "flagged_cases.json")]
    assert evidence.mlflow_run_id == "run-9"
    assert evidence.flagged_cases[0].sample_id == "s1"

    (tmp_path / "flagged_cases.json").unlink()
    assert MlflowRunReader(download=download).read("run-9", experiment()).flagged_cases == ()
    with pytest.raises(ValueError, match="different MLflow run"):
        MlflowRunReader(download=download).read("run-other", experiment())
    with pytest.raises(FileNotFoundError):
        MlflowRunReader(download=lambda run_id, path: None).read("run-9", experiment())
