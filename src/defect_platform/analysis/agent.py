"""Analysis plane agent: turn run evidence into a gated next step.

The agent reads validation evidence for one finished run and emits at most one
immutable handoff:

* ``data_request`` to the Data plane when a weak class is data-limited (too few
  validation samples) or carries suspected label errors.  Retraining on data
  known to be insufficient wastes compute, so data requests take precedence.
* ``experiment_proposal`` to the Control plane with a complete next
  ``ExperimentConfig`` that changes only training hyperparameters.
* nothing, when every class meets policy or no admissible change remains.

Decisions are deterministic for a given evidence snapshot and policy.  The
agent never changes the dataset version, certified runtime, model weights or
class catalog, and it never submits work: the receiving plane gates every
handoff with the validators below.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, ClassVar, Literal

from pydantic import Field

from ..contracts import ExperimentConfig, StrictModel
from ..planes import (
    HandoffEnvelope,
    Plane,
    PlaneAgentContext,
    ValidationIssue,
    make_handoff,
)
from .evidence import RunEvidence

PROPOSE_CAPABILITY = "propose_handoff"
EXPERIMENT_PROPOSAL = "experiment_proposal"
DATA_REQUEST = "data_request"

# Fields the Analysis plane may change between a parent and proposed experiment.
# Everything else (dataset, runtime, weights, catalog, object) is owned elsewhere.
TUNABLE_FIELDS = frozenset({"experiment_id", "class_weights", "loss", "focal_gamma", "epochs",
                            "learning_rate", "model.unfreeze_last_n"})


class AnalysisPolicy(StrictModel):
    min_class_recall: float = Field(default=0.8, gt=0, le=1)
    min_class_support: int = Field(default=30, ge=1)
    max_label_error_fraction: float = Field(default=0.1, gt=0, le=1)
    max_class_weight: float = Field(default=10.0, gt=1)
    max_review_rate: float = Field(default=0.2, ge=0, le=1)
    max_unfreeze_last_n: int = Field(default=4, ge=0)
    focal_after_weak_classes: int = Field(default=2, ge=1)
    max_flagged_samples_per_request: int = Field(default=50, ge=0)


class DataNeed(StrictModel):
    class_name: str
    reason: Literal["insufficient_support", "suspected_label_noise"]
    validation_support: int = Field(ge=0)
    validation_recall: float = Field(ge=0, le=1)
    confused_with: tuple[str, ...] = ()
    sample_ids: tuple[str, ...] = ()


class DataRequest(StrictModel):
    object_slug: str
    dataset_version_id: str
    source_mlflow_run_id: str
    source_experiment_id: str
    needs: tuple[DataNeed, ...] = Field(min_length=1)


class ExperimentProposal(StrictModel):
    parent_experiment: ExperimentConfig
    experiment: ExperimentConfig
    source_mlflow_run_id: str
    changes: dict[str, dict[str, Any]] = Field(min_length=1)
    rationale: tuple[str, ...] = Field(min_length=1)


class AnalysisResult(StrictModel):
    kind: Literal["experiment_proposal", "data_request", "no_action"]
    rationale: tuple[str, ...]
    weak_classes: tuple[str, ...] = ()
    handoff: HandoffEnvelope | None = None

    model_config: ClassVar = {"frozen": True, "extra": "forbid"}


class AnalysisAgent:
    """Stateless Analysis plane agent bound to one plane context and policy."""

    def __init__(self, context: PlaneAgentContext, policy: AnalysisPolicy | None = None):
        if context.plane != Plane.ANALYSIS:
            raise PermissionError("the analysis agent requires an analysis-plane context")
        self.context = context
        self.policy = policy or AnalysisPolicy()

    def analyze(self, evidence: RunEvidence) -> AnalysisResult:
        if not self.context.can(PROPOSE_CAPABILITY):
            raise PermissionError(f"agent lacks the {PROPOSE_CAPABILITY} capability")
        policy = self.policy
        per_class = evidence.validation.per_class
        weak = tuple(name for name in evidence.classes
                     if per_class[name].recall < policy.min_class_recall)
        review_rate = evidence.abstention.review_rate if evidence.abstention else None
        review_high = review_rate is not None and review_rate > policy.max_review_rate
        if not weak and not review_high:
            return AnalysisResult(kind="no_action", rationale=(
                f"every class meets validation recall >= {policy.min_class_recall}",))

        needs = self._data_needs(evidence, weak)
        if needs:
            request = DataRequest(object_slug=evidence.experiment.object_slug,
                                  dataset_version_id=evidence.experiment.dataset_version_id,
                                  source_mlflow_run_id=evidence.mlflow_run_id,
                                  source_experiment_id=evidence.experiment.experiment_id,
                                  needs=needs)
            rationale = tuple(f"{need.class_name}: {need.reason.replace('_', ' ')}"
                              for need in needs)
            return AnalysisResult(kind="data_request", rationale=rationale, weak_classes=weak,
                                  handoff=self._handoff(Plane.DATA, DATA_REQUEST,
                                                        request, evidence, "request_data"))

        experiment, rationale = self._next_experiment(evidence, weak, review_rate)
        changes = config_changes(evidence.experiment, experiment)
        changes.pop("experiment_id", None)
        if not changes:
            return AnalysisResult(kind="no_action", weak_classes=weak, rationale=(
                *rationale, "no admissible hyperparameter change remains; needs human review"))
        experiment = experiment.model_copy(update={
            "experiment_id": _derived_id(evidence.experiment.experiment_id, changes)})
        proposal = ExperimentProposal(parent_experiment=evidence.experiment,
                                      experiment=experiment,
                                      source_mlflow_run_id=evidence.mlflow_run_id,
                                      changes=changes, rationale=rationale)
        return AnalysisResult(kind="experiment_proposal", rationale=rationale, weak_classes=weak,
                              handoff=self._handoff(Plane.CONTROL, EXPERIMENT_PROPOSAL,
                                                    proposal, evidence, "propose_experiment"))

    def _data_needs(self, evidence: RunEvidence, weak: tuple[str, ...]) -> tuple[DataNeed, ...]:
        policy = self.policy
        needs = []
        for name in weak:
            metrics = evidence.validation.per_class[name]
            suspected = [case.sample_id for case in evidence.flagged_cases
                         if case.reason == "suspected_label_error" and case.true_class == name]
            if metrics.support < policy.min_class_support:
                reason, samples = "insufficient_support", ()
            elif metrics.support and len(suspected) / metrics.support >= \
                    policy.max_label_error_fraction:
                reason = "suspected_label_noise"
                samples = tuple(sorted(suspected)[:policy.max_flagged_samples_per_request])
            else:
                continue
            needs.append(DataNeed(
                class_name=name, reason=reason, validation_support=metrics.support,
                validation_recall=metrics.recall, sample_ids=samples,
                confused_with=tuple(other for other, _ in evidence.confused_with(name))))
        return tuple(needs)

    def _next_experiment(self, evidence: RunEvidence, weak: tuple[str, ...],
                         review_rate: float | None) -> tuple[ExperimentConfig, tuple[str, ...]]:
        policy = self.policy
        parent = evidence.experiment
        weights = dict(parent.class_weights)
        rationale = []
        for name in weak:
            recall = evidence.validation.per_class[name].recall
            current = weights.get(name, 1.0)
            # Scale by the recall shortfall, bounded so one run cannot swing the loss.
            scale = min(2.0, policy.min_class_recall / max(recall, 0.05))
            proposed = round(min(policy.max_class_weight, current * scale), 4)
            if proposed > current:
                weights[name] = proposed
                rationale.append(f"{name}: validation recall {recall:.3f} < "
                                 f"{policy.min_class_recall}; class weight {current} -> {proposed}")
        update: dict[str, Any] = {"class_weights": weights}
        if len(weak) >= policy.focal_after_weak_classes and parent.loss == "cross_entropy":
            update["loss"] = "focal"
            rationale.append(f"{len(weak)} weak classes; switch to focal loss "
                             f"(gamma {parent.focal_gamma})")
        model = parent.model
        if review_rate is not None and review_rate > policy.max_review_rate:
            if model.unfreeze_last_n < policy.max_unfreeze_last_n:
                model = model.model_copy(update={"unfreeze_last_n": model.unfreeze_last_n + 1})
                rationale.append(f"abstention review rate {review_rate:.3f} > "
                                 f"{policy.max_review_rate}; unfreeze_last_n -> "
                                 f"{model.unfreeze_last_n}")
            else:
                rationale.append(f"abstention review rate {review_rate:.3f} > "
                                 f"{policy.max_review_rate} but backbone unfreezing is at the cap")
        update["model"] = model
        return parent.model_copy(update=update), tuple(rationale)

    def _handoff(self, target: Plane, object_type: str, payload: StrictModel,
                 evidence: RunEvidence, scope: str) -> HandoffEnvelope:
        return make_handoff(source_plane=Plane.ANALYSIS, target_plane=target,
                            object_type=object_type, payload=payload.model_dump(mode="json"),
                            authority_scope=(scope,),
                            evidence_fingerprints=(evidence.report_sha256,))


def config_changes(parent: ExperimentConfig, proposed: ExperimentConfig) -> dict[str, dict[str, Any]]:
    """Flattened ``{field: {"from", "to"}}`` differences between two experiments."""
    before, after = _flatten(parent.model_dump(mode="json")), _flatten(proposed.model_dump(mode="json"))
    return {key: {"from": before.get(key), "to": after.get(key)}
            for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)}


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        # class_weights is one tunable value; other nested models are flattened.
        if isinstance(value, dict) and key != "class_weights":
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _derived_id(parent_id: str, changes: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(changes, sort_keys=True).encode()).hexdigest()[:8]
    return f"{parent_id}-a{digest}"


def _issue(code: str, layer: str, message: str, owner: Plane = Plane.ANALYSIS) -> ValidationIssue:
    return ValidationIssue(code=code, layer=layer, message=message, owner=owner)


def validate_experiment_proposal(payload: dict[str, Any]) -> tuple[ValidationIssue, ...]:
    """Control-plane validator for an Analysis ``experiment_proposal`` payload.

    Pass it to ``gate_handoff`` as ``validator``.  Acceptance only admits the
    proposal; Control still applies its own admission and budget policy.
    """
    try:
        proposal = ExperimentProposal.model_validate(payload)
    except ValueError as exc:
        return (_issue("PROPOSAL_INVALID", "structure", str(exc)[:2000]),)
    issues = []
    actual = config_changes(proposal.parent_experiment, proposal.experiment)
    outside = sorted(set(actual) - TUNABLE_FIELDS)
    if outside:
        issues.append(_issue("PROPOSAL_OUT_OF_SCOPE", "authority",
                             f"analysis may not change {', '.join(outside)}"))
    if "experiment_id" not in actual:
        issues.append(_issue("PROPOSAL_ID_REUSED", "integrity",
                             "a proposal needs a new experiment id"))
    declared = {key: value for key, value in actual.items() if key != "experiment_id"}
    if proposal.changes != declared:
        issues.append(_issue("PROPOSAL_CHANGES_MISMATCH", "integrity",
                             "declared changes do not match the parent and proposed experiments"))
    if not declared:
        issues.append(_issue("PROPOSAL_EMPTY", "meaning",
                             "proposal does not change any hyperparameter"))
    return tuple(issues)


def validate_data_request(payload: dict[str, Any]) -> tuple[ValidationIssue, ...]:
    """Data-plane validator for an Analysis ``data_request`` payload."""
    try:
        request = DataRequest.model_validate(payload)
    except ValueError as exc:
        return (_issue("DATA_REQUEST_INVALID", "structure", str(exc)[:2000]),)
    names = [need.class_name for need in request.needs]
    if len(set(names)) != len(names):
        return (_issue("DATA_REQUEST_DUPLICATE_CLASS", "meaning",
                       "each class may appear in a data request once"),)
    return ()
