"""Budgeted experiment search over a base experiment's certified, immutable inputs.

The search proposes ``ExperimentConfig`` variants, launches them through a
submitter that keeps every run-level guardrail (certified digest, reviewed
catalog, per-run cost cap), stops weak trials early by comparing learning
curves across runs, and ranks finished trials by validation MCC then macro F1.

It never changes what the base experiment pins: dataset version, runtime,
catalog fingerprint, backbone weights and preprocessing. Budget accounting is
conservative: each trial reserves its full estimated cost at launch and keeps
it, because nothing in the platform confirms that a stopped or failed job
spent less.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import Field, model_validator

from defect_platform.contracts import (
    ExperimentConfig,
    ModelSpec,
    RunState,
    StrictModel,
    VertexJobConfig,
)

if TYPE_CHECKING:
    from defect_platform.control.controller import RunController
    from defect_platform.trainer.metrics import Evaluation

# Identity and immutable-input fields a search may never vary.
PINNED_FIELDS = frozenset({
    "experiment_id", "object_slug", "dataset_version_id", "runtime_id", "catalog_sha256",
    "model", "model.backbone", "model.weights_uri", "model.weights_sha256",
    "model.image_size", "model.preprocessing",
})
SEARCHABLE_FIELDS = frozenset(
    {name for name in ExperimentConfig.model_fields if name not in PINNED_FIELDS}
    | {f"model.{name}" for name in ModelSpec.model_fields if f"model.{name}" not in PINNED_FIELDS}
)
_MAX_DRAWS_PER_PROPOSAL = 256


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _config_sha256(experiment: ExperimentConfig) -> str:
    """Identity of what a trial trains, independent of its generated experiment ID."""
    return _sha256(experiment.model_dump(mode="json", exclude={"experiment_id"}))


class SearchSpace(StrictModel):
    """Discrete choices per searchable field, addressed as ``field`` or ``model.field``."""

    choices: dict[str, list[Any]] = Field(min_length=1)
    seed: int = 0
    include_base: bool = True

    @model_validator(mode="after")
    def searchable_choices(self) -> SearchSpace:
        for path, values in self.choices.items():
            if path not in SEARCHABLE_FIELDS:
                raise ValueError(f"{path!r} is pinned by the base experiment or is not a field")
            if not values:
                raise ValueError(f"{path!r} needs at least one choice")
            if len({_canonical(value) for value in values}) != len(values):
                raise ValueError(f"{path!r} choices must be unique")
        return self

    @property
    def size(self) -> int:
        size = 1
        for values in self.choices.values():
            size *= len(values)
        return size

    @property
    def sha256(self) -> str:
        return _sha256(self.model_dump(mode="json"))


class SearchBudget(StrictModel):
    max_trials: int = Field(gt=0)
    max_total_cost_usd: float = Field(gt=0, allow_inf_nan=False)
    max_concurrent: int = Field(default=1, gt=0)
    max_consecutive_failures: int = Field(default=3, gt=0)


class TrialScore(StrictModel):
    """Validation selection metrics, ranked MCC first and macro F1 second."""

    mcc: float = Field(ge=-1, le=1, allow_inf_nan=False)
    macro_f1: float = Field(ge=0, le=1, allow_inf_nan=False)

    @property
    def selection_key(self) -> tuple[float, float]:
        # Mirrors trainer.metrics.Evaluation.selection_key.
        return self.mcc, self.macro_f1

    @classmethod
    def from_evaluation(cls, evaluation: Evaluation) -> TrialScore:
        mcc, macro_f1 = evaluation.selection_key
        return cls(mcc=mcc, macro_f1=macro_f1)


class EpochObservation(StrictModel):
    epoch: int = Field(gt=0)
    score: TrialScore


class EarlyStoppingPolicy(StrictModel):
    """Median stopping rule across runs.

    From ``grace_epochs`` on, a trial stops when its best score so far is
    strictly below the median of its peers' best scores at the same epoch.
    Peers are all other trials that reached that epoch, including finished
    and stopped ones; the rule needs at least ``min_peers`` of them.
    """

    grace_epochs: int = Field(default=2, gt=0)
    min_peers: int = Field(default=2, gt=0)

    def should_stop(self, curve: list[EpochObservation],
                    peers: list[list[EpochObservation]]) -> str | None:
        if not curve or curve[-1].epoch < self.grace_epochs:
            return None
        epoch = curve[-1].epoch
        own = _best_through(curve, epoch)
        peer_bests = [best for best in (_best_through(peer, epoch) for peer in peers
                                        if peer and peer[-1].epoch >= epoch) if best is not None]
        if own is None or len(peer_bests) < self.min_peers:
            return None
        peer_bests.sort()
        median = peer_bests[(len(peer_bests) - 1) // 2]
        if own < median:
            return (f"epoch {epoch}: best (mcc, macro_f1) {own} is below the median {median} "
                    f"of {len(peer_bests)} peer trials")
        return None


def _best_through(curve: list[EpochObservation], epoch: int) -> tuple[float, float] | None:
    keys = [item.score.selection_key for item in curve if item.epoch <= epoch]
    return max(keys) if keys else None


class TrialStatus(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Trial(StrictModel):
    trial_id: str
    index: int = Field(ge=0)
    overrides: dict[str, Any]
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    experiment: ExperimentConfig
    status: TrialStatus
    run_id: str | None = None
    reserved_cost_usd: float = Field(ge=0)
    curve: list[EpochObservation] = Field(default_factory=list)
    final: TrialScore | None = None
    stop_reason: str | None = None
    failure: str | None = None

    @property
    def best_observed(self) -> TrialScore | None:
        if self.final is not None:
            return self.final
        if not self.curve:
            return None
        return max((item.score for item in self.curve), key=lambda score: score.selection_key)


class SearchState(StrictModel):
    """Serializable progress so a search can resume after a restart."""

    search_id: str
    base_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    space_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    draws: int = Field(default=0, ge=0)
    seen: list[str] = Field(default_factory=list)
    trials: list[Trial] = Field(default_factory=list)
    reserved_cost_usd: float = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    halted_reason: str | None = None


class TrialSubmitter(Protocol):
    def estimate_cost(self, experiment: ExperimentConfig) -> float: ...
    def submit(self, experiment: ExperimentConfig, idempotency_key: str) -> str: ...
    def stop(self, run_id: str, reason: str) -> None: ...


class RunControllerSubmitter:
    """Launches trials through ``RunController.submit`` so every run guardrail applies.

    ``stop`` records the early stop as a cancellation in the run store; halting
    the external job is left to the workflow that owns it.
    """

    def __init__(self, controller: RunController, job: VertexJobConfig):
        self.controller, self.job = controller, job

    def estimate_cost(self, experiment: ExperimentConfig) -> float:
        return self.controller.estimate_cost(self.job)

    def submit(self, experiment: ExperimentConfig, idempotency_key: str) -> str:
        record, _ = self.controller.submit(experiment=experiment, job=self.job,
                                           idempotency_key=idempotency_key)
        if record.state == RunState.FAILED:
            raise RuntimeError(f"run {record.run_id} failed at submission: "
                               f"{record.failure_code}: {record.failure_message}")
        return record.run_id

    def stop(self, run_id: str, reason: str) -> None:
        if self.controller.get(run_id).state in {RunState.SUCCEEDED, RunState.FAILED,
                                                 RunState.CANCELED}:
            return
        self.controller.update_state(run_id, RunState.CANCELED, failure_code="EARLY_STOPPED",
                                     failure_message=reason)


class ExperimentSearch:
    def __init__(self, *, base: ExperimentConfig, space: SearchSpace, budget: SearchBudget,
                 submitter: TrialSubmitter, early_stopping: EarlyStoppingPolicy | None = None,
                 state: SearchState | None = None,
                 id_factory: Callable[[], str] = lambda: uuid.uuid4().hex[:12]):
        self.base, self.space, self.budget, self.submitter = base, space, budget, submitter
        self.early_stopping = early_stopping
        base_sha256 = _sha256(base.model_dump(mode="json"))
        if state is None:
            state = SearchState(search_id=id_factory(), base_sha256=base_sha256,
                                space_sha256=space.sha256)
        elif (state.base_sha256, state.space_sha256) != (base_sha256, space.sha256):
            raise ValueError("saved search state belongs to a different base experiment or space")
        self.state = state
        # Reject a choice that is invalid on its own before any spend.
        for path, values in space.choices.items():
            for value in values:
                self._variant({path: value}, index=0)

    # Proposal

    def _variant(self, overrides: dict[str, Any], *, index: int) -> ExperimentConfig:
        data = self.base.model_dump(mode="python")
        for path, value in overrides.items():
            target = data
            *parents, leaf = path.split(".")
            for name in parents:
                target = target[name]
            target[leaf] = value
        fingerprint = _sha256(overrides)
        data["experiment_id"] = f"{self.base.experiment_id}-t{index:03d}-{fingerprint[:8]}"
        return ExperimentConfig.model_validate(data)

    def _draw(self) -> dict[str, Any] | None:
        if self.space.include_base and _sha256({}) not in self.state.seen:
            return {}
        paths = sorted(self.space.choices)
        for _ in range(_MAX_DRAWS_PER_PROPOSAL):
            if len(self.state.seen) >= self.space.size + int(self.space.include_base):
                return None
            rng = random.Random(f"{self.space.seed}:{self.state.draws}")
            self.state.draws += 1
            overrides = {path: rng.choice(self.space.choices[path]) for path in paths}
            if _sha256(overrides) not in self.state.seen:
                return overrides
        return None

    def propose(self) -> tuple[dict[str, Any], ExperimentConfig] | None:
        """Return the next unseen variant, skipping combinations the contract rejects."""
        launched = {trial.fingerprint for trial in self.state.trials}
        while (overrides := self._draw()) is not None:
            self.state.seen.append(_sha256(overrides))
            try:
                experiment = self._variant(overrides, index=len(self.state.trials))
            except ValueError:
                continue
            if _config_sha256(experiment) not in launched:
                return overrides, experiment
        return None

    # Scheduling

    def running(self) -> list[Trial]:
        return [trial for trial in self.state.trials if trial.status == TrialStatus.RUNNING]

    @property
    def remaining_usd(self) -> float:
        return self.budget.max_total_cost_usd - self.state.reserved_cost_usd

    def step(self) -> list[Trial]:
        """Launch trials into free concurrency slots while trial and cost budget remain."""
        launched: list[Trial] = []
        while (self.state.halted_reason is None
               and len(self.running()) < self.budget.max_concurrent):
            if len(self.state.trials) >= self.budget.max_trials:
                self.state.halted_reason = "trial budget exhausted"
                break
            proposal = self.propose()
            if proposal is None:
                self.state.halted_reason = "search space exhausted"
                break
            overrides, experiment = proposal
            cost = float(self.submitter.estimate_cost(experiment))
            if not 0 <= cost < float("inf"):
                raise ValueError(f"submitter returned an invalid cost estimate: {cost!r}")
            if cost > self.remaining_usd + 1e-9:
                # Return the draw so a resumed search with more budget can still try it.
                self.state.seen.pop()
                self.state.halted_reason = (
                    f"cost budget exhausted: next trial needs ${cost:.2f}, "
                    f"${self.remaining_usd:.2f} remains")
                break
            fingerprint = _config_sha256(experiment)
            trial = Trial(trial_id=experiment.experiment_id, index=len(self.state.trials),
                          overrides=overrides, fingerprint=fingerprint, experiment=experiment,
                          status=TrialStatus.RUNNING, reserved_cost_usd=cost)
            self.state.trials.append(trial)
            self.state.reserved_cost_usd += cost
            try:
                trial.run_id = self.submitter.submit(
                    experiment, f"search:{self.state.search_id}:{fingerprint}")
            except Exception as exc:  # noqa: BLE001 - recorded on the trial, never retried here
                self._record_failure(trial, f"submission failed: {exc}")
                continue
            launched.append(trial)
        return launched

    def _trial(self, trial_id: str) -> Trial:
        for trial in self.state.trials:
            if trial.trial_id == trial_id:
                return trial
        raise KeyError(f"unknown trial: {trial_id}")

    def _running_trial(self, trial_id: str) -> Trial:
        trial = self._trial(trial_id)
        if trial.status != TrialStatus.RUNNING:
            raise ValueError(f"trial {trial_id} is {trial.status}, not running")
        return trial

    def _record_failure(self, trial: Trial, reason: str) -> None:
        trial.status, trial.failure = TrialStatus.FAILED, reason[:2000]
        self.state.consecutive_failures += 1
        if self.state.consecutive_failures >= self.budget.max_consecutive_failures:
            self.state.halted_reason = (
                f"{self.state.consecutive_failures} consecutive trial failures; "
                f"last: {trial.failure}")

    # Results

    def report(self, trial_id: str, epoch: int, score: TrialScore) -> str | None:
        """Record a validation observation; returns the stop reason if the trial was stopped."""
        trial = self._running_trial(trial_id)
        if trial.curve and epoch <= trial.curve[-1].epoch:
            raise ValueError("epoch observations must be strictly increasing")
        if epoch > trial.experiment.epochs:
            raise ValueError("epoch exceeds the trial's configured epochs")
        trial.curve.append(EpochObservation(epoch=epoch, score=score))
        if self.early_stopping is None:
            return None
        peers = [other.curve for other in self.state.trials if other.trial_id != trial_id]
        reason = self.early_stopping.should_stop(trial.curve, peers)
        if reason is not None:
            trial.status, trial.stop_reason = TrialStatus.STOPPED, reason
            if trial.run_id is not None:
                self.submitter.stop(trial.run_id, reason)
        return reason

    def complete(self, trial_id: str, score: TrialScore) -> Trial:
        trial = self._running_trial(trial_id)
        trial.status, trial.final = TrialStatus.SUCCEEDED, score
        self.state.consecutive_failures = 0
        return trial

    def fail(self, trial_id: str, reason: str) -> Trial:
        trial = self._running_trial(trial_id)
        self._record_failure(trial, reason)
        return trial

    def leaderboard(self, *, include_stopped: bool = False) -> list[Trial]:
        """Succeeded trials ranked by final MCC, then macro F1; earlier trials win exact ties."""
        statuses = {TrialStatus.SUCCEEDED} | ({TrialStatus.STOPPED} if include_stopped else set())
        ranked = [trial for trial in self.state.trials
                  if trial.status in statuses and trial.best_observed is not None]
        return sorted(ranked, key=lambda trial: (tuple(-v for v in trial.best_observed.selection_key),
                                                 trial.index))

    def best(self) -> Trial | None:
        board = self.leaderboard()
        return board[0] if board else None

    @property
    def finished(self) -> bool:
        return self.state.halted_reason is not None and not self.running()


__all__ = [
    "PINNED_FIELDS",
    "SEARCHABLE_FIELDS",
    "EarlyStoppingPolicy",
    "EpochObservation",
    "ExperimentSearch",
    "RunControllerSubmitter",
    "SearchBudget",
    "SearchSpace",
    "SearchState",
    "Trial",
    "TrialScore",
    "TrialStatus",
    "TrialSubmitter",
]
