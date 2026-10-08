"""Drive one goal from a baseline run to a delivered model or a "why not" report.

The goal runner is deterministic control-plane code, not an LLM session. Each
step reads the finished run's validation evidence, scores it with the goal's
one objective, and either delivers, stops with a reason, or launches the next
experiment. Next experiments come from the Analysis plane through the handoff
gate, then from a short ladder of admissible hyperparameter changes. Both
paths may change only the fields Analysis is allowed to tune.

State is written to ``<goal_dir>/state.json`` after every step, so an
interrupted goal resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from ..analysis.agent import (
    PROPOSE_CAPABILITY,
    AnalysisAgent,
    AnalysisPolicy,
    ExperimentProposal,
    config_changes,
    validate_experiment_proposal,
)
from ..analysis.evidence import RunEvidence, evidence_from_report
from ..contracts import DatasetVersion, ExperimentConfig, StrictModel
from ..planes import HandoffState, Plane, PlaneAgentContext, SQLiteHandoffLedger, gate_handoff
from .goal import Goal, GoalScore, evaluate_goal

logger = logging.getLogger(__name__)

STATE_FILE = "state.json"
MAX_EPOCHS = 100
MAX_UNFREEZE = 4
MIN_LEARNING_RATE = 1e-6

StopReason = Literal["goal_met", "run_limit", "deadline", "plateau", "data_limited",
                     "no_admissible_change", "run_failed", "dataset_infeasible"]


class RunExecutor(Protocol):
    def run(self, experiment: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
        """Train one experiment, leave its artifacts in run_dir, return evaluation.json."""


class LocalExecutor:
    """Train in this process on the local CPU or GPU, the way a Vertex job would."""

    def __init__(self, dataset: DatasetVersion, classes: list[str],
                 train: Callable[..., dict[str, Any]] | None = None):
        self.dataset = dataset
        self.classes = list(classes)
        self._train = train

    def run(self, experiment: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
        train = self._train
        if train is None:
            from ..trainer.runner import run_request as train
        request = {"experiment": experiment.model_dump(mode="json"),
                   "dataset": self.dataset.model_dump(mode="json"),
                   "classes": self.classes, "output_uri": str(run_dir)}
        with tempfile.TemporaryDirectory(prefix="defect-goal-run-") as work:
            return train(request, work)


class GoalRunRecord(StrictModel):
    run_id: str
    experiment: ExperimentConfig
    origin: Literal["baseline", "analysis", "fallback"]
    rationale: tuple[str, ...] = ()
    status: Literal["running", "succeeded", "failed"]
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    score: GoalScore | None = None


class GoalState(StrictModel):
    goal: Goal
    goal_sha256: str
    dataset_version_id: str
    phase: Literal["running", "model_delivered", "goal_not_met"] = "running"
    stop_reason: StopReason | None = None
    stop_detail: str = ""
    best_run_id: str | None = None
    runs: list[GoalRunRecord] = Field(default_factory=list)
    data_needs: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: datetime

    @property
    def finished(self) -> bool:
        return self.phase != "running"


def _fingerprint(experiment: ExperimentConfig) -> str:
    values = experiment.model_dump(mode="json")
    values.pop("experiment_id")
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class GoalRunner:
    def __init__(self, goal: Goal, base_experiment: ExperimentConfig, dataset: DatasetVersion,
                 classes: list[str], goal_dir: str | Path, executor: RunExecutor,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)):
        if base_experiment.object_slug != goal.object_slug or dataset.object_slug != goal.object_slug:
            raise ValueError("goal, experiment and dataset must name the same object")
        if base_experiment.dataset_version_id != dataset.version_id:
            raise ValueError("base experiment does not use the goal's dataset version")
        if goal.max_review_rate is not None and base_experiment.max_review_error_rate is None:
            raise ValueError("a review-rate target needs max_review_error_rate in the experiment")
        self.goal = goal
        self.base = base_experiment
        self.dataset = dataset
        self.classes = list(classes)
        self.dir = Path(goal_dir)
        self.executor = executor
        self.clock = clock
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger = SQLiteHandoffLedger(self.dir / "handoffs.sqlite")
        self.analysis_context = PlaneAgentContext(agent_id=f"goal-{goal.goal_id}-analysis",
                                                  plane=Plane.ANALYSIS,
                                                  capabilities=frozenset({PROPOSE_CAPABILITY}))
        self.control_context = PlaneAgentContext(agent_id=f"goal-{goal.goal_id}-control",
                                                 plane=Plane.CONTROL,
                                                 capabilities=frozenset({"receive_handoff"}))
        self.policy = AnalysisPolicy(
            min_class_recall=goal.min_class_recall or AnalysisPolicy().min_class_recall,
            min_class_support=goal.min_validation_support,
            max_review_rate=goal.max_review_rate if goal.max_review_rate is not None else 1.0)
        self.state = self._load_state()

    # -- persistence -----------------------------------------------------

    def _load_state(self) -> GoalState:
        path = self.dir / STATE_FILE
        if not path.exists():
            return GoalState(goal=self.goal, goal_sha256=self.goal.sha256,
                             dataset_version_id=self.dataset.version_id, updated_at=self.clock())
        state = GoalState.model_validate_json(path.read_text())
        if state.goal_sha256 != self.goal.sha256 or state.dataset_version_id != self.dataset.version_id:
            raise ValueError(f"{self.dir} holds a different goal or dataset; use a new goal directory")
        # A run interrupted mid-training left no trustworthy result; train it again.
        state.runs = [run for run in state.runs if run.status != "running"]
        return state

    def _save(self) -> None:
        self.state.updated_at = self.clock()
        path = self.dir / STATE_FILE
        temporary = path.with_suffix(".tmp")
        temporary.write_text(self.state.model_dump_json(indent=2))
        temporary.replace(path)

    def run_dir(self, run_id: str) -> Path:
        return self.dir / "runs" / run_id

    def evidence(self, record: GoalRunRecord) -> RunEvidence:
        report = json.loads((self.run_dir(record.run_id) / "evaluation.json").read_text())
        return evidence_from_report(report, record.experiment, mlflow_run_id=record.run_id)

    # -- the loop --------------------------------------------------------

    def run(self) -> GoalState:
        if self.state.finished:
            return self.state
        if not self.state.runs:
            empty = sorted(split for split in ("train", "validation", "test")
                           if not self.dataset.sample_counts.get(split))
            if empty:
                self._stop("dataset_infeasible",
                           f"the dataset has no samples in: {', '.join(empty)}")
                return self.state
        next_run: tuple[ExperimentConfig, str, tuple[str, ...]] | None = None
        if not self.state.runs:
            next_run = (self.base, "baseline", ("baseline with the configured experiment",))
        else:
            next_run = self._decide()
        while next_run is not None:
            experiment, origin, rationale = next_run
            if len(self.state.runs) >= self.goal.max_runs:
                self._stop("run_limit", f"reached the limit of {self.goal.max_runs} runs")
                break
            if self.goal.deadline is not None and self.clock() >= self.goal.deadline:
                self._stop("deadline", f"the deadline {self.goal.deadline.isoformat()} passed")
                break
            record = self._execute(experiment, origin, rationale)
            if record.status == "failed":
                self._stop("run_failed", f"{record.run_id} failed: {record.error}")
                break
            next_run = self._decide()
        return self.state

    def _execute(self, experiment: ExperimentConfig, origin: str,
                 rationale: tuple[str, ...]) -> GoalRunRecord:
        run_id = f"{self.goal.goal_id}-run{len(self.state.runs) + 1:02d}"
        experiment = experiment.model_copy(update={"experiment_id": run_id})
        record = GoalRunRecord(run_id=run_id, experiment=experiment, origin=origin,
                               rationale=rationale, status="running", started_at=self.clock())
        self.state.runs.append(record)
        self._save()
        run_dir = self.run_dir(run_id)
        shutil.rmtree(run_dir, ignore_errors=True)
        logger.info("goal_run_started", extra={"stage": "goal_run_started", "run_id": run_id})
        try:
            self.executor.run(experiment, run_dir)
            record.score = evaluate_goal(self.goal, self.evidence(record))
            record.status = "succeeded"
        except Exception as exc:  # noqa: BLE001 - any trainer failure ends the goal with evidence
            record.status, record.error = "failed", f"{type(exc).__name__}: {exc}"[:2000]
        record.finished_at = self.clock()
        self._save()
        return record

    def _succeeded(self) -> list[GoalRunRecord]:
        return [run for run in self.state.runs if run.status == "succeeded"]

    def _best(self) -> GoalRunRecord:
        return max(self._succeeded(), key=lambda run: run.score.rank_key)

    def _decide(self) -> tuple[ExperimentConfig, str, tuple[str, ...]] | None:
        """Deliver, stop, or return the next experiment to run."""
        best = self._best()
        self.state.best_run_id = best.run_id
        if best.score.met:
            return self._deliver(best)
        if self._plateaued():
            return self._stop("plateau", f"{self.goal.plateau_runs} runs in a row did not improve "
                              f"validation {self.goal.primary_metric} by {self.goal.min_improvement}")
        tried = {_fingerprint(run.experiment) for run in self.state.runs}
        evidence = self.evidence(best)
        result = AnalysisAgent(self.analysis_context, self.policy).analyze(evidence)
        if result.kind == "data_request":
            self.state.data_needs = [need for need in result.handoff.payload["needs"]]
            return self._stop("data_limited", "; ".join(result.rationale))
        if result.kind == "experiment_proposal":
            decision = gate_handoff(result.handoff, Plane.CONTROL, self.control_context,
                                    validator=validate_experiment_proposal, ledger=self.ledger)
            if decision.state == HandoffState.ACCEPTED:
                proposal = ExperimentProposal.model_validate(result.handoff.payload)
                if _fingerprint(proposal.experiment) not in tried:
                    return proposal.experiment, "analysis", result.rationale
        for experiment, reason in self._fallbacks(best.experiment):
            if _fingerprint(experiment) not in tried:
                return experiment, "fallback", (reason,)
        return self._stop("no_admissible_change",
                          "every admissible hyperparameter change has been tried")

    def _plateaued(self) -> bool:
        primaries = [run.score.primary for run in self._succeeded()]
        window = self.goal.plateau_runs
        if len(primaries) <= window:
            return False
        return max(primaries[-window:]) < max(primaries[:-window]) + self.goal.min_improvement

    def _fallbacks(self, parent: ExperimentConfig) -> list[tuple[ExperimentConfig, str]]:
        """Admissible single changes to try once Analysis has nothing new to propose."""
        candidates: list[tuple[dict[str, Any], str]] = []
        if parent.epochs < MAX_EPOCHS:
            epochs = min(parent.epochs * 2, MAX_EPOCHS)
            candidates.append(({"epochs": epochs}, f"train longer: epochs {parent.epochs} -> {epochs}"))
        if parent.learning_rate / 2 >= MIN_LEARNING_RATE:
            rate = parent.learning_rate / 2
            candidates.append(({"learning_rate": rate},
                               f"lower learning rate {parent.learning_rate:g} -> {rate:g}"))
        if parent.model.unfreeze_last_n < MAX_UNFREEZE:
            depth = parent.model.unfreeze_last_n + 1
            candidates.append(({"model": parent.model.model_copy(update={"unfreeze_last_n": depth})},
                               f"fine-tune the last {depth} backbone layer(s)"))
        if parent.loss == "cross_entropy":
            candidates.append(({"loss": "focal"}, "switch to focal loss"))
        admissible = []
        for update, reason in candidates:
            experiment = parent.model_copy(update={**update,
                                                   "experiment_id": parent.experiment_id + "-next"})
            proposal = {"parent_experiment": parent.model_dump(mode="json"),
                        "experiment": experiment.model_dump(mode="json"),
                        "source_mlflow_run_id": parent.experiment_id,
                        "changes": {key: value for key, value
                                    in config_changes(parent, experiment).items()
                                    if key != "experiment_id"},
                        "rationale": [reason]}
            if not validate_experiment_proposal(proposal):
                admissible.append((experiment, reason))
        return admissible

    # -- outcomes --------------------------------------------------------

    def _stop(self, reason: StopReason, detail: str) -> None:
        self.state.phase = "goal_not_met"
        self.state.stop_reason, self.state.stop_detail = reason, detail
        if self._succeeded():
            self.state.best_run_id = self._best().run_id
        from .goal_report import write_goal_not_met
        write_goal_not_met(self)
        self._save()
        logger.info("goal_not_met", extra={"stage": "goal_not_met", "reason": reason})

    def _deliver(self, best: GoalRunRecord) -> None:
        self.state.phase, self.state.stop_reason = "model_delivered", "goal_met"
        self.state.stop_detail = f"{best.run_id} met every target"
        from .goal_report import write_model_delivered
        write_model_delivered(self, best)
        self._save()
        logger.info("goal_met", extra={"stage": "goal_met", "run_id": best.run_id})
