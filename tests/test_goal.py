"""Goal scoring and the goal runner's decisions, with a scripted trainer."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from defect_platform.analysis.evidence import evidence_from_report
from defect_platform.contracts import DatasetVersion, ExperimentConfig, ModelSpec
from defect_platform.control.goal import Goal, evaluate_goal
from defect_platform.control.goal_runner import GoalRunner, GoalState

CLASSES = ["ok", "scratch", "dent"]


def _report(experiment: ExperimentConfig, recalls: dict[str, float], support: int = 40,
            mcc: float | None = None) -> dict:
    per_class, confusion = {}, []
    for i, name in enumerate(CLASSES):
        right = round(recalls[name] * support)
        row = [0] * len(CLASSES)
        row[i] = right
        # Misses go to the next class, so every weak class has one clear confusion.
        row[(i + 1) % len(CLASSES)] = support - right
        confusion.append(row)
        per_class[name] = {"precision": 0.9, "recall": right / support,
                           "f1": right / support, "support": support}
    mean = sum(recalls.values()) / len(recalls)
    split = {"mcc": mean if mcc is None else mcc, "macro_f1": mean, "accuracy": mean,
             "per_class": per_class, "confusion": confusion}
    return {"classes": CLASSES, "validation": split, "test": split,
            "training_config": {"optimizer": experiment.optimizer, "loss": experiment.loss,
                                "focal_gamma": experiment.focal_gamma,
                                "horizontal_flip_probability": experiment.horizontal_flip_probability,
                                "class_weights": experiment.class_weights, "seed": experiment.seed,
                                "max_review_error_rate": experiment.max_review_error_rate},
            "abstention": None}


class ScriptedTrainer:
    """Writes the artifacts a real run leaves, with metrics chosen by ``outcome``."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.experiments: list[ExperimentConfig] = []

    def run(self, experiment: ExperimentConfig, run_dir: Path) -> dict:
        self.experiments.append(experiment)
        report = self.outcome(experiment)
        (run_dir / "model").mkdir(parents=True)
        (run_dir / "model" / "model.pt").write_text("weights")
        (run_dir / "evaluation.json").write_text(json.dumps(report))
        return report


def _dataset(**counts) -> DatasetVersion:
    return DatasetVersion(version_id="v1", object_slug="panel", root_uri="/data/v1",
                          manifest_uri="/data/v1/manifest.json", shard_uris={},
                          sample_counts={"train": 100, "validation": 120, "test": 120, **counts},
                          sha256="c" * 64, source_snapshot_uri="/data/v1/source.json")


def _experiment(**updates) -> ExperimentConfig:
    return ExperimentConfig(experiment_id="base", object_slug="panel", dataset_version_id="v1",
                            runtime_id="local", epochs=3,
                            model=ModelSpec(weights_uri="/weights", weights_sha256="a" * 64),
                            **updates)


def _goal(**updates) -> Goal:
    return Goal(**{"goal_id": "panel", "object_slug": "panel", "min_primary": 0.8,
                   "min_class_recall": 0.75, **updates})


def _runner(tmp_path, outcome, goal=None, **kwargs) -> tuple[GoalRunner, ScriptedTrainer]:
    trainer = ScriptedTrainer(outcome)
    runner = GoalRunner(goal or _goal(), kwargs.pop("experiment", _experiment()),
                        kwargs.pop("dataset", _dataset()), CLASSES, tmp_path / "goal", trainer,
                        **kwargs)
    return runner, trainer


def _dent_needs_weight(experiment: ExperimentConfig) -> dict:
    dent = 0.9 if experiment.class_weights.get("dent", 1.0) > 1 else 0.3
    return _report(experiment, {"ok": 0.95, "scratch": 0.95, "dent": dent})


def test_goal_scores_every_target_on_validation(tmp_path):
    experiment = _experiment()
    evidence = evidence_from_report(_report(experiment, {"ok": 0.9, "scratch": 0.5, "dent": 0.9},
                                            mcc=0.7), experiment, mlflow_run_id="r1")
    score = evaluate_goal(_goal(max_review_rate=0.1), evidence)
    assert not score.met
    assert [item.target for item in score.shortfalls] == [
        "validation mcc", "scratch recall", "review rate"]
    assert score.shortfalls[-1].achieved is None  # no calibrated threshold: everything is reviewed


def test_reachable_goal_iterates_with_analysis_and_delivers_the_model(tmp_path):
    runner, trainer = _runner(tmp_path, _dent_needs_weight)
    state = runner.run()

    assert state.phase == "model_delivered"
    assert [run.origin for run in state.runs] == ["baseline", "analysis"]
    assert trainer.experiments[1].class_weights == {"ok": 1.0, "scratch": 1.0, "dent": 2.0}
    assert [e.experiment_id for e in trainer.experiments] == ["panel-run01", "panel-run02"]
    goal_dir = tmp_path / "goal"
    assert (goal_dir / "deliverable" / "model" / "model.pt").exists()
    scorecard = json.loads((goal_dir / "scorecard.json").read_text())
    assert scorecard["outcome"] == "model_delivered"
    assert scorecard["run_id"] == "panel-run02"
    assert "test" in scorecard
    assert "model delivered" in (goal_dir / "REPORT.md").read_text()
    # The Analysis proposal went through the handoff gate and was recorded.
    assert (goal_dir / "handoffs.sqlite").exists()


def test_a_plateau_ends_in_a_report_that_names_the_confusion(tmp_path):
    runner, _ = _runner(
        tmp_path, lambda e: _report(e, {"ok": 0.95, "scratch": 0.4, "dent": 0.95}))
    state = runner.run()

    assert state.phase == "goal_not_met"
    assert state.stop_reason == "plateau"
    assert len(state.runs) == runner.goal.plateau_runs + 1
    assert {run.origin for run in state.runs[1:]} <= {"analysis", "fallback"}
    report = (tmp_path / "goal" / "REPORT.md").read_text()
    assert "**scratch** is mostly predicted as **dent**" in report
    assert "What was tried" in report
    assert not (tmp_path / "goal" / "deliverable").exists()


def test_too_few_validation_samples_stops_with_a_data_request(tmp_path):
    runner, trainer = _runner(
        tmp_path, lambda e: _report(e, {"ok": 1.0, "scratch": 0.5, "dent": 1.0}, support=4))
    state = runner.run()

    assert state.stop_reason == "data_limited"
    assert len(trainer.experiments) == 1
    assert state.data_needs[0]["class_name"] == "scratch"
    assert "Add labeled examples of **scratch**" in (tmp_path / "goal" / "REPORT.md").read_text()


def test_run_limit_stops_a_goal_that_is_still_improving(tmp_path):
    def improving(experiment):
        recall = 0.3 + 0.1 * len(trainer.experiments)
        return _report(experiment, {"ok": 0.95, "scratch": recall, "dent": 0.95})

    runner, trainer = _runner(tmp_path, improving, goal=_goal(max_runs=3))
    state = runner.run()
    assert state.stop_reason == "run_limit"
    assert len(state.runs) == 3


def test_a_passed_deadline_launches_nothing(tmp_path):
    now = datetime(2026, 10, 8, tzinfo=UTC)
    runner, trainer = _runner(tmp_path, _dent_needs_weight,
                              goal=_goal(deadline=now - timedelta(hours=1)), clock=lambda: now)
    state = runner.run()
    assert state.stop_reason == "deadline"
    assert trainer.experiments == []


def test_a_failed_run_ends_the_goal_with_its_error(tmp_path):
    def broken(experiment):
        raise ValueError("class weights must match the catalog labels")

    runner, _ = _runner(tmp_path, broken)
    state = runner.run()
    assert state.stop_reason == "run_failed"
    assert "class weights must match" in state.runs[0].error
    assert "start the goal again" in (tmp_path / "goal" / "REPORT.md").read_text()


def test_an_empty_split_is_reported_before_any_training(tmp_path):
    runner, trainer = _runner(tmp_path, _dent_needs_weight, dataset=_dataset(validation=0))
    state = runner.run()
    assert state.stop_reason == "dataset_infeasible"
    assert trainer.experiments == []


def test_goal_resumes_from_its_saved_state(tmp_path):
    runner, trainer = _runner(tmp_path, _dent_needs_weight)
    # Simulate a crash after the baseline finished and the next run started.
    runner.run()
    state_path = tmp_path / "goal" / "state.json"
    state = GoalState.model_validate_json(state_path.read_text())
    state.phase, state.stop_reason = "running", None
    state.runs[1].status, state.runs[1].score = "running", None
    state_path.write_text(state.model_dump_json())

    resumed, trainer = _runner(tmp_path, _dent_needs_weight)
    assert resumed.run().phase == "model_delivered"
    assert [e.experiment_id for e in trainer.experiments] == ["panel-run02"]

    finished, trainer = _runner(tmp_path, _dent_needs_weight)
    assert finished.run().phase == "model_delivered"
    assert trainer.experiments == []

    with pytest.raises(ValueError, match="different goal"):
        _runner(tmp_path, _dent_needs_weight, goal=_goal(min_primary=0.9))


def test_runner_rejects_a_review_target_the_trainer_cannot_measure(tmp_path):
    with pytest.raises(ValueError, match="max_review_error_rate"):
        _runner(tmp_path, _dent_needs_weight, goal=_goal(max_review_rate=0.1))
