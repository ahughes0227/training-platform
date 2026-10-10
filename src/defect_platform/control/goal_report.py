"""The two goal deliverables: a model package with its scorecard, or a "why not" report.

Both are written to the goal directory as ``scorecard.json`` (machine-readable)
and ``REPORT.md`` (for the manager). Only the delivered candidate's test-split
results are read, once, here; every earlier decision used validation data.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from .goal import Goal, GoalScore

if TYPE_CHECKING:
    from .goal_runner import GoalRunner, GoalRunRecord

DELIVERABLE_DIR = "deliverable"


def _targets(goal: Goal) -> list[str]:
    targets = [f"validation {goal.primary_metric} >= {goal.min_primary}"]
    if goal.min_class_recall is not None:
        targets.append(f"every class recall >= {goal.min_class_recall}")
    if goal.max_review_rate is not None:
        targets.append(f"review rate <= {goal.max_review_rate}")
    return targets


def _runs_table(runner: GoalRunner) -> list[str]:
    metric = runner.goal.primary_metric
    lines = [f"| Run | Origin | Change | Validation {metric} | Met |", "| --- | --- | --- | --- | --- |"]
    for run in runner.state.runs:
        value = f"{run.score.primary:.3f}" if run.score else f"failed: {run.error}"
        met = "yes" if run.score and run.score.met else "no"
        change = "; ".join(run.rationale) or "-"
        lines.append(f"| {run.run_id} | {run.origin} | {change} | {value} | {met} |")
    return lines


def _per_class(report: dict[str, Any], classes: list[str], split: str) -> list[str]:
    per_class = report[split]["per_class"]
    confusion = report[split]["confusion"]
    lines = ["| Class | Recall | Precision | Samples | Most often mistaken for |",
             "| --- | --- | --- | --- | --- |"]
    for i, name in enumerate(classes):
        metrics = per_class[name]
        wrong = sorted(((count, classes[j]) for j, count in enumerate(confusion[i])
                        if j != i and count), reverse=True)
        mistaken = ", ".join(f"{other} ({count})" for count, other in wrong[:2]) or "-"
        lines.append(f"| {name} | {metrics['recall']:.3f} | {metrics['precision']:.3f} | "
                     f"{int(metrics['support'])} | {mistaken} |")
    return lines


def _report(runner: GoalRunner, run_id: str) -> dict[str, Any]:
    return json.loads((runner.run_dir(run_id) / "evaluation.json").read_text())


def _score_dict(score: GoalScore | None) -> dict[str, Any] | None:
    return score.model_dump(mode="json") if score else None


def write_model_delivered(runner: GoalRunner, best: GoalRunRecord) -> None:
    goal = runner.goal
    report = _report(runner, best.run_id)
    package = runner.dir / DELIVERABLE_DIR
    shutil.rmtree(package, ignore_errors=True)
    shutil.copytree(runner.run_dir(best.run_id) / "model", package / "model")
    (package / "evaluation.json").write_text(json.dumps(report, indent=2))
    test = report["test"]
    scorecard = {
        "outcome": "model_delivered", "goal": goal.model_dump(mode="json"),
        "run_id": best.run_id, "experiment": best.experiment.model_dump(mode="json"),
        "model_dir": str(package / "model"), "validation": _score_dict(best.score),
        "test": {"mcc": test["mcc"], "macro_f1": test["macro_f1"], "accuracy": test["accuracy"],
                 "per_class": test["per_class"]},
        "runs": len(runner.state.runs),
        "executed_on": best.executed_on, "releasable": best.releasable,
    }
    (runner.dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2))
    lines = [
        f"# Goal {goal.goal_id}: model delivered", "",
        f"{best.run_id} met every target on validation data after {len(runner.state.runs)} run(s).",
        f"The model is in `{package / 'model'}`. {_release_note(best)}", "",
        "## Targets", "", *[f"- {target}" for target in _targets(goal)], "",
        "## Result on held-out test data", "",
        "The test split was read once, for this model only, after it was chosen on validation data.", "",
        f"- MCC {test['mcc']:.3f}, macro F1 {test['macro_f1']:.3f}, accuracy {test['accuracy']:.3f}", "",
        *_per_class(report, runner.classes, "test"), "",
        "## Runs", "", *_runs_table(runner), "",
    ]
    (runner.dir / "REPORT.md").write_text("\n".join(lines))


def _release_note(run: GoalRunRecord) -> str:
    if run.releasable:
        return f"It was trained on {run.executed_on} and is ready to stage for release."
    return (f"It was trained on {run.executed_on}, not a certified runtime image, so it is "
            "for evaluation only and cannot be staged for release.")


def _recommendations(runner: GoalRunner, best: GoalRunRecord | None) -> list[str]:
    goal, state = runner.goal, runner.state
    advice = []
    for need in state.data_needs:
        confused = ", ".join(need.get("confused_with") or ()) or "other classes"
        if need["reason"] == "insufficient_support":
            advice.append(f"Add labeled examples of **{need['class_name']}**: validation has "
                          f"{need['validation_support']}, and at least {goal.min_validation_support} "
                          f"are needed to tell tuning from noise. It is mostly confused with {confused}.")
        else:
            advice.append(f"Re-check the labels of **{need['class_name']}**: "
                          f"{len(need.get('sample_ids') or ())} samples look mislabeled.")
    if best is not None and best.score is not None and not state.data_needs:
        evidence = runner.evidence(best)
        for shortfall in best.score.shortfalls:
            name = shortfall.target.removesuffix(" recall")
            if name in evidence.classes:
                confused = evidence.confused_with(name)
                if confused:
                    other = confused[0][0]
                    advice.append(
                        f"**{name}** is mostly predicted as **{other}**, and tuning did not separate "
                        f"them. Add examples that show the difference, check the two class "
                        f"definitions for overlap, or merge them if they are the same defect.")
        relaxed = [f"{item.target} {item.achieved:.3f}" for item in best.score.shortfalls
                   if item.achieved is not None]
        if relaxed:
            advice.append("If the current data is all there is, the best achievable here is "
                          + " and ".join(relaxed) + ".")
    if state.stop_reason == "run_limit":
        advice.append(f"Allow more than {goal.max_runs} runs; results had not stopped improving.")
    if state.stop_reason == "deadline":
        advice.append("Extend the deadline.")
    if state.stop_reason == "run_failed":
        advice.append("Fix the failing run (its error is above) and start the goal again; it resumes.")
    if state.stop_reason == "dataset_infeasible":
        advice.append("Rebuild the dataset so every split has samples of every class.")
    return advice or ["No change inside this goal's limits is likely to reach it."]


def write_goal_not_met(runner: GoalRunner) -> None:
    goal, state = runner.goal, runner.state
    best = next((run for run in state.runs if run.run_id == state.best_run_id), None)
    scorecard = {
        "outcome": "goal_not_met", "goal": goal.model_dump(mode="json"),
        "stop_reason": state.stop_reason, "stop_detail": state.stop_detail,
        "best_run_id": best.run_id if best else None,
        "best_validation": _score_dict(best.score) if best else None,
        "data_needs": state.data_needs, "runs": len(state.runs),
    }
    (runner.dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2))
    lines = [f"# Goal {goal.goal_id}: not met", "",
             f"**Why it stopped:** {state.stop_detail}.", "",
             "## Targets", "", *[f"- {target}" for target in _targets(goal)], ""]
    if best is not None and best.score is not None:
        lines += ["## Closest result", "",
                  (f"{best.run_id} came closest, with validation {goal.primary_metric} "
                   f"{best.score.primary:.3f}. Its model is in "
                   f"`{runner.run_dir(best.run_id) / 'model'}` for inspection; it is not delivered."),
                  "",
                  "| Target | Required | Best achieved |", "| --- | --- | --- |",
                  *[f"| {item.target} | {item.required} | "
                    f"{'none' if item.achieved is None else f'{item.achieved:.3f}'} |"
                    for item in best.score.shortfalls], "",
                  "Per class on validation data:", "",
                  *_per_class(_report(runner, best.run_id), runner.classes, "validation"), ""]
    lines += ["## What would most likely reach the goal", "",
              *[f"- {item}" for item in _recommendations(runner, best)], ""]
    if state.runs:
        lines += ["## What was tried", "", *_runs_table(runner), ""]
    (runner.dir / "REPORT.md").write_text("\n".join(lines))
