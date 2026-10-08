"""``defect goal``: hand the platform a goal and get a model or a report back."""
# Typer uses parameter defaults to define the command-line schema.
# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from defect_platform.contracts import DatasetVersion, ExperimentConfig
from defect_platform.control.goal import Goal
from defect_platform.control.goal_runner import STATE_FILE, GoalRunner, GoalState, LocalExecutor
from defect_platform.control.goal_vertex import VertexExecutor, VertexGoalConfig

goal_app = typer.Typer(help="Start a goal and track it to a model or a report.", no_args_is_help=True)


def load_goal_file(path: Path) -> tuple[Goal, ExperimentConfig, DatasetVersion]:
    """Read goal.yaml: the goal, the base experiment, and the dataset version (inline or a path)."""
    try:
        values = yaml.safe_load(path.read_text())
        dataset = values["dataset"]
        if isinstance(dataset, str):
            dataset = yaml.safe_load((path.parent / dataset).read_text())
        return (Goal.model_validate(values["goal"]),
                ExperimentConfig.model_validate(values["experiment"]),
                DatasetVersion.model_validate(dataset))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise typer.BadParameter(f"invalid goal file {path}: {exc}") from exc


def _summary(state: GoalState, goal_dir: Path) -> str:
    lines = [f"Goal {state.goal.goal_id}: {state.phase.replace('_', ' ')}"]
    if state.stop_detail:
        lines.append(f"Why: {state.stop_detail}")
    metric = state.goal.primary_metric
    for run in state.runs:
        result = (f"validation {metric} {run.score.primary:.3f}"
                  f"{' (meets goal)' if run.score.met else ''}" if run.score else run.status)
        lines.append(f"  {run.run_id} [{run.origin}] {result}")
    if state.finished:
        lines.append(f"Report: {goal_dir / 'REPORT.md'}")
    return "\n".join(lines)


def load_vertex_config(path: Path) -> VertexGoalConfig:
    try:
        return VertexGoalConfig.model_validate(yaml.safe_load(path.read_text()))
    except (OSError, TypeError, ValueError) as exc:
        raise typer.BadParameter(f"invalid Vertex config {path}: {exc}") from exc


def run_goal_file(goal_file: Path, goal_dir: Path | None = None,
                  vertex: VertexGoalConfig | None = None) -> tuple[GoalState, Path]:
    from defect_platform.dataset import load_dataset_semantics

    goal, experiment, dataset = load_goal_file(goal_file)
    goal_dir = goal_dir or goal_file.parent / "goals" / goal.goal_id
    classes = load_dataset_semantics(dataset).catalog.labels
    try:
        executor = (VertexExecutor(vertex, dataset, classes) if vertex
                    else LocalExecutor(dataset, classes))
        runner = GoalRunner(goal, experiment, dataset, classes, goal_dir, executor)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return runner.run(), goal_dir


@goal_app.command("start")
def goal_start(goal_file: Path = typer.Argument(..., exists=True, readable=True),
               goal_dir: Path | None = typer.Option(None, "--dir", help="Where runs and the report go"),
               vertex: Path | None = typer.Option(None, "--vertex", exists=True, readable=True,
                                                  help="Run each experiment on Vertex AI with this config")):
    """Run a goal until it is met or stops, then print the outcome.

    Runs train on this machine, or on Vertex AI with --vertex. Starting again
    with the same goal and directory resumes it.
    """
    state, goal_dir = run_goal_file(goal_file, goal_dir,
                                    load_vertex_config(vertex) if vertex else None)
    typer.echo(_summary(state, goal_dir))


@goal_app.command("status")
def goal_status(goal_dir: Path = typer.Argument(..., exists=True, file_okay=False)):
    """Show a goal's phase, each run's score against the goal, and the report location."""
    path = goal_dir / STATE_FILE
    if not path.exists():
        raise typer.BadParameter(f"{goal_dir} has no goal state")
    typer.echo(_summary(GoalState.model_validate_json(path.read_text()), goal_dir))


@goal_app.command("demo")
def goal_demo(directory: Path = typer.Argument(..., help="Empty directory for the demo"),
              unreachable: bool = typer.Option(False, "--unreachable",
                                               help="Make two classes indistinguishable"),
              vertex: Path | None = typer.Option(None, "--vertex", exists=True, readable=True,
                                                 help="Publish the demo to GCS and train on Vertex AI")):
    """Build a tiny CPU-sized problem and run its goal end to end."""
    from defect_platform.control.goal_demo import write_demo

    config = load_vertex_config(vertex) if vertex else None
    goal_file = write_demo(directory, unreachable=unreachable,
                           gcs_prefix=f"{config.staging_uri.rstrip('/')}/demo" if config else None)
    typer.echo(f"Demo goal written to {goal_file}")
    state, goal_dir = run_goal_file(goal_file, vertex=config)
    typer.echo(_summary(state, goal_dir))
