"""The Cloud Workflows definition parses and agrees with the controller that starts it."""

from pathlib import Path

import yaml

from defect_platform.control.controller import GoogleWorkflowsStarter

WORKFLOW = Path(__file__).parents[1] / "infra" / "workflow.yaml"


def _steps() -> dict:
    definition = yaml.safe_load(WORKFLOW.read_text())
    return {name: body for step in definition["main"]["steps"] for name, body in step.items()}


def test_workflow_parses_and_takes_the_polling_budget_from_its_arguments():
    steps = _steps()
    assert {"submit_vertex", "poll_vertex", "classify_state", "wait",
            "record_success", "record_failure", "record_timeout"} <= set(steps)
    assignments = {key: value for item in steps["initialize"]["assign"]
                   for key, value in item.items()}
    assert assignments["max_polls"] == "${args.max_polls}"


def test_workflow_sleep_matches_the_controller_polling_interval():
    # polls_for() converts max_run_hours into a poll count using POLL_SECONDS; if
    # the workflow slept longer, the window would no longer cover the job.
    assert _steps()["wait"]["args"]["seconds"] == GoogleWorkflowsStarter.POLL_SECONDS
