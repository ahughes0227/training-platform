import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "role_guard", Path(__file__).parents[1] / "scripts" / "role_guard.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_trainer_cannot_build_image():
    assert module.policy_decision("trainer", {
        "tool_name": "run_in_terminal", "tool_input": {"command": "docker build ."}
    })


def test_experiment_can_inspect_but_not_edit_dockerfile():
    assert module.policy_decision("experiment", {
        "tool_name": "read_file", "tool_input": {"path": "infra/docker/trainer.Dockerfile"}
    }) is None
    assert module.policy_decision("experiment", {
        "tool_name": "edit_file", "tool_input": {"path": "infra/docker/trainer.Dockerfile"}
    })


def test_runtime_can_edit_dockerfile_but_cannot_submit_vertex():
    assert module.policy_decision("runtime", {
        "tool_name": "edit_file", "tool_input": {"path": "infra/docker/trainer.Dockerfile"}
    }) is None
    assert module.policy_decision("runtime", {
        "tool_name": "run_in_terminal", "tool_input": {"command": "gcloud ai custom-jobs create"}
    })


def test_analysis_owns_only_analysis_and_cannot_submit_or_release():
    assert module.policy_decision("analysis", {
        "tool_name": "edit_file", "tool_input": {"path": "src/defect_platform/analysis/agent.py"}
    }) is None
    assert module.policy_decision("analysis", {
        "tool_name": "edit_file", "tool_input": {"path": "src/defect_platform/control/controller.py"}
    })
    for command in ("defect run submit exp.yaml", "defect release promote r1",
                    "defect dataset build spec.yaml", "gcloud ai custom-jobs create"):
        assert module.policy_decision("analysis", {
            "tool_name": "run_in_terminal", "tool_input": {"command": command}
        }), command


def test_write_tools_naming_file_path_are_bounded_by_ownership():
    # Claude Code's Write/Edit tools name their argument file_path.
    assert module.policy_decision("analysis", {
        "tool_name": "Write", "tool_input": {"file_path": "src/defect_platform/analysis/x.py"}
    }) is None
    assert module.policy_decision("analysis", {
        "tool_name": "Write", "tool_input": {"file_path": "src/defect_platform/trainer/training.py"}
    })
    assert module.policy_decision("trainer", {
        "tool_name": "Edit", "tool_input": {"file_path": "infra/docker/trainer.Dockerfile"}
    })


def test_platform_tools_that_launch_runs_are_denied_outside_the_runner_role():
    launch = {"tool_name": "mcp__training-platform__launch_run", "tool_input": {"project": "panel"}}
    for role in ("analysis", "trainer", "runtime", "certifier"):
        assert module.policy_decision(role, launch)
    assert module.policy_decision("experiment", launch) is None
    assert module.policy_decision("analysis", {
        "tool_name": "mcp__training-platform__track_runs", "tool_input": {}
    }) is None


def test_every_cli_submission_path_is_denied_not_only_run_submit():
    for command in ("defect train start projects/panel/", "defect train guided 'notes'",
                    "defect run submit request.yaml"):
        assert module.policy_decision("analysis", {
            "tool_name": "run_in_terminal", "tool_input": {"command": command}
        })
