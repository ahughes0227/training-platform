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
