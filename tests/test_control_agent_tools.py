from __future__ import annotations

import pytest
import yaml
from test_control import FakeVertex, fixtures

from defect_platform.contracts import ObjectSpec, RunState
from defect_platform.control.agent_tools import (
    AgentToolbox,
    ControllerBackend,
    ReadOnlyStoreBackend,
    ToolError,
)
from defect_platform.control.controller import RunController
from defect_platform.control.store import SQLiteRunStore
from defect_platform.skills import skill_definitions


def _write(path, value):
    path.write_text(yaml.safe_dump(value.model_dump(mode="json"), sort_keys=False))


def _setup(tmp_path, *, catalogs=None):
    runtime, experiment, dataset, job, catalog, trusted = fixtures(tmp_path)
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    vertex = FakeVertex()
    controller = RunController(store=store, runtimes=catalog, datasets=catalog, vertex=vertex,
                               catalogs=trusted if catalogs is None else catalogs)
    root = tmp_path / "projects"
    project = root / "panel"
    project.mkdir(parents=True)
    _write(project / "object.yaml", ObjectSpec(slug="panel", display_name="Panel",
                                               classes=["crack", "dent"]))
    _write(project / "experiment.yaml", experiment)
    _write(project / "vertex.yaml", job)
    _write(project / "dataset-version.yaml", dataset)
    _write(project / "runtime.yaml", runtime)
    toolbox = AgentToolbox(projects_root=root, backend=ControllerBackend(controller))
    return toolbox, store, vertex, project, runtime


def test_tools_map_to_catalog_skills_and_respect_mutation_flags():
    skills = {skill.name: skill for skill in skill_definitions()}
    toolbox = AgentToolbox(projects_root="unused", backend=ReadOnlyStoreBackend(lambda: None))
    names = {tool.name for tool in toolbox.tools()}
    assert {"create_project", "plan_run", "launch_run", "track_runs",
            "compare_experiments"} <= names
    for tool in toolbox.tools():
        assert tool.skill in skills
        assert not tool.mutation or skills[tool.skill].mutation
        assert tool.input_schema()["type"] == "object"
        assert tool.input_schema().get("additionalProperties") is False
    with pytest.raises(ToolError, match="invalid arguments"):
        toolbox.call("plan_run", {"project": "panel", "budget": 1000})


def test_create_project_writes_draft_facade_and_refuses_overwrite(tmp_path):
    toolbox = AgentToolbox(projects_root=tmp_path / "projects",
                           backend=ReadOnlyStoreBackend(lambda: SQLiteRunStore(tmp_path / "r.db")))
    created = toolbox.call("create_project", {"display_name": "Door Panel",
                                              "classes": ["scratch", "dent", "crack"]})
    root = tmp_path / "projects" / "door-panel"
    assert created["project"] == "door-panel" and created["class_catalog"] == "draft"
    obj = yaml.safe_load((root / "object.yaml").read_text())
    assert obj["classes"] == ["scratch", "dent", "crack"]
    assert obj["class_catalog"]["review_status"] == "draft"
    assert yaml.safe_load((root / "project.yaml").read_text())["project_id"] == \
        created["project_ref"]["project_id"]

    listed = toolbox.call("list_projects")["projects"]
    assert [item["state"] for item in listed] == ["data_ingestion"]
    plan = toolbox.call("plan_run", {"project": "door-panel"})
    assert plan["ready"] is False and plan["plan_id"] is None
    assert any("CHANGE_ME" in blocker or "missing" in blocker for blocker in plan["blockers"])
    with pytest.raises(ToolError, match="already exists"):
        toolbox.call("create_project", {"display_name": "Door Panel", "classes": ["a", "b"]})
    with pytest.raises(ToolError, match="project not found"):
        toolbox.call("view_project", {"project": "../door-panel"})


def test_plan_then_launch_uses_controller_and_is_idempotent(tmp_path):
    toolbox, store, vertex, _project, runtime = _setup(tmp_path)
    plan = toolbox.call("plan_run", {"project": "panel"})
    assert plan["ready"] and plan["blockers"] == []
    assert plan["estimated_cost_usd"] == 20 and plan["max_run_cost_usd"] == 25
    assert plan["runtime_image_digest"] == runtime.image_digest
    assert store.list() == []

    launched = toolbox.call("launch_run", {"project": "panel", "plan_id": plan["plan_id"]})
    again = toolbox.call("launch_run", {"project": "panel", "plan_id": plan["plan_id"]})
    assert launched["created"] and not again["created"]
    assert launched["run"]["state"] == RunState.SUBMITTED.value
    assert launched["run"]["runtime_image_digest"] == runtime.image_digest
    assert again["run"]["run_id"] == launched["run"]["run_id"]
    assert len(vertex.calls) == 1
    assert vertex.calls[0]["runtime"].image_digest == runtime.image_digest

    tracked = toolbox.call("track_runs", {"run_id": launched["run"]["run_id"]})
    assert tracked["run"]["run_id"] == launched["run"]["run_id"]
    view = toolbox.call("view_project", {"project": "panel"})
    assert view["state"] == "training" and len(view["runs"]) == 1


def test_launch_refuses_changed_files_and_over_budget(tmp_path):
    toolbox, store, vertex, project, _runtime = _setup(tmp_path)
    plan = toolbox.call("plan_run", {"project": "panel"})
    experiment = yaml.safe_load((project / "experiment.yaml").read_text())
    experiment["epochs"] = 500
    (project / "experiment.yaml").write_text(yaml.safe_dump(experiment))
    with pytest.raises(ToolError, match="changed since plan_run"):
        toolbox.call("launch_run", {"project": "panel", "plan_id": plan["plan_id"]})

    job = yaml.safe_load((project / "vertex.yaml").read_text())
    job["max_run_cost_usd"] = 19
    (project / "vertex.yaml").write_text(yaml.safe_dump(job))
    plan = toolbox.call("plan_run", {"project": "panel"})
    assert not plan["ready"] and "exceeds run cap" in plan["blockers"][0]
    with pytest.raises(ToolError, match="exceeds run cap"):
        toolbox.call("launch_run", {"project": "panel", "plan_id": plan["plan_id"]})
    assert store.list() == [] and vertex.calls == []


def test_controller_remains_authority_when_plan_looks_ready(tmp_path):
    class NoTrustedCatalog:
        def get_catalog(self, _object_slug, _sha256):
            return None

    toolbox, store, vertex, _project, _runtime = _setup(tmp_path, catalogs=NoTrustedCatalog())
    plan = toolbox.call("plan_run", {"project": "panel"})
    assert plan["ready"]
    with pytest.raises(ToolError, match="class catalog"):
        toolbox.call("launch_run", {"project": "panel", "plan_id": plan["plan_id"]})
    assert store.list() == [] and vertex.calls == []


def test_launch_without_control_service_is_refused(tmp_path):
    _toolbox, _store, _vertex, project, _runtime = _setup(tmp_path)
    toolbox = AgentToolbox(projects_root=project.parent,
                           backend=ReadOnlyStoreBackend(lambda: SQLiteRunStore(tmp_path / "r.db")))
    plan = toolbox.call("plan_run", {"project": "panel"})
    with pytest.raises(ToolError, match="DEFECT_CONTROL_SERVICE_URL"):
        toolbox.call("launch_run", {"project": "panel", "plan_id": plan["plan_id"]})


def test_compare_experiments_reports_differences_and_ranking(tmp_path):
    toolbox, store, _vertex, project, _runtime = _setup(tmp_path)
    run_ids = []
    for learning_rate in (0.001, 0.0003):
        experiment = yaml.safe_load((project / "experiment.yaml").read_text())
        experiment.update(learning_rate=learning_rate, experiment_id=f"exp-{learning_rate}")
        (project / "experiment.yaml").write_text(yaml.safe_dump(experiment))
        plan = toolbox.call("plan_run", {"project": "panel"})
        run_ids.append(toolbox.call("launch_run", {"project": "panel",
                                                   "plan_id": plan["plan_id"]})["run"]["run_id"])
    for run_id in run_ids:
        store.update(store.get(run_id).model_copy(update={"mlflow_run_id": f"ml-{run_id}"}))

    class Metrics:
        def metrics(self, run):
            return {"macro_f1": 0.9 if run.experiment_id == "exp-0.0003" else 0.8}

    toolbox.metrics = Metrics()
    result = toolbox.call("compare_experiments", {"run_ids": run_ids, "best_by": "macro_f1"})
    assert set(result["differences"]) == {"learning_rate"}
    assert result["ranking"]["run_ids"] == [run_ids[1], run_ids[0]]
    assert result["runs"][0]["parameters"]["epochs"] == 10
