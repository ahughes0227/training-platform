"""Agent tools over the skill catalog.

Each tool adapts one catalog skill (``defect_platform.skills``) to an existing
authority: project folders, the durable control service, and the run store. The
tools hold no admission policy of their own. Spend caps, certified image digests,
reviewed class catalogs, and idempotency are enforced by ``RunController.submit``,
whether it runs behind the control API or is injected for local use.

Launching is two steps. ``plan_run`` reads the project and returns a ``plan_id``
fingerprint of the exact request; ``launch_run`` submits only when the project
still produces that fingerprint, so an agent cannot launch a request it has not seen.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from defect_platform.contracts import (
    CertifiedRuntime,
    DatasetVersion,
    ExperimentConfig,
    ObjectSpec,
    RunRecord,
    VertexJobConfig,
)
from defect_platform.control.controller import RunController
from defect_platform.skills import (
    FilesystemProjectReader,
    SkillDefinition,
    check_local_readiness,
    generate_project_ref,
    skill_definitions,
    welcome_text,
)

PLACEHOLDER = "CHANGE_ME"
RUN_REQUEST_FILES = {"experiment": "experiment.yaml", "job": "vertex.yaml",
                     "dataset": "dataset-version.yaml", "runtime": "runtime.yaml"}


class ToolError(Exception):
    """A tool refused or could not complete; the message is shown to the agent."""


class RunBackend(Protocol):
    def submit(self, *, experiment: ExperimentConfig, job: VertexJobConfig,
               dataset: DatasetVersion, runtime: CertifiedRuntime, classes: list[str],
               idempotency_key: str) -> tuple[RunRecord, bool]: ...
    def get(self, run_id: str) -> RunRecord: ...
    def list(self, object_slug: str | None = None) -> list[RunRecord]: ...
    def payload(self, run_id: str) -> dict | None: ...


class ControlServiceBackend:
    """Submit through the durable control API, which runs ``RunController`` server-side."""

    def __init__(self, client) -> None:
        self.client = client

    def submit(self, *, experiment, job, dataset, runtime, classes, idempotency_key):
        return self.client.submit({
            "experiment": experiment.model_dump(mode="json"), "job": job.model_dump(mode="json"),
            "dataset": dataset.model_dump(mode="json"), "runtime": runtime.model_dump(mode="json"),
            "classes": classes, "idempotency_key": idempotency_key})

    def get(self, run_id: str) -> RunRecord:
        return self.client.get(run_id)

    def list(self, object_slug: str | None = None) -> list[RunRecord]:
        return self.client.list(object_slug)

    def payload(self, run_id: str) -> dict | None:
        return None


class ControllerBackend:
    """Submit to an in-process ``RunController`` with the same overrides the API uses."""

    def __init__(self, controller: RunController) -> None:
        self.controller = controller

    def submit(self, *, experiment, job, dataset, runtime, classes, idempotency_key):
        return self.controller.submit(experiment=experiment, job=job,
                                      idempotency_key=idempotency_key,
                                      runtime_override=runtime, dataset_override=dataset,
                                      classes=classes)

    def get(self, run_id: str) -> RunRecord:
        return self.controller.get(run_id)

    def list(self, object_slug: str | None = None) -> list[RunRecord]:
        return self.controller.list(object_slug)

    def payload(self, run_id: str) -> dict | None:
        return self.controller.store.get_payload(run_id)


class ReadOnlyStoreBackend:
    """Read runs from the configured store; launching needs the control service."""

    def __init__(self, store_factory: Callable[[], Any]) -> None:
        self._factory, self._store = store_factory, None

    @property
    def store(self):
        if self._store is None:
            self._store = self._factory()
        return self._store

    def submit(self, **_kwargs):
        raise ToolError("Set DEFECT_CONTROL_SERVICE_URL to launch runs through the durable control service")

    def get(self, run_id: str) -> RunRecord:
        record = self.store.get(run_id)
        if record is None:
            raise KeyError(f"run not found: {run_id}")
        return record

    def list(self, object_slug: str | None = None) -> list[RunRecord]:
        return self.store.list(object_slug)

    def payload(self, run_id: str) -> dict | None:
        return self.store.get_payload(run_id)


class MetricsReader(Protocol):
    def metrics(self, run: RunRecord) -> dict[str, float]: ...


class MLflowMetricsReader:
    """Read final metrics that ``MLflowTracker.finish`` logged for a run."""

    def __init__(self, tracking_uri: str) -> None:
        from defect_platform.control.controller import MLflowTracker
        self.tracker = MLflowTracker(tracking_uri)

    def metrics(self, run: RunRecord) -> dict[str, float]:
        if not run.mlflow_run_id:
            return {}
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("MLflow metrics require optional cloud dependencies") from exc
        client = mlflow.MlflowClient(tracking_uri=self.tracker.tracking_uri)
        with self.tracker._auth():
            return dict(client.get_run(run.mlflow_run_id).data.metrics)


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(ToolArgs):
    pass


class ProjectArgs(ToolArgs):
    project: str = Field(description="Project folder name (the object slug) under the projects root.")


class CreateProjectArgs(ToolArgs):
    display_name: str = Field(min_length=1, description="Human name of the inspected object.")
    classes: list[str] = Field(min_length=2, description="Ordered defect class names.")
    description: str = Field(default="", description="What the object looks like.")
    slug: str | None = Field(default=None, description="Folder name; derived from display_name if omitted.")


class PlanRunArgs(ProjectArgs):
    idempotency_key: str | None = Field(
        default=None, max_length=200,
        description="Stable key for this request; defaults to object:experiment:dataset:runtime.")


class LaunchRunArgs(PlanRunArgs):
    plan_id: str = Field(pattern=r"^[a-f0-9]{64}$", description="plan_id returned by plan_run.")


class TrackRunsArgs(ToolArgs):
    run_id: str | None = Field(default=None, description="One run to inspect.")
    project: str | None = Field(default=None, description="Limit the list to one object slug.")


class CompareArgs(ToolArgs):
    run_ids: list[str] = Field(min_length=2, max_length=10)
    best_by: str | None = Field(
        default=None,
        description="Validation metric used to rank runs, e.g. validation_mcc. Held-out "
                    "test metrics cannot be used for selection.")
    goal: Literal["max", "min"] = "max"


class AgentTool:
    def __init__(self, name: str, skill: str, description: str, args: type[ToolArgs],
                 handler: Callable[..., dict[str, Any]], *, mutation: bool = False,
                 idempotent: bool = True, open_world: bool = False) -> None:
        self.name, self.skill, self.description, self.args = name, skill, description, args
        self.handler, self.mutation = handler, mutation
        self.idempotent, self.open_world = idempotent, open_world

    def input_schema(self) -> dict[str, Any]:
        schema = self.args.model_json_schema()
        schema.pop("title", None)
        schema.setdefault("properties", {})
        return schema

    def call(self, arguments: dict[str, Any] | None) -> dict[str, Any]:
        try:
            parsed = self.args.model_validate(arguments or {})
        except ValidationError as exc:
            raise ToolError(f"invalid arguments for {self.name}: {exc}") from exc
        return self.handler(parsed)


class AgentToolbox:
    def __init__(self, *, projects_root: str | Path = "projects", backend: RunBackend,
                 metrics: MetricsReader | None = None, readiness_root: str | Path = ".") -> None:
        self.projects_root, self.backend = Path(projects_root), backend
        self.metrics, self.readiness_root = metrics, Path(readiness_root)
        skills = {skill.name: skill for skill in skill_definitions()}
        self._tools: dict[str, AgentTool] = {}
        for tool in (
            AgentTool("welcome", "Welcome", "Explain the platform and list the skill catalog.",
                      NoArgs, lambda _a: {"text": welcome_text()}),
            AgentTool("check_readiness", "Check Readiness",
                      "Check local prerequisites (Python, packages, GPU, control/MLflow endpoints).",
                      NoArgs, lambda _a: check_local_readiness(root=self.readiness_root).as_dict()),
            AgentTool("list_projects", "List Projects",
                      "List projects with derived lifecycle state, blocker, and next action.",
                      NoArgs, self._list_projects),
            AgentTool("view_project", "View Project",
                      "Show one project's state, configuration files, and runs.",
                      ProjectArgs, self._view_project),
            AgentTool("create_project", "Create Project",
                      "Create a project folder from the object templates. Class definitions start "
                      "as a draft catalog; a person must review them before paid training.",
                      CreateProjectArgs, self._create_project, mutation=True, idempotent=False),
            AgentTool("plan_run", "Run Experiment",
                      "Read a project's run request and report cost, cap, runtime digest, and "
                      "blockers without submitting. Returns the plan_id launch_run needs.",
                      PlanRunArgs, self._plan_run),
            AgentTool("launch_run", "Run Experiment",
                      "Submit a planned run through the control plane, which enforces the spend "
                      "cap, certified image digest, and reviewed class catalog. Repeating the same "
                      "idempotency key returns the existing run.",
                      LaunchRunArgs, self._launch_run, mutation=True, open_world=True),
            AgentTool("track_runs", "Track Runs",
                      "Show one run by run_id, or list runs, optionally for one project.",
                      TrackRunsArgs, self._track_runs),
            AgentTool("compare_experiments", "Compare Experiments",
                      "Compare runs' parameters, lineage, and logged metrics; optionally rank by a metric.",
                      CompareArgs, self._compare),
        ):
            definition: SkillDefinition = skills[tool.skill]
            if tool.mutation and not definition.mutation:
                raise ValueError(f"tool {tool.name} mutates but skill {tool.skill} is read-only")
            self._tools[tool.name] = tool

    def tools(self) -> list[AgentTool]:
        return list(self._tools.values())

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        try:
            return tool.call(arguments)
        except ToolError:
            raise
        except (ValueError, KeyError, RuntimeError, OSError) as exc:
            raise ToolError(str(exc).strip("'\"") or exc.__class__.__name__) from exc

    # Project skills -----------------------------------------------------------

    def _reader(self) -> FilesystemProjectReader:
        return FilesystemProjectReader(self.projects_root, run_store=self.backend)

    def _project_dir(self, project: str) -> Path:
        path = self.projects_root / project
        if path.name != project or not path.is_dir() or project.startswith("."):
            raise ToolError(f"project not found: {project}")
        return path

    def _list_projects(self, _args: NoArgs) -> dict[str, Any]:
        return {"projects": [item.as_dict() for item in self._reader().list_projects()]}

    def _view_project(self, args: ProjectArgs) -> dict[str, Any]:
        path = self._project_dir(args.project)
        summary = self._reader()._read(path)
        files = {name: (path / name).exists() for name in
                 ("object.yaml", "class-catalog.yaml", "dataset.yaml", *RUN_REQUEST_FILES.values())}
        runs = self.backend.list(args.project)
        return {**summary.as_dict(), "files": files,
                "runs": [run.model_dump(mode="json") for run in runs[:20]]}

    def _create_project(self, args: CreateProjectArgs) -> dict[str, Any]:
        slug = args.slug or "-".join(args.display_name.lower().split())
        spec = ObjectSpec(slug=slug, display_name=args.display_name,
                          description=args.description, classes=args.classes)
        root = self.projects_root / spec.slug
        if root.exists() and any(root.iterdir()):
            raise ToolError(f"project {spec.slug} already exists; edit its files instead")
        from defect_platform.cli import _render_object_templates
        _render_object_templates(root, spec)
        ref = generate_project_ref(seed=spec.slug)
        (root / "project.yaml").write_text(yaml.safe_dump({
            **ref.as_dict(), "description": spec.description,
            "architecture": "DINOv3 + MLP head"}, sort_keys=False))
        return {"project": spec.slug, "project_ref": ref.as_dict(), "path": str(root),
                "class_catalog": "draft",
                "next_action": "Write class definitions in object.yaml and have a person approve "
                               "them with `defect object review-catalog`, then build the dataset."}

    # Run skills ---------------------------------------------------------------

    def _load_request(self, path: Path) -> tuple[dict[str, Any] | None, list[str]]:
        blockers: list[str] = []
        raw: dict[str, Any] = {}
        for key, name in (("object", "object.yaml"), *RUN_REQUEST_FILES.items()):
            file = path / name
            if not file.exists():
                blockers.append(f"{name} is missing")
                continue
            text = file.read_text()
            if PLACEHOLDER in text:
                blockers.append(f"{name} still contains {PLACEHOLDER} placeholders")
            value = yaml.safe_load(text)
            if not isinstance(value, dict):
                blockers.append(f"{name} must contain a YAML mapping")
                continue
            raw[key] = value
        if blockers:
            return None, blockers
        try:
            request = {
                "experiment": ExperimentConfig.model_validate(raw["experiment"]),
                "job": VertexJobConfig.model_validate(raw["job"]),
                "dataset": DatasetVersion.model_validate(raw["dataset"]),
                "runtime": CertifiedRuntime.model_validate(raw["runtime"]),
                "classes": ObjectSpec.model_validate(raw["object"]).classes,
            }
        except ValidationError as exc:
            return None, [f"invalid run request: {exc}"]
        return request, []

    def _plan(self, args: PlanRunArgs) -> tuple[dict[str, Any], dict[str, Any] | None]:
        path = self._project_dir(args.project)
        request, blockers = self._load_request(path)
        if request is None:
            return {"project": args.project, "ready": False, "blockers": blockers,
                    "plan_id": None}, None
        experiment: ExperimentConfig = request["experiment"]
        job: VertexJobConfig = request["job"]
        runtime: CertifiedRuntime = request["runtime"]
        key = args.idempotency_key or (f"{experiment.object_slug}:{experiment.experiment_id}:"
                                       f"{experiment.dataset_version_id}:{experiment.runtime_id}")
        estimated = RunController.estimate_cost(job)
        # These mirror checks RunController.submit enforces, so an agent sees them before
        # launching. The controller remains the authority and re-checks everything.
        if experiment.object_slug != args.project:
            blockers.append(f"experiment object {experiment.object_slug!r} differs from project folder")
        if estimated > job.max_run_cost_usd:
            blockers.append(f"estimated cost ${estimated:.2f} exceeds run cap ${job.max_run_cost_usd:.2f}")
        if not runtime.certified or "@sha256:" not in runtime.image_digest:
            blockers.append(f"runtime {runtime.runtime_id!r} is not certified with an immutable digest")
        if runtime.runtime_id != experiment.runtime_id:
            blockers.append("runtime.yaml does not match the experiment runtime_id")
        if request["dataset"].version_id != experiment.dataset_version_id:
            blockers.append("dataset-version.yaml does not match the experiment dataset_version_id")
        canonical = {name: (value.model_dump(mode="json") if isinstance(value, BaseModel) else value)
                     for name, value in request.items()}
        plan_id = hashlib.sha256(json.dumps({**canonical, "idempotency_key": key},
                                            sort_keys=True).encode()).hexdigest()
        plan = {"project": args.project, "ready": not blockers, "blockers": blockers,
                "plan_id": plan_id, "idempotency_key": key,
                "experiment_id": experiment.experiment_id,
                "dataset_version_id": experiment.dataset_version_id,
                "runtime_id": runtime.runtime_id, "runtime_image_digest": runtime.image_digest,
                "runtime_certified": runtime.certified, "classes": request["classes"],
                "machine": {"machine_type": job.machine_type,
                            "accelerator_type": job.accelerator_type,
                            "accelerator_count": job.accelerator_count},
                "max_run_hours": job.max_run_hours, "estimated_cost_usd": estimated,
                "max_run_cost_usd": job.max_run_cost_usd,
                "parameters": _parameters(canonical["experiment"])}
        return plan, {**request, "idempotency_key": key}

    def _plan_run(self, args: PlanRunArgs) -> dict[str, Any]:
        return self._plan(args)[0]

    def _launch_run(self, args: LaunchRunArgs) -> dict[str, Any]:
        plan, request = self._plan(args)
        if plan["plan_id"] != args.plan_id:
            raise ToolError("project files or idempotency key changed since plan_run; plan again "
                            "and review the new plan before launching")
        if request is None or plan["blockers"]:
            raise ToolError("run is blocked: " + "; ".join(plan["blockers"]))
        run, created = self.backend.submit(**request)
        return {"created": created, "run": run.model_dump(mode="json"),
                "estimated_cost_usd": plan["estimated_cost_usd"],
                "max_run_cost_usd": plan["max_run_cost_usd"]}

    def _track_runs(self, args: TrackRunsArgs) -> dict[str, Any]:
        if args.run_id:
            return {"run": self.backend.get(args.run_id).model_dump(mode="json")}
        return {"runs": [run.model_dump(mode="json") for run in self.backend.list(args.project)]}

    def _compare(self, args: CompareArgs) -> dict[str, Any]:
        rows = []
        for run_id in dict.fromkeys(args.run_ids):
            run = self.backend.get(run_id)
            payload = self.backend.payload(run_id) or {}
            params = _parameters(payload["experiment"]) if "experiment" in payload else None
            metrics: dict[str, float] | None = None
            if self.metrics is not None and run.mlflow_run_id:
                metrics = self.metrics.metrics(run)
            rows.append({"run_id": run.run_id, "experiment_id": run.experiment_id,
                         "state": run.state.value, "dataset_version_id": run.dataset_version_id,
                         "runtime_id": run.runtime_id,
                         "runtime_image_digest": run.runtime_image_digest,
                         "estimated_cost_usd": payload.get("estimated_cost_usd"),
                         "parameters": params, "metrics": metrics})
        differences = {}
        known = [row["parameters"] for row in rows if row["parameters"] is not None]
        for name in sorted({key for params in known for key in params}):
            values = [params.get(name) for params in known]
            if any(value != values[0] for value in values):
                differences[name] = {row["run_id"]: row["parameters"].get(name)
                                     for row in rows if row["parameters"] is not None}
        for field in ("dataset_version_id", "runtime_image_digest"):
            if len({row[field] for row in rows}) > 1:
                differences[field] = {row["run_id"]: row[field] for row in rows}
        result: dict[str, Any] = {"runs": rows, "differences": differences}
        if args.best_by:
            # Selecting on the held-out split would make every reported number
            # optimistic, so ranking is restricted to validation metrics.
            if not args.best_by.startswith("validation"):
                raise ToolError(
                    f"{args.best_by!r} is not a validation metric; ranking runs on held-out "
                    "test results is not allowed. Use a validation_* metric.")
            scored = [row for row in rows if row["metrics"] and args.best_by in row["metrics"]]
            scored.sort(key=lambda row: row["metrics"][args.best_by], reverse=args.goal == "max")
            result["ranking"] = {"metric": args.best_by, "goal": args.goal,
                                 "run_ids": [row["run_id"] for row in scored],
                                 "unscored": [row["run_id"] for row in rows if row not in scored]}
        return result


def _parameters(experiment: dict[str, Any]) -> dict[str, Any]:
    params = {key: value for key, value in experiment.items()
              if key not in {"experiment_id", "object_slug", "dataset_version_id", "runtime_id",
                             "catalog_sha256", "model"}}
    model = experiment.get("model") or {}
    params.update({f"model.{key}": value for key, value in model.items()
                   if key not in {"weights_uri", "weights_sha256"}})
    return params


def toolbox_from_env() -> AgentToolbox:
    """Build tools from the same environment variables the CLI reads."""
    service_url = os.getenv("DEFECT_CONTROL_SERVICE_URL")
    backend: RunBackend
    if service_url:
        from defect_platform.control.client import ControlAPIClient
        backend = ControlServiceBackend(ControlAPIClient(service_url))
    else:
        from defect_platform.control.store import run_store_from_env
        backend = ReadOnlyStoreBackend(run_store_from_env)
    tracking_uri = os.getenv("DEFECT_MLFLOW_TRACKING_URI")
    return AgentToolbox(projects_root=os.getenv("DEFECT_PROJECTS_ROOT", "projects"),
                        backend=backend,
                        metrics=MLflowMetricsReader(tracking_uri) if tracking_uri else None)
