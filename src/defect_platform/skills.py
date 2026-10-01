"""Conversational skill catalog and read-only project/readiness projections.

Skills are adapters over the existing project files and control-plane stores.  This
module deliberately does not create a second source of truth or submit work.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import yaml

from .contracts import RunRecord, RunState
from .control.store import RunStore


class SkillNamespace(StrEnum):
    ONBOARDING = "onboarding"
    USER = "user"
    DEVELOPER = "developer"
    INTERNAL = "internal"


class SkillDefinition:
    def __init__(self, name: str, namespace: SkillNamespace, purpose: str,
                 *, mutation: bool = False) -> None:
        if not 1 <= len(name.split()) <= 3:
            raise ValueError("skill names must contain one to three words")
        self.name = name
        self.namespace = namespace
        self.purpose = purpose
        self.mutation = mutation

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "namespace": self.namespace.value,
                "purpose": self.purpose, "mutation": self.mutation}


SKILLS: tuple[SkillDefinition, ...] = (
    SkillDefinition("Welcome", SkillNamespace.ONBOARDING, "Explain the platform and available skills."),
    SkillDefinition("Check Readiness", SkillNamespace.ONBOARDING, "Check local operational prerequisites."),
    SkillDefinition("Choose Role", SkillNamespace.ONBOARDING, "Discover role and available capabilities."),
    SkillDefinition("List Projects", SkillNamespace.USER, "List project state and next actions."),
    SkillDefinition("View Project", SkillNamespace.USER, "Inspect project lineage and evidence."),
    SkillDefinition("Create Project", SkillNamespace.USER, "Create a bounded project facade.", mutation=True),
    SkillDefinition("Configure Data", SkillNamespace.USER, "Prepare and publish an immutable dataset.", mutation=True),
    SkillDefinition("Configure Experiment", SkillNamespace.USER, "Create a parameterized experiment.", mutation=True),
    SkillDefinition("Run Experiment", SkillNamespace.USER, "Submit a validated experiment.", mutation=True),
    SkillDefinition("Track Runs", SkillNamespace.USER, "Inspect training run progress."),
    SkillDefinition("Compare Experiments", SkillNamespace.USER, "Compare experiment parameters and results."),
    SkillDefinition("Review Results", SkillNamespace.USER, "Summarize evaluation evidence."),
    SkillDefinition("Prepare Release", SkillNamespace.USER, "Stage an exact model release.", mutation=True),
    SkillDefinition("Deploy Model", SkillNamespace.USER, "Promote or roll back an approved release.", mutation=True),
    SkillDefinition("Change Workflow", SkillNamespace.DEVELOPER, "Change lifecycle orchestration.", mutation=True),
    SkillDefinition("Fix Bug", SkillNamespace.DEVELOPER, "Reproduce and repair a defect.", mutation=True),
    SkillDefinition("Add Feature", SkillNamespace.DEVELOPER, "Implement a bounded capability.", mutation=True),
    SkillDefinition("Remove Feature", SkillNamespace.DEVELOPER, "Remove a capability safely.", mutation=True),
    SkillDefinition("Update Contract", SkillNamespace.DEVELOPER, "Change shared interfaces.", mutation=True),
    SkillDefinition("Update Trainer", SkillNamespace.DEVELOPER, "Change trainer-owned behavior.", mutation=True),
    SkillDefinition("Update Runtime", SkillNamespace.DEVELOPER, "Change runtime packaging.", mutation=True),
    SkillDefinition("Certify Runtime", SkillNamespace.DEVELOPER, "Certify an existing runtime digest.", mutation=True),
    SkillDefinition("Validate Change", SkillNamespace.DEVELOPER, "Run scoped validation."),
    SkillDefinition("Document Change", SkillNamespace.DEVELOPER, "Update source-grounded documentation.", mutation=True),
    SkillDefinition("Read Authority", SkillNamespace.INTERNAL, "Read authoritative records."),
    SkillDefinition("Derive State", SkillNamespace.INTERNAL, "Derive canonical project state."),
    SkillDefinition("Check Boundary", SkillNamespace.INTERNAL, "Enforce role and ownership boundaries."),
    SkillDefinition("Plan Handoff", SkillNamespace.INTERNAL, "Create an immutable layer handoff."),
    SkillDefinition("Gate Handoff", SkillNamespace.INTERNAL, "Validate and record a handoff."),
    SkillDefinition("Record Evidence", SkillNamespace.INTERNAL, "Preserve operational evidence."),
    SkillDefinition("Reconcile State", SkillNamespace.INTERNAL, "Reconcile uncertain observations."),
    SkillDefinition("Recover Failure", SkillNamespace.INTERNAL, "Classify and bound recovery."),
    SkillDefinition("Check Integrity", SkillNamespace.INTERNAL, "Verify immutable lineage."),
    SkillDefinition("Render View", SkillNamespace.INTERNAL, "Render typed records for a user."),
)


def skill_definitions(namespace: SkillNamespace | None = None) -> tuple[SkillDefinition, ...]:
    return tuple(skill for skill in SKILLS if namespace is None or skill.namespace is namespace)


def welcome_text() -> str:
    """Return the onboarding welcome screen used by a conversational host."""
    lines = [
        "Welcome to the DINOv3 Training Platform.",
        "Create bounded projects, publish immutable datasets, run parameterized experiments,",
        "review evidence, and promote approved models to serving.",
        "",
        "Available skills:",
    ]
    for namespace in SkillNamespace:
        names = ", ".join(skill.name for skill in skill_definitions(namespace))
        lines.append(f"- {namespace.value.title()}: {names}")
    lines.extend(("", "Say 'check readiness' to inspect this computer before operating."))
    return "\n".join(lines)


def render_project_table(projects: list[ProjectSummary]) -> str:
    """Render a compact, stable table suitable for a chat response."""
    headers = ("Project", "Object", "Architecture", "State", "Description", "Next action")
    rows = [headers]
    for project in projects:
        rows.append((project.ref.display_name, project.object_name, project.architecture,
                     project.state.value, project.description or "—", project.next_action or "—"))
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(headers))]
    return "\n".join(" | ".join(str(value).ljust(widths[index])
                                  for index, value in enumerate(row)) for row in rows)


class ProjectState(StrEnum):
    DRAFT = "draft"
    DATA_INGESTION = "data_ingestion"
    DATASET_READY = "dataset_ready"
    EXPERIMENT_READY = "experiment_ready"
    TRAINING = "training"
    EVALUATION = "evaluation"
    RELEASE_READY = "release_ready"
    DEPLOYED = "deployed"
    FAILED = "failed"
    CANCELED = "canceled"
    ARCHIVED = "archived"


class ProjectRef:
    def __init__(self, project_id: str, display_name: str, created_at: datetime) -> None:
        self.project_id, self.display_name = project_id, display_name
        self.created_at = created_at.astimezone(UTC)

    def as_dict(self) -> dict[str, object]:
        return {"project_id": self.project_id, "display_name": self.display_name,
                "created_at": self.created_at.isoformat()}


def generate_project_ref(*, now: datetime | None = None, seed: str = "") -> ProjectRef:
    """Generate a stable, readable reference; ``seed`` makes tests deterministic."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    adjectives = ("Amber", "Quiet", "Bright", "Cobalt", "Swift", "Mellow", "Clear", "Silver")
    nouns = ("Falcon", "Maple", "Orbit", "Harbor", "Comet", "Pine", "River", "Lumen")
    digest = hashlib.sha256(f"{seed}|{now.isoformat()}".encode()).digest()
    name = f"{adjectives[digest[0] % len(adjectives)]}-{nouns[digest[1] % len(nouns)]}"
    stamp = now.strftime("%Y%m%d-%H%M%SZ")
    return ProjectRef(project_id=f"{name.lower()}-{stamp}-{digest.hex()[:8]}",
                      display_name=f"{name} {stamp}", created_at=now)


class ProjectSummary:
    def __init__(self, ref: ProjectRef, object_name: str, description: str,
                 architecture: str, state: ProjectState, *, blocker: str | None = None,
                 latest_run: RunRecord | None = None, next_action: str | None = None) -> None:
        self.ref, self.object_name, self.description = ref, object_name, description
        self.architecture, self.state, self.blocker = architecture, state, blocker
        self.latest_run, self.next_action = latest_run, next_action

    def as_dict(self) -> dict[str, object]:
        return {"project": self.ref.as_dict(), "object": self.object_name,
                "description": self.description, "architecture": self.architecture,
                "state": self.state.value, "blocker": self.blocker,
                "latest_run": self.latest_run.model_dump(mode="json") if self.latest_run else None,
                "next_action": self.next_action}


class ReadinessCheck:
    def __init__(self, check_id: str, status: str, message: str, evidence: tuple[str, ...] = ()) -> None:
        if status not in {"ready", "blocked", "unconfigured"}:
            raise ValueError("invalid readiness status")
        self.check_id, self.status, self.message, self.evidence = check_id, status, message, evidence

    def as_dict(self) -> dict[str, object]:
        return {"check_id": self.check_id, "status": self.status,
                "message": self.message, "evidence": list(self.evidence)}


class ReadinessReport:
    def __init__(self, checks: list[ReadinessCheck]) -> None:
        self.checks = checks

    @property
    def status(self) -> str:
        if any(check.status == "blocked" for check in self.checks):
            return "blocked"
        return "ready"

    def as_dict(self) -> dict[str, object]:
        return {"status": self.status, "checks": [check.as_dict() for check in self.checks]}


class ProjectReader(Protocol):
    def list_projects(self) -> list[ProjectSummary]: ...


class FilesystemProjectReader:
    """Read project folders and derive a user-facing state without mutating them."""

    def __init__(self, root: str | Path = "projects", run_store: RunStore | None = None) -> None:
        self.root, self.run_store = Path(root), run_store

    def list_projects(self) -> list[ProjectSummary]:
        if not self.root.exists():
            return []
        return [self._read(path) for path in sorted(self.root.iterdir()) if path.is_dir()]

    def _yaml(self, path: Path) -> dict:
        if not path.exists():
            return {}
        value = yaml.safe_load(path.read_text()) or {}
        return value if isinstance(value, dict) else {}

    def _read(self, path: Path) -> ProjectSummary:
        obj_data = self._yaml(path / "object.yaml")
        project_data = self._yaml(path / "project.yaml")
        object_name = str(obj_data.get("display_name", path.name))
        description = str(project_data.get("description", obj_data.get("description", "")))
        architecture = str(project_data.get("architecture", "DINOv3 + MLP head"))
        created = project_data.get("created_at")
        created_at = datetime.fromisoformat(created) if isinstance(created, str) else datetime.fromtimestamp(path.stat().st_mtime, UTC)
        ref = ProjectRef(str(project_data.get("project_id", path.name)),
                         str(project_data.get("display_name", path.name)), created_at)
        runs = self.run_store.list(path.name) if self.run_store else []
        latest = runs[0] if runs else None
        state, blocker, action = self._state(path, latest)
        return ProjectSummary(ref, object_name, description, architecture, state,
                              blocker=blocker, latest_run=latest, next_action=action)

    @staticmethod
    def _state(path: Path, latest: RunRecord | None) -> tuple[ProjectState, str | None, str | None]:
        if latest and latest.state is RunState.FAILED:
            return ProjectState.FAILED, latest.failure_message or latest.failure_code, "Review the failed run"
        if latest and latest.state is RunState.CANCELED:
            return ProjectState.CANCELED, "The latest run was canceled", "Create or submit a new experiment"
        if latest and latest.state in {RunState.PENDING, RunState.PREPARING, RunState.SUBMITTED, RunState.RUNNING}:
            return ProjectState.TRAINING, None, "Wait for the run to finish"
        if latest and latest.state is RunState.SUCCEEDED:
            return ProjectState.EVALUATION, None, "Review evaluation results"
        if not (path / "object.yaml").exists():
            return ProjectState.DRAFT, "Object definition is missing", "Create object.yaml"
        if not (path / "dataset-version.yaml").exists():
            return ProjectState.DATA_INGESTION, "No immutable dataset version", "Review and build the dataset"
        if not (path / "experiment.yaml").exists():
            return ProjectState.DATASET_READY, "No experiment configuration", "Configure an experiment"
        if not (path / "runtime.yaml").exists():
            return ProjectState.EXPERIMENT_READY, "No certified runtime binding", "Select a certified runtime"
        return ProjectState.EXPERIMENT_READY, None, "Submit an experiment"


def check_local_readiness(*, root: str | Path = ".") -> ReadinessReport:
    """Perform non-mutating checks for the onboarding readiness skill."""
    root = Path(root)
    checks: list[ReadinessCheck] = [
        ReadinessCheck("python", "ready", platform.python_version(), (platform.python_implementation(),)),
        ReadinessCheck("uv", "ready", shutil.which("uv") or "uv is unavailable",
                       () if shutil.which("uv") else ("Install uv",)),
        ReadinessCheck("project_root", "ready" if root.exists() else "blocked",
                       str(root.resolve())),
    ]
    for module in ("pydantic", "yaml", "typer"):
        present = importlib.util.find_spec(module) is not None
        checks.append(ReadinessCheck(f"import:{module}", "ready" if present else "blocked",
                                     "importable" if present else "not importable"))
    gpu = shutil.which("nvidia-smi")
    if not gpu:
        checks.append(ReadinessCheck("gpu", "unconfigured", "No NVIDIA GPU probe is configured"))
    else:
        try:
            result = subprocess.run([gpu, "--query-gpu=name", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=5, check=False)
            checks.append(ReadinessCheck("gpu", "ready" if result.returncode == 0 else "blocked",
                                        result.stdout.strip() or result.stderr.strip()))
        except (OSError, subprocess.SubprocessError) as exc:
            checks.append(ReadinessCheck("gpu", "blocked", str(exc)))
    for env_name in ("DEFECT_CONTROL_SERVICE_URL", "DEFECT_MLFLOW_TRACKING_URI"):
        checks.append(ReadinessCheck(env_name, "ready" if os.getenv(env_name) else "unconfigured",
                                     os.getenv(env_name, "not configured")))
    return ReadinessReport(checks)
