"""Guided command line for object setup and durable training runs."""
# Typer uses parameter defaults to define the command-line schema.
# ruff: noqa: B008

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import typer
import yaml

from defect_platform.catalog_store import DirectoryCatalogStore, catalog_store_from_env
from defect_platform.contracts import (
    CertifiedRuntime,
    DatasetSpec,
    DatasetVersion,
    ExperimentConfig,
    LabelReviewEvidence,
    ModelSpec,
    ObjectSpec,
    VertexJobConfig,
)
from defect_platform.control.client import ControlAPIClient
from defect_platform.control.goal_commands import goal_app
from defect_platform.control.setup import propose_setup
from defect_platform.control.store import run_store_from_env
from defect_platform.control.vm_queue_commands import vm_queue_app
from defect_platform.dataset import build_dataset, preview_dataset
from defect_platform.runtime_release import runtime_app
from defect_platform.semantics import ClassCatalog, catalog_for_object
from defect_platform.serve.commands import release_app
from defect_platform.telemetry import bind_context, configure_logging, reset_context

app = typer.Typer(help="Create defect datasets and manage DINOv3 training runs.", no_args_is_help=True)
object_app = typer.Typer(help="Create and find object projects.", no_args_is_help=True)
run_app = typer.Typer(help="Submit and inspect training runs.", no_args_is_help=True)
train_app = typer.Typer(help="Training commands.", no_args_is_help=True)
dataset_app = typer.Typer(help="Preview, review, and build immutable datasets.", no_args_is_help=True)
setup_app = typer.Typer(help="Prepare an object and training configuration.", no_args_is_help=True)
app.add_typer(object_app, name="object")
app.add_typer(run_app, name="run")
app.add_typer(train_app, name="train")
app.add_typer(dataset_app, name="dataset")
app.add_typer(setup_app, name="setup")
app.add_typer(release_app, name="release")
app.add_typer(runtime_app, name="runtime")
app.add_typer(goal_app, name="goal")
train_app.add_typer(vm_queue_app, name="queue")


def _read_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise typer.BadParameter(f"cannot read config file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise typer.BadParameter("configuration file must contain a YAML mapping")
    return value


def _run_submit(config_path: Path, idempotency_key: str | None) -> None:
    project_dir: Path | None = None
    if config_path.is_dir():
        project_dir = config_path
        object_path = project_dir / "object.yaml"
        config = {
            "experiment": _read_yaml(project_dir / "experiment.yaml"),
            "job": _read_yaml(project_dir / "vertex.yaml"),
            "dataset": _read_yaml(project_dir / "dataset-version.yaml"),
            "runtime": _read_yaml(project_dir / "runtime.yaml"),
            "classes": ObjectSpec.model_validate(_read_yaml(object_path)).classes,
        }
        if (project_dir / "platform.yaml").exists():
            config["platform"] = _read_yaml(project_dir / "platform.yaml")
    else:
        config = _read_yaml(config_path)
    try:
        experiment = ExperimentConfig.model_validate(config["experiment"])
        job = VertexJobConfig.model_validate(config["job"])
        dataset = DatasetVersion.model_validate(config["dataset"])
        runtime = CertifiedRuntime.model_validate(config["runtime"])
        if config.get("classes"):
            classes = list(config["classes"])
        else:
            object_path = (project_dir / "object.yaml") if project_dir else Path("projects") / experiment.object_slug / "object.yaml"
            classes = ObjectSpec.model_validate(_read_yaml(object_path)).classes
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(f"invalid run request: {exc}") from exc
    key = idempotency_key or config.get("idempotency_key") or (
        f"{experiment.object_slug}:{experiment.experiment_id}:"
        f"{experiment.dataset_version_id}:{experiment.runtime_id}")
    service_url = config.get("platform", {}).get("control_service_url") or os.getenv("DEFECT_CONTROL_SERVICE_URL")
    if not service_url:
        raise typer.BadParameter("Set DEFECT_CONTROL_SERVICE_URL to submit through the durable control service")
    run, _created = ControlAPIClient(service_url).submit({
        "experiment": experiment.model_dump(mode="json"),
        "job": job.model_dump(mode="json"),
        "dataset": dataset.model_dump(mode="json"),
        "runtime": runtime.model_dump(mode="json"),
        "classes": classes,
        "idempotency_key": key,
    })
    token = bind_context(run_id=run.run_id, object_slug=run.object_slug,
                         dataset_version_id=run.dataset_version_id)
    try:
        import logging
        logging.getLogger(__name__).info("training run accepted", extra={"stage": "run_submit"})
    finally:
        reset_context(token)
    _write_run_locator(run)
    typer.echo(f"Run ID: {run.run_id}\nState: {run.state.value}\nDataset: {run.dataset_version_id}\n"
               f"Run files: {run.output_uri or 'pending'}\nMLflow: {run.mlflow_run_id or 'pending'}")


@run_app.command("submit")
def run_submit(config: Path = typer.Argument(..., exists=True, readable=True),
               idempotency_key: str | None = typer.Option(None, "--idempotency-key", "-k")):
    """Validate a run request, enforce its spend cap, and return a run ID."""
    _run_submit(config, idempotency_key)


@train_app.command("start")
def train_start(config: Path = typer.Argument(..., exists=True, readable=True),
                idempotency_key: str | None = typer.Option(None, "--idempotency-key", "-k")):
    """Alias for `run submit` for the simple training workflow."""
    _run_submit(config, idempotency_key)


@run_app.command("status")
def run_status(run_id: str = typer.Argument(...),
               as_json: bool = typer.Option(False, "--json")):
    """Show the state and useful locations for one run."""
    service_url = os.getenv("DEFECT_CONTROL_SERVICE_URL")
    run = ControlAPIClient(service_url).get(run_id) if service_url else run_store_from_env().get(run_id)
    if run is None:
        raise typer.BadParameter(f"Run {run_id} was not found")
    _write_run_locator(run)
    if as_json:
        typer.echo(json.dumps(run.model_dump(mode="json"), indent=2))
    else:
        _print_run(run)


@run_app.command("list")
def run_list(object_slug: str | None = typer.Option(None, "--object"),
             as_json: bool = typer.Option(False, "--json")):
    """List recent runs, optionally filtered by object."""
    service_url = os.getenv("DEFECT_CONTROL_SERVICE_URL")
    runs = ControlAPIClient(service_url).list(object_slug) if service_url else run_store_from_env().list(object_slug)
    if as_json:
        typer.echo(json.dumps([run.model_dump(mode="json") for run in runs], indent=2))
        return
    if not runs:
        typer.echo("No training runs found.")
        return
    typer.echo("RUN ID                                OBJECT       STATE      CREATED (UTC)        DATASET")
    for run in runs:
        typer.echo(f"{run.run_id:<38} {run.object_slug:<12} {run.state.value:<10} "
                   f"{run.created_at:%Y-%m-%d %H:%M}  {run.dataset_version_id}")


def _write_run_locator(run) -> None:
    project_dir = Path("projects") / run.object_slug
    target = project_dir / "runs" / f"{run.run_id}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (target.parent / "README.md").exists():
        (target.parent / "README.md").write_text(
            "# Training runs\n\nEach YAML file links a run to its current state and remote artifacts. "
            "Large datasets, checkpoints, and logs stay in GCS and MLflow.\n")
    payload = {"run_id": run.run_id, "object_slug": run.object_slug,
               "experiment_id": run.experiment_id, "dataset_version_id": run.dataset_version_id,
               "runtime_id": run.runtime_id, "state": run.state.value,
               "created_at": run.created_at.isoformat(), "vertex_job_name": run.vertex_job_name,
               "mlflow_run_id": run.mlflow_run_id, "output_uri": run.output_uri,
               "logs_uri": run.logs_uri, "failure_code": run.failure_code,
               "failure_message": run.failure_message, "catalog_sha256": run.catalog_sha256,
               "dataset_semantic_sha256": run.dataset_semantic_sha256,
               "runtime_image_digest": run.runtime_image_digest,
               "runtime_source_commit": run.runtime_source_commit}
    target.write_text(yaml.safe_dump(payload, sort_keys=False))


def _print_run(run) -> None:
    typer.echo(f"Run: {run.run_id}\nObject: {run.object_slug}\nState: {run.state.value}\n"
               f"Created: {run.created_at.isoformat()}\nDataset: {run.dataset_version_id}\n"
               f"Vertex job: {run.vertex_job_name or 'waiting for workflow'}\n"
               f"MLflow run: {run.mlflow_run_id or 'not configured'}\n"
               f"Output files: {run.output_uri or 'not available'}\n"
               f"Logs: {run.logs_uri or 'use the linked Vertex job logs'}")
    if run.failure_code:
        typer.echo(f"Failure: {run.failure_code}: {run.failure_message or ''}")


def _dataset_inputs(dataset_config: Path, object_config: Path):
    try:
        dataset = DatasetSpec.model_validate(_read_yaml(dataset_config))
        object_spec = ObjectSpec.model_validate(_read_yaml(object_config))
    except ValueError as exc:
        raise typer.BadParameter(f"invalid dataset/object configuration: {exc}") from exc
    return dataset, object_spec


@dataset_app.command("preview")
def dataset_preview(config: Path = typer.Argument(..., exists=True, readable=True),
                    object_config: Path = typer.Option(..., "--object", exists=True, readable=True)):
    """Read configured label sources and report mappings, counts, and review exceptions."""
    dataset, object_spec = _dataset_inputs(config, object_config)
    try:
        result = preview_dataset(dataset, object_spec)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result.to_dict(), indent=2))
    if not result.ready:
        typer.echo("Resolve the listed items before building. Suggestions are never applied automatically.", err=True)


@dataset_app.command("build")
def dataset_build(config: Path = typer.Argument(..., exists=True, readable=True),
                   object_config: Path = typer.Option(..., "--object", exists=True, readable=True),
                   output: Path | None = typer.Option(None, "--output", "-o"),
                   interactive_review: bool = typer.Option(True, "--review/--no-review")):
    """Review label aliases when needed, then build immutable WebDataset shards."""
    dataset, object_spec = _dataset_inputs(config, object_config)
    try:
        preview = preview_dataset(dataset, object_spec)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    mapping = dict(dataset.label_mapping)
    original_mapping = dict(mapping)
    exceptions_before_review = [
        {"source": item.source, "row_number": item.row_number,
         "image_uri": item.image_uri, "raw_label": item.raw_label,
         "suggested_class": item.suggested_class, "reason": item.reason}
        for item in preview.exceptions
    ]
    reviewable = [item for item in preview.exceptions if item.reason == "unmapped_label"]
    if reviewable and interactive_review:
        for raw_label in dict.fromkeys(item.raw_label for item in reviewable):
            item = next(item for item in reviewable if item.raw_label == raw_label)
            suggestion = item.suggested_class or ""
            typer.echo(f"Unmapped label {raw_label!r}; suggestions: {', '.join(item.candidates) or 'none'}")
            answer = typer.prompt("Canonical class (blank leaves unresolved)", default=suggestion)
            if answer:
                if answer not in object_spec.classes:
                    raise typer.BadParameter(f"{answer!r} is not one of: {', '.join(object_spec.classes)}")
                mapping[raw_label] = answer
        dataset = dataset.model_copy(update={"label_mapping": mapping})
        preview = preview_dataset(dataset, object_spec)
        if mapping != original_mapping:
            reviewer = typer.prompt("Name or ID of the person approving these label mappings")
            evidence = {"reviewed_at": datetime.now(UTC).isoformat(), "reviewer": reviewer,
                        "accepted_mapping_additions": {key: value for key, value in mapping.items()
                                                        if original_mapping.get(key) != value},
                        "preview_before_review": exceptions_before_review}
            dataset = dataset.model_copy(update={"label_review": LabelReviewEvidence.model_validate(evidence)})
            config.write_text(yaml.safe_dump(dataset.model_dump(mode="json"), sort_keys=False))
            review_file = config.parent / f"label-review-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
            review_file.write_text(json.dumps(evidence, indent=2))
            typer.echo(f"Accepted mapping recorded in {review_file}")
    if not preview.ready:
        typer.echo(json.dumps(preview.to_dict(), indent=2), err=True)
        raise typer.BadParameter("dataset still has unresolved review exceptions; edit its source or mapping and retry")
    try:
        version = build_dataset(dataset, object_spec)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    output_path = output or config.parent / "dataset-version.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(version.model_dump(mode="json"), sort_keys=False))
    typer.echo(f"Dataset {version.version_id} built with {sum(version.sample_counts.values())} samples.")
    typer.echo(f"Version details: {output_path}")


@object_app.command("init")
def object_init():
    """Create a plain-language object project template interactively."""
    display_name = typer.prompt("Object name")
    slug_default = display_name.lower().replace(" ", "-")
    slug = typer.prompt("Short folder name", default=slug_default)
    description = typer.prompt("What does the object look like?", default="")
    class_text = typer.prompt("Defect classes, separated by commas")
    classes = [value.strip() for value in class_text.split(",") if value.strip()]
    try:
        spec = ObjectSpec(slug=slug, display_name=display_name,
                          description=description, classes=classes)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    root = Path("projects") / spec.slug
    if root.exists() and any(root.iterdir()):
        raise typer.BadParameter(f"Project {root} already exists; edit its files instead of overwriting them")
    _render_object_templates(root, spec)
    typer.echo(f"Created the editable object, dataset, experiment, and Vertex settings in {root}/")


def _render_object_templates(root: Path, spec: ObjectSpec) -> None:
    templates = Path(__file__).resolve().parents[2] / "templates" / "object"
    if not templates.is_dir():
        raise RuntimeError(f"object templates are missing: {templates}")
    root.mkdir(parents=True, exist_ok=True)
    replacements = {"{{ slug }}": spec.slug, "{{slug}}": spec.slug,
                    "{{ display_name }}": spec.display_name, "{{display_name}}": spec.display_name,
                    "{{ description }}": spec.description, "{{description}}": spec.description,
                    "{{ first_class }}": spec.classes[0], "{{first_class}}": spec.classes[0],
                    "{{ second_class }}": spec.classes[1], "{{second_class}}": spec.classes[1]}
    for name in ("object.yaml", "dataset.yaml", "experiment.yaml", "vertex.yaml", "README.md"):
        content = (templates / name).read_text()
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
        if name == "object.yaml":
            catalog = catalog_for_object(spec)
            if catalog.review_status == "legacy":
                catalog = catalog.model_copy(update={"review_status": "draft"})
            spec = ObjectSpec.model_validate({**spec.model_dump(mode="json"),
                                              "class_catalog": catalog.model_dump(mode="json")})
            content = yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False)
            (root / "class-catalog.yaml").write_text(
                yaml.safe_dump(catalog.model_dump(mode="json"), sort_keys=False))
        content = content.replace("defect runs list --object", "defect run list --object")
        content = content.replace("defect runs show RUN_ID", "defect run status RUN_ID")
        (root / name).write_text(content)


def _publish_reviewed_object(spec: ObjectSpec, store: DirectoryCatalogStore,
                             reviewer: str, *, approved: bool) -> ObjectSpec:
    if not approved:
        raise ValueError("explicit class catalog approval is required")
    catalog = catalog_for_object(spec)
    if catalog.review_status != "reviewed":
        catalog = ClassCatalog.model_validate({**catalog.model_dump(mode="json"),
            "review_status": "reviewed", "reviewed_by": reviewer,
            "reviewed_at": datetime.now(UTC).isoformat()})
    store.publish(catalog)
    return ObjectSpec.model_validate({**spec.model_dump(mode="json"),
                                     "class_catalog": catalog.model_dump(mode="json")})


@object_app.command("review-catalog")
def review_catalog(
    object_config: Path = typer.Argument(..., exists=True, readable=True),
    reviewer: str = typer.Option(..., "--reviewer"),
    approve: bool = typer.Option(False, "--approve", help="Explicitly approve the displayed definitions"),
    store_root: str | None = typer.Option(None, "--store", help="Operator catalog directory or gs:// root"),
) -> None:
    """Review definitions in object.yaml and publish an immutable approved catalog."""
    store = DirectoryCatalogStore(store_root) if store_root else catalog_store_from_env()
    if store is None:
        raise typer.BadParameter("Set DEFECT_CLASS_CATALOG_ROOT or --store to the operator approval store")
    try:
        spec = ObjectSpec.model_validate(_read_yaml(object_config))
        for item in catalog_for_object(spec).classes:
            typer.echo(f"{item.class_id}: {item.label} — {item.definition or 'MISSING DEFINITION'}")
        spec = _publish_reviewed_object(spec, store, reviewer, approved=approve)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    object_config.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
    (object_config.parent / "class-catalog.yaml").write_text(
        yaml.safe_dump(spec.class_catalog.model_dump(mode="json"), sort_keys=False))
    typer.echo(f"Approved catalog: {spec.class_catalog.sha256}")


def _guided_catalog(spec: ObjectSpec, store: DirectoryCatalogStore) -> ObjectSpec:
    catalog = catalog_for_object(spec)
    if catalog.review_status == "reviewed":
        accepted = store.get_catalog(spec.slug, catalog.sha256)
        if accepted is not None:
            return spec
    values = catalog.model_dump(mode="json")
    for item in values["classes"]:
        if not item["definition"].strip():
            item["definition"] = typer.prompt(f"What exactly counts as {item['label']}?")
        typer.echo(f"{item['class_id']}: {item['label']} — {item['definition']}")
    # A setup agent may propose meanings; it cannot publish its own review status.
    values.update(review_status="draft", reviewed_by=None, reviewed_at=None)
    catalog = ClassCatalog.model_validate(values)
    candidate = ObjectSpec.model_validate({**spec.model_dump(mode="json"),
                                          "class_catalog": catalog.model_dump(mode="json")})
    approved = typer.confirm("Approve these ordered class definitions for this object?", default=False)
    if not approved:
        raise typer.BadParameter("Review class definitions in object.yaml before training")
    reviewer = typer.prompt("Name or ID of the person approving the definitions")
    try:
        return _publish_reviewed_object(candidate, store, reviewer, approved=approved)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@setup_app.command("ask")
def setup_ask(request: str = typer.Argument(..., help="Describe the object, labels, notes, and data location."),
              output: Path = typer.Option(Path("setup-draft.yaml"), "--output", "-o"),
              model: str | None = typer.Option(None, "--model")):
    """Ask a configured LiteLLM model clarifying questions and save a validated draft."""
    configured_model = model or os.environ.get("DEFECT_LITELLM_MODEL")
    try:
        draft = propose_setup(request, model=configured_model)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(draft.model_dump(mode="json"), sort_keys=False))
    for question in draft.clarifying_questions:
        typer.echo(f"Question: {question}")
    for item in draft.review_items:
        typer.echo(f"Needs review: {item}")
    typer.echo(f"Validated draft saved to {output}")


@train_app.command("guided")
def train_guided(
    description: str = typer.Argument(..., help="Object details, notes, image and label locations"),
    model: str | None = typer.Option(None, "--model", help="Configured LiteLLM model"),
    project_root: Path = typer.Option(Path("projects"), "--project-root"),
) -> None:
    """Ask up to three rounds of setup questions, review labels, then submit training."""
    service_url = os.getenv("DEFECT_CONTROL_SERVICE_URL")
    if not service_url:
        raise typer.BadParameter("Set DEFECT_CONTROL_SERVICE_URL for durable cloud training")
    configured_model = model or os.getenv("DEFECT_LITELLM_MODEL")
    if not configured_model:
        raise typer.BadParameter("Set DEFECT_LITELLM_MODEL for guided setup")
    catalog_store = catalog_store_from_env()
    if catalog_store is None:
        raise typer.BadParameter("Set DEFECT_CLASS_CATALOG_ROOT to the operator approval store")
    conversation = description
    draft = None
    for _ in range(3):
        draft = propose_setup(conversation, model=configured_model)
        if not draft.clarifying_questions:
            break
        answers = []
        for question in draft.clarifying_questions[:5]:
            answers.append((question, typer.prompt(question)))
        conversation += "\n\nClarifications:\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in answers)
    if draft is None or draft.object is None or not draft.sources or draft.clarifying_questions:
        raise typer.BadParameter("Object classes and labeled source are still incomplete; use `defect setup ask` to review the draft")
    if draft.review_items:
        typer.echo("Setup items to review before data publication:")
        for item in draft.review_items:
            typer.echo(f"- {item}")
        if not typer.confirm("Have you resolved these setup items?", default=False):
            raise typer.BadParameter("Resolve setup review items before training")
    obj = draft.object
    root = project_root / obj.slug
    if root.exists() and any(root.iterdir()):
        existing = ObjectSpec.model_validate(_read_yaml(root / "object.yaml"))
        if existing.model_dump(exclude={"class_catalog"}) != obj.model_dump(exclude={"class_catalog"}):
            raise typer.BadParameter(f"Existing project {root} differs from the proposed object; review it first")
        obj = existing
    else:
        _render_object_templates(root, obj)
    obj = _guided_catalog(obj, catalog_store)
    (root / "object.yaml").write_text(yaml.safe_dump(obj.model_dump(mode="json"), sort_keys=False))
    (root / "class-catalog.yaml").write_text(
        yaml.safe_dump(obj.class_catalog.model_dump(mode="json"), sort_keys=False))
    output_uri = os.getenv("DEFECT_DATASET_OUTPUT_URI") or typer.prompt("GCS folder for versioned datasets")
    dataset_spec = DatasetSpec(object_slug=obj.slug, sources=draft.sources, output_uri=output_uri)
    dataset_config = root / "dataset.yaml"
    if dataset_config.exists() and "CHANGE_ME" not in dataset_config.read_text():
        configured_dataset = DatasetSpec.model_validate(_read_yaml(dataset_config))
        if configured_dataset.sources != draft.sources:
            raise typer.BadParameter("The project's dataset sources differ from the setup draft; review dataset.yaml")
        dataset_spec = configured_dataset
    else:
        dataset_config.write_text(yaml.safe_dump(dataset_spec.model_dump(mode="json"), sort_keys=False))
    dataset_build(dataset_config, root / "object.yaml", root / "dataset-version.yaml", True)
    dataset = DatasetVersion.model_validate(_read_yaml(root / "dataset-version.yaml"))

    runtime_input = os.getenv("DEFECT_DEFAULT_RUNTIME_RECORD") or typer.prompt("Certified runtime YAML path")
    runtime = CertifiedRuntime.model_validate(_read_yaml(Path(runtime_input)))
    if not runtime.certified:
        raise typer.BadParameter("Selected runtime is not certified")
    (root / "runtime.yaml").write_text(yaml.safe_dump(runtime.model_dump(mode="json"), sort_keys=False))
    vertex_input = os.getenv("DEFECT_DEFAULT_VERTEX_CONFIG") or typer.prompt("Vertex job YAML path")
    job = VertexJobConfig.model_validate(_read_yaml(Path(vertex_input)))
    (root / "vertex.yaml").write_text(yaml.safe_dump(job.model_dump(mode="json"), sort_keys=False))
    weights_uri = os.getenv("DEFECT_DINOV3_WEIGHTS_URI") or typer.prompt("Pinned DINOv3 weights location")
    weights_sha = os.getenv("DEFECT_DINOV3_WEIGHTS_SHA256") or typer.prompt("Weights SHA-256")
    experiment_values = {
        "experiment_id": f"{obj.slug}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
        "object_slug": obj.slug,
        "dataset_version_id": dataset.version_id,
        "runtime_id": runtime.runtime_id,
        "catalog_sha256": obj.class_catalog.sha256,
        "model": ModelSpec(weights_uri=weights_uri, weights_sha256=weights_sha).model_dump(mode="json"),
    }
    allowed_experiment_fields = {
        "epochs", "batch_size", "learning_rate", "optimizer", "loss", "focal_gamma",
        "horizontal_flip_probability", "class_weights", "seed", "max_review_error_rate",
    }
    unknown_experiment_fields = set(draft.experiment) - allowed_experiment_fields
    if unknown_experiment_fields:
        raise typer.BadParameter("Setup agent proposed unsupported experiment fields: "
                                 + ", ".join(sorted(unknown_experiment_fields)))
    experiment_values.update(draft.experiment)
    experiment = ExperimentConfig.model_validate(experiment_values)
    (root / "experiment.yaml").write_text(yaml.safe_dump(experiment.model_dump(mode="json"), sort_keys=False))
    request_path = root / "run-request.yaml"
    request_path.write_text(yaml.safe_dump({
        "experiment": experiment.model_dump(mode="json"), "job": job.model_dump(mode="json"),
        "dataset": dataset.model_dump(mode="json"), "runtime": runtime.model_dump(mode="json"),
        "classes": obj.classes, "platform": {"control_service_url": service_url},
    }, sort_keys=False))
    _run_submit(request_path, None)


@app.callback()
def main():
    """Configuration is read from each request plus DEFECT_STATE_DATABASE_URL."""
    configure_logging()
