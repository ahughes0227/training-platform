"""Human-approved model release commands."""

from __future__ import annotations

import json
import os
from uuid import uuid4

import typer

from defect_platform.contracts import ModelRelease, RunState
from defect_platform.control.client import ControlAPIClient
from defect_platform.control.store import database_url_from_env, run_store_from_env
from defect_platform.serve.deploy import apply_rayservice, rayservice_manifest
from defect_platform.serve.ledger import GCSReleaseLedger, SqlReleaseLedger
from defect_platform.serve.releases import MlflowRegistryAdapter, ReleaseManager


release_app = typer.Typer(help="Review, promote, deploy, and roll back a trained model.", no_args_is_help=True)


def _tracking_uri() -> str:
    uri = os.getenv("DEFECT_MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise typer.BadParameter("Set DEFECT_MLFLOW_TRACKING_URI before managing releases")
    return uri


def _ledger() -> GCSReleaseLedger | SqlReleaseLedger:
    gcs_uri = os.getenv("DEFECT_RELEASE_GCS_URI")
    if gcs_uri:
        return GCSReleaseLedger(gcs_uri)
    return SqlReleaseLedger(database_url_from_env())


def _run(run_id: str):
    control_url = os.getenv("DEFECT_CONTROL_SERVICE_URL")
    if control_url:
        return ControlAPIClient(control_url).get(run_id)
    return run_store_from_env().get(run_id)


def _deploy(release: ModelRelease, tracking_uri: str, context: str | None) -> None:
    manifest = rayservice_manifest(
        release,
        mlflow_tracking_uri=tracking_uri,
        namespace=os.getenv("DEFECT_RAY_NAMESPACE", "defect-serving"),
        ray_version=os.getenv("DEFECT_RAY_VERSION", "2.58.0"),
        gpu_count=float(os.getenv("DEFECT_SERVE_GPUS", "1")),
    )
    apply_rayservice(manifest, context=context)


@release_app.command("stage")
def stage(
    run_id: str = typer.Argument(..., help="Successful training run ID"),
    model_version: str = typer.Option(..., "--model-version", help="MLflow candidate version"),
    serving_image_digest: str = typer.Option(..., "--serving-image-digest"),
    model_name: str | None = typer.Option(None, "--model-name"),
) -> None:
    """Record a candidate linked to its successful run and MLflow model version."""
    run = _run(run_id)
    if run is None:
        raise typer.BadParameter(f"Run {run_id} was not found")
    if run.state != RunState.SUCCEEDED or not run.mlflow_run_id:
        raise typer.BadParameter("Only a successful run with an MLflow run ID can be staged")
    if "@sha256:" not in serving_image_digest:
        raise typer.BadParameter("Serving image must use an immutable sha256 digest")
    name = model_name or f"defect-{run.object_slug}"
    registry = MlflowRegistryAdapter(_tracking_uri())
    version = registry.get_model_version(name, model_version)
    if str(version.run_id) != run.mlflow_run_id:
        raise typer.BadParameter("MLflow version does not belong to the selected training run")
    release = ModelRelease(
        object_slug=run.object_slug,
        model_name=name,
        model_version=model_version,
        release_id="rel-" + uuid4().hex[:20],
        run_id=run.run_id,
        dataset_version_id=run.dataset_version_id,
        runtime_id=run.runtime_id,
        serving_image_digest=serving_image_digest,
        state="staged",
    )
    _ledger().save(release)
    typer.echo(json.dumps(release.model_dump(mode="json"), indent=2))


@release_app.command("show")
def show(release_id: str) -> None:
    """Show the model, lineage, approval, and serving image for one release."""
    typer.echo(json.dumps(_ledger().get(release_id).model_dump(mode="json"), indent=2))


@release_app.command("promote")
def promote(
    release_id: str,
    approver: str = typer.Option(..., "--approver", help="Human approver identity"),
    context: str | None = typer.Option(None, "--kube-context"),
    deploy: bool = typer.Option(True, "--deploy/--no-deploy"),
) -> None:
    """Explicitly approve a staged version and update the live RayService."""
    uri = _tracking_uri()
    ledger = _ledger()
    candidate = ledger.get(release_id)
    manager = ReleaseManager(MlflowRegistryAdapter(uri), ledger)
    result = manager.promote(
        candidate, approver,
        deploy=(lambda item: _deploy(item, uri, context)) if deploy else None,
    )
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@release_app.command("rollback")
def rollback(
    object_slug: str,
    approver: str = typer.Option(..., "--approver", help="Human approver identity"),
    context: str | None = typer.Option(None, "--kube-context"),
    deploy: bool = typer.Option(True, "--deploy/--no-deploy"),
) -> None:
    """Restore the preceding approved model version and RayService."""
    uri = _tracking_uri()
    manager = ReleaseManager(MlflowRegistryAdapter(uri), _ledger())
    result = manager.rollback(
        object_slug, approver,
        deploy=(lambda item: _deploy(item, uri, context)) if deploy else None,
    )
    typer.echo(json.dumps({"restored_release": result.release_id, "model_version": result.model_version}, indent=2))
