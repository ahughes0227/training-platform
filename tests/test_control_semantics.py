from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import pytest

from defect_platform.control.controller import RunController
from defect_platform.control.store import SQLiteRunStore
from defect_platform.dataset import load_dataset_semantics
from defect_platform.infrastructure_contract import InfrastructureCapabilities
from defect_platform.semantics import ClassCatalog
from test_control import FakeVertex, FakeWorkflow, fixtures


class CountingTracker:
    tracking_uri = "unused"

    def __init__(self):
        self.starts = 0

    def start(self, **_kwargs):
        self.starts += 1
        return "mlflow-run"

    def finish(self, *_args, **_kwargs):
        pass


@dataclass
class UnreviewedCatalog:
    sha256: str
    object_slug: str
    labels: list[str]
    review_status: Literal["draft"] = "draft"


@dataclass
class CatalogStore:
    catalog: ClassCatalog | UnreviewedCatalog | None

    def get_catalog(self, object_slug: str, catalog_sha256: str) -> ClassCatalog | None:
        if self.catalog is None:
            return None
        if self.catalog.object_slug != object_slug or self.catalog.sha256 != catalog_sha256:
            return None
        return cast(ClassCatalog, self.catalog)


def _controller(tmp_path, *, catalogs, loader=None):
    _runtime, _experiment, _dataset, _job, _runtime_catalog, trusted = fixtures(tmp_path)

    class RuntimeDataset:
        def get_certified(self, runtime_id: str):
            return _runtime if runtime_id == _runtime.runtime_id else None

        def get(self, version_id: str):
            return _dataset if version_id == _dataset.version_id else None

    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    vertex = FakeVertex()
    workflow = FakeWorkflow(store)
    tracker = CountingTracker()
    controller = RunController(
        store=store,
        runtimes=RuntimeDataset(),
        datasets=RuntimeDataset(),
        catalogs=catalogs if catalogs is not None else trusted,
        semantic_loader=loader,
        workflows=workflow,
        vertex=vertex,
        tracker=tracker,
    )
    manifest = load_dataset_semantics(_dataset)
    approved = trusted.get_catalog(_dataset.object_slug, manifest.catalog.sha256)
    assert approved is not None
    return controller, store, vertex, workflow, tracker, _experiment, _job, _dataset, approved


@pytest.mark.parametrize(
    "case",
    [
        "missing_store",
        "missing_catalog",
        "unreviewed_catalog",
        "manifest_mismatch",
        "mutated_manifest",
    ],
)
def test_semantic_admission_rejects_before_run_or_external_side_effects(tmp_path, case):
    controller, store, vertex, workflow, tracker, experiment, job, dataset, approved = _controller(
        tmp_path, catalogs=None
    )
    good = load_dataset_semantics(dataset)
    if case == "missing_store":
        controller.catalogs = None
    elif case == "missing_catalog":
        controller.catalogs = CatalogStore(None)
    elif case == "unreviewed_catalog":
        controller.catalogs = CatalogStore(
            UnreviewedCatalog(
                sha256=approved.sha256,
                object_slug=approved.object_slug,
                labels=approved.labels,
            )
        )
    else:
        bad = good.model_copy(
            update={"kind": "model"}
            if case == "mutated_manifest"
            else {"dataset_version_id": "other-version"}
        )
        controller.semantic_loader = lambda _version: bad
    expected_errors = {
        "missing_store": "trusted class catalog store is not configured",
        "missing_catalog": "matching class catalog is not present",
        "unreviewed_catalog": "explicitly reviewed class catalog",
        "manifest_mismatch": "manifest is missing, unverified, or mismatched",
        "mutated_manifest": "model semantic manifests require complete training policy",
    }
    with pytest.raises((ValueError, RuntimeError), match=expected_errors[case]):
        controller.submit(experiment=experiment, job=job, idempotency_key="blocked")
    assert store.list() == []
    assert vertex.calls == []
    assert workflow.calls == []
    assert tracker.starts == 0


def test_swapped_client_classes_reject_before_side_effects(tmp_path):
    controller, store, vertex, workflow, tracker, experiment, job, _dataset, _catalog = _controller(
        tmp_path, catalogs=None
    )
    with pytest.raises(ValueError, match="requested class order"):
        controller.submit(
            experiment=experiment,
            job=job,
            idempotency_key="swapped-classes",
            classes=["dent", "crack"],
        )
    assert store.list() == []
    assert vertex.calls == [] and workflow.calls == [] and tracker.starts == 0


def test_client_class_order_cannot_override_trusted_manifest_and_refs_are_persisted(tmp_path):
    controller, store, vertex, workflow, _tracker, experiment, job, dataset, catalog = _controller(
        tmp_path, catalogs=None
    )
    run, created = controller.submit(
        experiment=experiment, job=job, idempotency_key="semantic-good"
    )
    assert created and run.state.value == "submitted"
    payload = store.get_payload(run.run_id)
    assert payload is not None
    assert payload["classes"] == catalog.labels
    assert payload["semantic_refs"] == {
        "dataset_semantic_sha256": dataset.semantic_sha256,
        "catalog_sha256": catalog.sha256,
    }
    assert run.catalog_sha256 == catalog.sha256
    assert run.dataset_semantic_sha256 == dataset.semantic_sha256
    assert workflow.calls[0][1]["semantic_refs"] == payload["semantic_refs"]
    controller.dispatch_vertex(run.run_id)
    assert vertex.calls[0]["classes"] == catalog.labels
    assert vertex.calls[0]["semantic_refs"] == payload["semantic_refs"]


def test_dispatch_rechecks_catalog_before_vertex_side_effect(tmp_path):
    controller, _store, vertex, _workflow, _tracker, experiment, job, _dataset, _catalog = (
        _controller(tmp_path, catalogs=None)
    )
    run, _ = controller.submit(experiment=experiment, job=job, idempotency_key="dispatch-recheck")
    vertex.calls.clear()
    controller.catalogs = CatalogStore(None)
    with pytest.raises(ValueError, match="trusted class catalog"):
        controller.dispatch_vertex(run.run_id)
    assert vertex.calls == []


def test_production_admission_can_require_operator_pinned_capability(tmp_path):
    controller, store, vertex, workflow, tracker, experiment, job, _dataset, _catalog = _controller(
        tmp_path, catalogs=None
    )
    controller.require_capabilities = True
    with pytest.raises(ValueError, match="infrastructure capabilities are required"):
        controller.submit(experiment=experiment, job=job, idempotency_key="no-capability")
    assert store.list() == []
    assert vertex.calls == [] and workflow.calls == [] and tracker.starts == 0


def test_capability_mlflow_endpoint_mismatch_rejects_before_side_effects(tmp_path):
    from types import SimpleNamespace

    controller, store, vertex, workflow, tracker, experiment, job, _dataset, _catalog = _controller(
        tmp_path, catalogs=None
    )
    controller.require_capabilities = True
    controller.capabilities = cast(
        InfrastructureCapabilities,
        SimpleNamespace(
            sha256="f" * 64,
            mlflow_uri="https://observed-mlflow.example",
            validate_training=lambda *_args, **_kwargs: None,
        ),
    )
    with pytest.raises(ValueError, match="MLflow tracking URI differs"):
        controller.submit(experiment=experiment, job=job, idempotency_key="bad-mlflow-endpoint")
    assert store.list() == []
    assert vertex.calls == [] and workflow.calls == [] and tracker.starts == 0
