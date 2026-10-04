"""Actual CSV build, DINOv3 fixture training, portable reload, and Ray HTTP."""

import base64
import csv
import importlib
import io
import json
import random
import shutil
import signal
import socket
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from defect_platform.catalog_store import DirectoryCatalogStore
from defect_platform.contracts import (
    DatasetSpec,
    ExperimentConfig,
    LabelSource,
    ModelSpec,
    ObjectSpec,
    SourceKind,
)
from defect_platform.dataset import build_dataset, load_dataset_semantics
from defect_platform.semantics import ClassCatalog, ClassDefinition, read_manifest
from defect_platform.serve.api import PredictionService, build_ray_app
from defect_platform.trainer.inference import load_inference_bundle, predict_crop
from defect_platform.trainer.runner import run_request
from defect_platform.trainer.weights import artifact_sha256

pytestmark = pytest.mark.integration


class _OptionalImportTimeout(TimeoutError):
    """Raised when an optional integration package cannot import promptly."""


@contextmanager
def _bounded_import(timeout: float = 15.0):
    """Bound optional package imports so a broken plugin cannot hang pytest."""

    if not hasattr(signal, "setitimer"):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def alarm_handler(_signum, _frame):
        raise _OptionalImportTimeout

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _optional_import(name: str, reason: str):
    """Import a heavyweight optional dependency with a diagnostic skip path."""

    try:
        with _bounded_import():
            return importlib.import_module(name)
    except _OptionalImportTimeout:
        pytest.skip(f"optional dependency {name!r} did not import within 15s: {reason}")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"optional dependency {name!r} is unavailable ({exc}): {reason}")


def _ray_environment_error(exc: BaseException) -> bool:
    """Return whether a Ray startup error is caused by the local harness.

    Ray starts several subprocesses and binds a loopback HTTP proxy.  Sandboxed
    runners commonly reject one of those operations even though the serving
    code is healthy.  Keep this list limited to startup/resource failures so
    errors raised by the fixture or prediction path still fail the test.
    """

    message = str(exc).lower()
    markers = (
        "address already in use",
        "cannot allocate memory",
        "can't start gcs",
        "failed to connect to gcs",
        "failed to start ray",
        "no space left on device",
        "permission denied",
        "resource temporarily unavailable",
        "too many open files",
    )
    return isinstance(exc, (OSError, RuntimeError, TimeoutError)) and any(
        marker in message for marker in markers
    )


def _free_loopback_port() -> int:
    """Select a loopback port, with a useful failure for restricted runners."""

    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])
    except OSError as exc:
        pytest.skip(f"loopback sockets are unavailable for Ray HTTP integration: {exc}")


def _wait_for_http(url: str, *, timeout: float = 30.0) -> None:
    """Wait for the local Serve proxy to accept requests before asserting output."""

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(0.2)
    detail = f" ({last_error})" if last_error else ""
    raise AssertionError(f"Ray Serve did not become ready within {timeout:g}s{detail}")


def test_approved_dataset_dinov3_bundle_and_real_ray_http(tmp_path, monkeypatch):
    # Keep optional dependencies out of collection and make a missing install
    # visible as an intentional skip.  This test remains strict after imports.
    ray = _optional_import("ray", "install the `serve` extra for Ray HTTP integration")
    torch = _optional_import("torch", "install the `train` extra for Ray HTTP integration")
    transformers = _optional_import(
        "transformers", "install the `train` extra for Ray HTTP integration"
    )
    serve = _optional_import("ray.serve", "install the `ray[serve]` extra")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("DEFECT_SERVE_GPUS", "0")
    torch.manual_seed(19)
    catalog = ClassCatalog(
        catalog_id="panel-defects",
        object_slug="panel",
        review_status="reviewed",
        reviewed_by="fixture-human",
        reviewed_at=datetime.now(UTC),
        classes=[
            ClassDefinition(
                class_id="crack",
                label="crack",
                definition="Material separation",
                aliases=["fracture"],
            ),
            ClassDefinition(class_id="dent", label="dent", definition="Inward deformation"),
        ],
    )
    store = DirectoryCatalogStore(tmp_path / "approved")
    store.publish(catalog)
    obj = ObjectSpec(
        slug="panel", display_name="Panel", classes=catalog.labels, class_catalog=catalog
    )
    source = tmp_path / "labels.csv"
    rng = random.Random(19)
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_uri", "label", "sample_id"])
        for index in range(32):
            color = (180, 10, 10) if index % 2 == 0 else (10, 10, 180)
            image = Image.new("RGB", (32, 32))
            image.putdata(
                [
                    tuple(min(255, value + rng.randrange(50)) for value in color)
                    for _ in range(32 * 32)
                ]
            )
            path = tmp_path / f"crop-{index}.png"
            image.save(path)
            writer.writerow([str(path), "fracture" if index % 2 == 0 else "dent", f"s-{index}"])
    spec = DatasetSpec(
        object_slug=obj.slug,
        output_uri=str(tmp_path / "datasets"),
        sources=[
            LabelSource(kind=SourceKind.CSV, location=str(source), sample_id_column="sample_id")
        ],
        shard_max_samples=4,
    )
    dataset = build_dataset(spec, obj, near_duplicate_distance=0)
    dataset_manifest = load_dataset_semantics(dataset)
    backbone = transformers.AutoModel.from_config(
        transformers.AutoConfig.for_model(
            "dinov3_vit",
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            num_register_tokens=2,
            patch_size=16,
            image_size=32,
        )
    )
    weights = tmp_path / "weights"
    backbone.save_pretrained(weights)
    experiment = ExperimentConfig(
        experiment_id="semantic-integration",
        object_slug=obj.slug,
        dataset_version_id=dataset.version_id,
        runtime_id="local-fixture",
        epochs=1,
        batch_size=4,
        catalog_sha256=catalog.sha256,
        horizontal_flip_probability=0,
        model=ModelSpec(
            weights_uri=str(weights),
            weights_sha256=artifact_sha256(weights),
            image_size=32,
            hidden_dim=8,
            dropout=0,
            unfreeze_last_n=1,
        ),
    )
    output = tmp_path / "output"
    report = run_request(
        {
            "experiment": experiment.model_dump(mode="json"),
            "dataset": dataset.model_dump(mode="json"),
            "classes": catalog.labels,
            "semantic_refs": {
                "dataset_semantic_sha256": dataset_manifest.sha256,
                "catalog_sha256": catalog.sha256,
            },
            "output_uri": str(tmp_path / "published"),
        },
        output_dir=output,
    )
    model_dir = output / "model"
    manifest = read_manifest(model_dir / "semantics.json")
    assert manifest.parent_semantic_sha256 == dataset.semantic_sha256
    assert manifest.dataset_sha256 == dataset.sha256
    assert manifest.catalog.sha256 == catalog.sha256
    assert report["validation"]["confusion"] and report["test"]["confusion"]
    shutil.rmtree(weights)
    expected_bundle = artifact_sha256(model_dir / "integrity.json")
    bundle = load_inference_bundle(
        model_dir,
        expected_semantic_sha256=manifest.sha256,
        expected_catalog_sha256=catalog.sha256,
        expected_bundle_sha256=expected_bundle,
    )
    crop = Image.open(tmp_path / "crop-0.png").convert("RGB")
    expected = predict_crop(bundle, crop)
    port = _free_loopback_port()
    model_location = str(model_dir)
    manifest_hash = manifest.sha256
    catalog_values = catalog.model_dump(mode="json")

    def factory():
        checked = load_inference_bundle(
            model_location,
            expected_semantic_sha256=manifest_hash,
            expected_catalog_sha256=catalog.sha256,
            expected_bundle_sha256=expected_bundle,
        )
        return PredictionService(
            lambda image, explain: predict_crop(checked, image, explain),
            "defect-panel",
            "fixture-1",
            catalog=ClassCatalog.model_validate(catalog_values),
            semantic_sha256=manifest_hash,
        )

    ray_temp = tempfile.mkdtemp(prefix="defect-ray-", dir="/tmp")
    ray_started = False
    serve_started = False
    try:
        try:
            ray.init(
                num_cpus=2,
                include_dashboard=False,
                _temp_dir=ray_temp,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
                        "HF_HUB_OFFLINE": "1",
                    }
                },
            )
            ray_started = True
        except Exception as exc:
            if _ray_environment_error(exc):
                pytest.skip(f"local Ray runtime is unavailable: {exc}")
            raise
        try:
            serve.start(http_options={"host": "127.0.0.1", "port": port})
            serve_started = True
        except Exception as exc:
            if _ray_environment_error(exc):
                pytest.skip(f"local Ray Serve HTTP proxy is unavailable: {exc}")
            raise
        serve.run(build_ray_app(factory), name="semantic-fixture", route_prefix="/")
        _wait_for_http(f"http://127.0.0.1:{port}/health")
        encoded = io.BytesIO()
        crop.save(encoded, format="PNG")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/predict",
            data=json.dumps(
                {"image_base64": base64.b64encode(encoded.getvalue()).decode(), "explain": True}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        assert result["class_name"] == expected["class"]
        assert result["class_id"] == expected["class_id"]
        assert result["review_required"] == expected["review"]
        assert result["catalog_sha256"] == catalog.sha256
        assert result["semantic_sha256"] == manifest.sha256
        assert result["model_version"] == "fixture-1"
        assert 0 <= result["confidence"] <= 1
        heatmap = Image.open(io.BytesIO(base64.b64decode(result["heatmap_png_base64"])))
        assert heatmap.size == crop.size
    finally:
        if serve_started:
            serve.shutdown()
        if ray_started:
            ray.shutdown()
        shutil.rmtree(ray_temp, ignore_errors=True)
