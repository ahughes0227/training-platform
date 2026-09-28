"""Vertex CustomJob entrypoint for a serialized training request.

Run with ``python -m defect_platform.trainer.runner --request gs://.../request.json``.
The request contains ExperimentConfig, DatasetVersion and classes. Experiment
results are written to a local directory and uploaded to output_uri when present.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from ..contracts import CertifiedRuntime, DatasetVersion, ExperimentConfig
from ..dataset import load_dataset_semantics
from .training import train_experiment


def _read_uri(uri: str) -> bytes:
    if not uri.startswith("gs://"):
        return Path(uri).read_bytes()
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError("GCS request loading requires google-cloud-storage") from exc
    bucket, _, key = uri[5:].partition("/")
    return storage.Client().bucket(bucket).blob(key).download_as_bytes()


def _upload_tree(local_dir: Path, destination: str) -> None:
    if not destination.startswith("gs://"):
        target = Path(destination); target.mkdir(parents=True, exist_ok=True)
        for path in local_dir.rglob("*"):
            if path.is_file():
                out = target / path.relative_to(local_dir)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(path.read_bytes())
        return
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError("GCS result upload requires google-cloud-storage") from exc
    bucket, _, prefix = destination[5:].partition("/")
    prefix = prefix.rstrip("/")
    client = storage.Client()
    for path in local_dir.rglob("*"):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(local_dir).as_posix()}".lstrip("/")
            client.bucket(bucket).blob(key).upload_from_filename(str(path))


def run_request(request: dict, output_dir: str | Path | None = None) -> dict:
    """Validate request contracts then execute a complete trainer run."""
    config = ExperimentConfig.model_validate(request["experiment"])
    dataset = DatasetVersion.model_validate(request["dataset"])
    classes = list(request["classes"])
    if config.object_slug != dataset.object_slug or config.dataset_version_id != dataset.version_id:
        raise ValueError("experiment object/dataset version do not match DatasetVersion")
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("classes must contain two or more unique labels in canonical order")
    runtime = None
    if "runtime" in request:
        runtime = CertifiedRuntime.model_validate(request["runtime"])
        if not runtime.certified or runtime.runtime_id != config.runtime_id:
            raise ValueError("request runtime must be certified and match experiment runtime_id")
        if "semantic_refs" not in request:
            raise ValueError("certified runtime requests require semantic_refs")
    semantic_manifest = load_dataset_semantics(dataset)
    if semantic_manifest.catalog.labels != classes:
        raise ValueError("classes do not match verified dataset semantic catalog order")
    if "semantic_refs" in request:
        refs = request["semantic_refs"]
        expected_keys = {"dataset_semantic_sha256", "catalog_sha256"}
        if not isinstance(refs, dict) or set(refs) != expected_keys:
            raise ValueError("semantic_refs must contain exactly dataset_semantic_sha256 and catalog_sha256")
        if any(not isinstance(refs[key], str) or not re.fullmatch(r"[0-9a-f]{64}", refs[key])
               for key in expected_keys):
            raise ValueError("semantic_refs values must be lowercase SHA256 fingerprints")
        expected_refs = {"dataset_semantic_sha256": semantic_manifest.sha256,
                         "catalog_sha256": semantic_manifest.catalog.sha256}
        if refs != expected_refs:
            raise ValueError("request semantic_refs do not match the verified dataset manifest")
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="defect-training-")
        output_dir = temp_dir.name
    destination = request.get("output_uri") or dataset.root_uri.rstrip("/") + "/runs/" + config.experiment_id
    report = train_experiment(
        config, dataset.root_uri, output_dir, classes,
        shard_uris=dataset.shard_uris,
        semantic_manifest=semantic_manifest,
        dataset_sha256=dataset.sha256,
        mlflow_tracking_uri=request.get("mlflow_tracking_uri"),
        mlflow_run_id=request.get("mlflow_run_id"),
        runtime_image_digest=runtime.image_digest if runtime else None,
        runtime_source_commit=runtime.source_commit if runtime else None,
    )
    report["output_uri"] = destination
    (Path(output_dir) / "evaluation.json").write_text(json.dumps(report, indent=2))
    _upload_tree(Path(output_dir), destination)
    return report


def main() -> None:
    from ..telemetry import configure_logging
    configure_logging()
    parser = argparse.ArgumentParser(description="Train DINOv3 + MLP from a platform run request")
    parser.add_argument("--request", required=True, help="JSON file or gs:// URI")
    parser.add_argument("--output-dir", help="Optional local output directory")
    args = parser.parse_args()
    request = json.loads(_read_uri(args.request))
    report = run_request(request, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
