"""Vertex CustomJob entrypoint for a serialized training request.

Run with ``python -m defect_platform.trainer.runner --request gs://.../request.json``.
The request contains ExperimentConfig, DatasetVersion and classes. Experiment
results are written to a local directory and uploaded to output_uri when present.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ..contracts import DatasetVersion, ExperimentConfig
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
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="defect-training-")
        output_dir = temp_dir.name
    destination = request.get("output_uri") or dataset.root_uri.rstrip("/") + "/runs/" + config.experiment_id
    report = train_experiment(
        config, dataset.root_uri, output_dir, classes,
        shard_uris=dataset.shard_uris,
        mlflow_tracking_uri=request.get("mlflow_tracking_uri"),
        mlflow_run_id=request.get("mlflow_run_id"),
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
