"""Immutable WebDataset version builder for local paths and GCS."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from defect_platform.contracts import DatasetSpec, DatasetVersion, ObjectSpec
from defect_platform.semantics import (
    catalog_for_object,
    make_dataset_manifest,
    validate_label_mapping,
)

from .duplicates import DuplicateReport, find_duplicates
from .labels import PreviewResult, map_label, preview_dataset
from .sources import LabelRow, read_label_source
from .splits import assign_splits


@dataclass(slots=True)
class _Sample:
    row: LabelRow
    label: str
    content_path: Path
    content_sha256: str
    key: str


def build_dataset(
    spec: DatasetSpec,
    object_spec: ObjectSpec,
    *,
    rows: list[LabelRow] | None = None,
    bq_client: object | None = None,
    near_duplicate_distance: int = 6,
) -> DatasetVersion:
    """Validate sources and publish one content-addressed immutable dataset.

    GCS publication uses create-only object writes. A successful build is committed
    by writing `_COMMIT.json` last. Calling again with identical inputs returns the
    existing version; attempting to overwrite a different version is rejected.
    """
    _check_object(spec, object_spec)
    catalog = catalog_for_object(object_spec)
    validate_label_mapping(catalog, spec.label_mapping)
    if rows is None:
        rows = [row for source in spec.sources for row in read_label_source(source, bq_client=bq_client)]
    preview = preview_dataset(spec, object_spec, rows=rows)
    if preview.exceptions:
        examples = "; ".join(
            f"{item.reason} at {item.source}:{item.row_number} label={item.raw_label!r}"
            for item in preview.exceptions[:8]
        )
        raise ValueError(f"Dataset has {len(preview.exceptions)} review exception(s): {examples}")
    with tempfile.TemporaryDirectory(prefix="defect-dataset-") as stage:
        stage_root = Path(stage)
        samples = _load_samples(rows, spec, object_spec, stage_root / "images")
        duplicate_report = find_duplicates(
            ((sample.row.image_uri, sample.content_path) for sample in samples),
            max_hamming_distance=near_duplicate_distance,
        )
        if duplicate_report.errors:
            first_uri, message = next(iter(duplicate_report.errors.items()))
            raise ValueError(f"Cannot validate image {first_uri!r}: {message}")
        _raise_duplicate_label_conflicts(duplicate_report, samples)

        # Connect duplicate and product groups so related images cannot cross splits.
        union = _UnionFind([sample.row.image_uri for sample in samples])
        for pair in duplicate_report.pairs:
            union.union(pair.left, pair.right)
        for sample in samples:
            if sample.row.group_id:
                union.union(sample.row.image_uri, "group:" + sample.row.group_id)
        group_ids = [union.find(sample.row.image_uri) for sample in samples]
        split_names = assign_splits([sample.label for sample in samples], group_ids, spec.split)
        records = []
        for index, (sample, split) in enumerate(zip(samples, split_names)):
            records.append(
                {
                    "key": sample.key,
                    "image_uri": sample.row.image_uri,
                    "label": sample.label,
                    "sample_id": sample.row.sample_id,
                    "group_id": sample.row.group_id,
                    "source": sample.row.source,
                    "source_row": sample.row.row_number,
                    "content_sha256": sample.content_sha256,
                    "split": split,
                    "duplicate_group": group_ids[index],
                }
            )
        image_hashes = {sample.row.image_uri: sample.content_sha256 for sample in samples}
        input_rows = []
        for row in rows:
            canonical, _ = map_label(row.raw_label, object_spec.classes, spec.label_mapping, catalog)
            input_rows.append(
                {
                    "source": row.source,
                    "source_row": row.row_number,
                    "image_uri": row.image_uri,
                    "raw_label": row.raw_label,
                    "label": canonical,
                    "sample_id": row.sample_id,
                    "group_id": row.group_id,
                    "content_sha256": image_hashes.get(row.image_uri),
                }
            )
        source_snapshot = {
            "format_version": 1,
            "object_slug": object_spec.slug,
            "classes": object_spec.classes,
            "class_catalog": catalog.model_dump(mode="json"),
            "label_mapping": spec.label_mapping,
            "label_review": spec.label_review.model_dump(mode="json") if spec.label_review else None,
            "sources": [source.model_dump(mode="json") for source in spec.sources],
            "rows": records,
            "input_rows": input_rows,
            "duplicate_report": {
                "exact": [asdict(pair) for pair in duplicate_report.exact_pairs],
                "near": [asdict(pair) for pair in duplicate_report.near_pairs],
                "group_ids_available": sum(bool(row.group_id) for row in rows),
                "row_count": len(rows),
            },
            "split": spec.split.model_dump(mode="json"),
        }
        source_path = stage_root / "source-snapshot.json"
        _write_json_file(source_path, source_snapshot)
        source_digest = _sha256_file(source_path)
        identity = {
            "object_slug": object_spec.slug,
            "classes": object_spec.classes,
            "catalog_sha256": catalog.sha256,
            "rows": [
                [record["content_sha256"], record["label"], record["split"], record["group_id"]]
                for record in records
            ],
            "mapping": spec.label_mapping,
            "split": spec.split.model_dump(mode="json"),
            "shard_max_samples": spec.shard_max_samples,
            "provenance_sha256": source_digest,
        }
        version_id = "ds-" + hashlib.sha256(_json_bytes(identity)).hexdigest()[:20]
        assignments_sha256 = _sha256(_json_bytes([
            {"key": record["key"], "class_id": catalog.decode(catalog.encode(record["label"])).class_id,
             "split": record["split"]} for record in records
        ]))
        semantic_manifest = make_dataset_manifest(
            object_spec, spec, version_id, source_sha256=source_digest,
            split_assignments_sha256=assignments_sha256,
        )
        semantic_bytes = _json_bytes({"manifest": semantic_manifest.model_dump(mode="json"),
                                       "sha256": semantic_manifest.sha256})
        semantic_path = stage_root / "semantics.json"
        semantic_path.write_bytes(semantic_bytes)
        build_dir = stage_root / "artifacts"
        build_dir.mkdir()
        _build_local_artifacts(samples, records, source_path, spec.shard_max_samples, build_dir)
        manifest_path = build_dir / "manifest.jsonl"
        manifest_digest = _sha256_file(manifest_path)
        shard_names = sorted(path.relative_to(build_dir).as_posix() for path in build_dir.glob("shards/*.tar"))
        shard_hashes = {name: _sha256_file(build_dir / name) for name in shard_names}
        manifest = {
            "format_version": 1,
            "version_id": version_id,
            "object_slug": object_spec.slug,
            "classes": object_spec.classes,
            "created_content_digest": version_id,
            "sample_count": len(records),
            "sample_counts": _counts(records),
            "manifest_sha256": manifest_digest,
            "source_snapshot_sha256": source_digest,
            "shards": shard_hashes,
            "duplicate_report": source_snapshot["duplicate_report"],
            "split": spec.split.model_dump(mode="json"),
            "semantic_manifest_sha256": semantic_manifest.sha256,
            "semantic_manifest_file_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
        }
        _write_json_file(build_dir / "manifest.json", manifest)
        shutil.copyfile(source_path, build_dir / "source-snapshot.json")
        shutil.copyfile(semantic_path, build_dir / "semantics.json")
        manifest_file_digest = _sha256_file(build_dir / "manifest.json")
        content_digest = _sha256(
            _json_bytes(
                {
                    "manifest": manifest_file_digest,
                    "manifest_jsonl": manifest_digest,
                    "source": source_digest,
                    "shards": shard_hashes,
                    "semantics": hashlib.sha256(semantic_bytes).hexdigest(),
                }
            )
        )
        _write_json_file(build_dir / "_COMMIT.json", {"version_id": version_id, "sha256": content_digest})
        root_uri = _join_uri(spec.output_uri, version_id)
        _publish(root_uri, build_dir, content_digest)
        shard_uris = {
            split: [_join_uri(root_uri, name) for name in shard_names if f"/{split}-" in "/" + name]
            for split in ("train", "validation", "test")
        }
        return DatasetVersion(
            version_id=version_id,
            object_slug=object_spec.slug,
            root_uri=root_uri,
            manifest_uri=_join_uri(root_uri, "manifest.json"),
            shard_uris=shard_uris,
            sample_counts=manifest["sample_counts"],
            sha256=content_digest,
            source_snapshot_uri=_join_uri(root_uri, "source-snapshot.json"),
            semantic_manifest_uri=_join_uri(root_uri, "semantics.json"),
            semantic_sha256=semantic_manifest.sha256,
        )


def _load_samples(rows: list[LabelRow], spec: DatasetSpec, object_spec: ObjectSpec, stage_dir: Path) -> list[_Sample]:
    result: list[_Sample] = []
    uri_to_hash_labels: dict[str, tuple[str, str]] = {}
    stage_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        label, _ = map_label(row.raw_label, object_spec.classes, spec.label_mapping, catalog_for_object(object_spec))
        if label is None:  # preview already checked; protects future callers
            raise ValueError(f"Unmapped label {row.raw_label!r} at {row.source}:{row.row_number}")
        previous = uri_to_hash_labels.get(row.image_uri)
        if previous:
            if previous[0] != label:
                raise ValueError(f"Conflicting labels for image {row.image_uri!r}: {previous[0]!r}, {label!r}")
            # Repeated same image and label is a duplicate row. Keep provenance in
            # snapshot separately but store the training example only once.
            continue
        stage_path = stage_dir / f"{i:012d}.image"
        digest = _copy_image_to_path(row.image_uri, stage_path)
        uri_to_hash_labels[row.image_uri] = (label, digest)
        key = hashlib.sha256(f"{row.image_uri}\0{digest}\0{i}".encode()).hexdigest()[:32]
        result.append(_Sample(row, label, stage_path, digest, key))
    return result


def _raise_duplicate_label_conflicts(report: DuplicateReport, samples: list[_Sample]) -> None:
    labels = {sample.row.image_uri: sample.label for sample in samples}
    for pair in report.pairs:
        left, right = labels.get(pair.left), labels.get(pair.right)
        if left is not None and right is not None and left != right:
            raise ValueError(
                f"{pair.kind} duplicate images have conflicting labels: "
                f"{pair.left!r}={left!r}, {pair.right!r}={right!r}; review labels before building"
            )


def _build_local_artifacts(
    samples: list[_Sample], records: list[dict], source_path: Path, max_samples: int, artifacts: Path
) -> None:
    manifest_path = artifacts / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            output.write("\n")
    groups: dict[str, list[tuple[_Sample, dict]]] = {name: [] for name in ("train", "validation", "test")}
    for sample, record in zip(samples, records):
        groups[record["split"]].append((sample, record))
    for split, values in groups.items():
        for shard_index, start in enumerate(range(0, len(values), max_samples)):
            subset = values[start : start + max_samples]
            shard_path = artifacts / "shards" / f"{split}-{shard_index:05d}.tar"
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(shard_path, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for sample, record in subset:
                    image_path, ext = _web_compatible_image_path(sample.content_path, sample.row.image_uri)
                    _add_tar_file(tar, f"{sample.key}.{ext}", image_path)
                    label_bytes = (record["label"] + "\n").encode("utf-8")
                    _add_tar_bytes(tar, f"{sample.key}.cls", label_bytes)
                    metadata = {
                        "image_uri": record["image_uri"],
                        "label": record["label"],
                        "sample_id": record["sample_id"],
                        "group_id": record["group_id"],
                        "sha256": record["content_sha256"],
                        "split": split,
                    }
                    _add_tar_bytes(tar, f"{sample.key}.json", _json_bytes(metadata))
    shutil.copyfile(source_path, artifacts / "source-snapshot.json")


def _add_tar_bytes(tar: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mtime = 0  # byte-identical rebuilds
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(content))


def _add_tar_file(tar: tarfile.TarFile, name: str, path: Path) -> None:
    info = tarfile.TarInfo(name)
    info.size = path.stat().st_size
    info.mtime = 0
    info.mode = 0o644
    with path.open("rb") as source:
        tar.addfile(info, source)


def _copy_image_to_path(uri: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if uri.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - cloud extra
            raise RuntimeError("Reading gs:// images requires the cloud extra") from exc
        parsed = urlparse(uri)
        storage.Client().bucket(parsed.netloc).blob(parsed.path.lstrip("/")).download_to_filename(str(destination))
    elif uri.startswith("file://"):
        shutil.copyfile(Path(urlparse(uri).path), destination)
    elif uri.startswith(("http://", "https://")):
        raise ValueError("HTTP image locations are not supported; stage them to local storage or GCS")
    else:
        shutil.copyfile(Path(uri).expanduser(), destination)
    try:
        from PIL import Image

        with Image.open(destination) as image:
            image.verify()
    except ImportError as exc:  # pragma: no cover - data extra
        raise RuntimeError("Building WebDataset shards requires Pillow") from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise ValueError(f"Image at {uri!r} is invalid or unsupported: {exc}") from exc
    return _sha256_file(destination)


def _publish(root_uri: str, artifacts: Path, content_digest: str) -> None:
    if root_uri.startswith("gs://"):
        _publish_gcs(root_uri, artifacts, content_digest)
    else:
        _publish_local(Path(root_uri).expanduser(), artifacts, content_digest)


def _publish_local(root: Path, artifacts: Path, content_digest: str) -> None:
    commit_path = root / "_COMMIT.json"
    if commit_path.exists():
        existing = json.loads(commit_path.read_text(encoding="utf-8"))
        if existing.get("sha256") == content_digest and _local_artifacts_match(root, artifacts):
            return
        raise FileExistsError(f"Refusing to overwrite immutable dataset {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    try:
        shutil.copytree(artifacts, temp, dirs_exist_ok=True)
        if root.exists():
            raise FileExistsError(f"Dataset target exists without a commit marker: {root}")
        try:
            os.replace(temp, root)
        except OSError:
            # Concurrent retries of the same content may race at the final rename.
            if commit_path.exists():
                existing = json.loads(commit_path.read_text(encoding="utf-8"))
                if existing.get("sha256") == content_digest and _local_artifacts_match(root, artifacts):
                    return
            raise
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _publish_gcs(root_uri: str, artifacts: Path, content_digest: str) -> None:
    try:
        from google.api_core.exceptions import PreconditionFailed
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - cloud extra
        raise RuntimeError("Publishing to GCS requires the cloud extra") from exc
    parsed = urlparse(root_uri)
    client = storage.Client()
    bucket = client.bucket(parsed.netloc)
    prefix = parsed.path.strip("/")
    marker_name = f"{prefix}/_COMMIT.json"
    marker = bucket.blob(marker_name)
    if marker.exists(client=client):
        existing = json.loads(marker.download_as_bytes().decode("utf-8"))
        if existing.get("sha256") == content_digest and _gcs_artifacts_match(bucket, prefix, artifacts, client):
            return
        raise FileExistsError(f"Refusing to overwrite immutable GCS dataset {root_uri}")
    try:
        for artifact_path in sorted(path for path in artifacts.rglob("*") if path.is_file()):
            name = artifact_path.relative_to(artifacts).as_posix()
            if name == "_COMMIT.json":
                continue
            blob_name = f"{prefix}/{name}"
            blob = bucket.blob(blob_name)
            blob.metadata = {"sha256": _sha256_file(artifact_path)}
            try:
                blob.upload_from_filename(str(artifact_path), if_generation_match=0, content_type=_content_type(name))
            except PreconditionFailed:
                # Resume a previously interrupted publication only when each
                # already-created immutable object has exactly the expected bytes.
                if (
                    not blob.exists(client=client)
                    or (blob.metadata or {}).get("sha256") != _sha256_file(artifact_path)
                ):
                    raise FileExistsError(f"Immutable GCS object already exists with different content: gs://{parsed.netloc}/{blob_name}")
        commit_path = artifacts / "_COMMIT.json"
        marker.metadata = {"sha256": _sha256_file(commit_path)}
        marker.upload_from_filename(str(commit_path), if_generation_match=0, content_type="application/json")
    except Exception as exc:
        if isinstance(exc, PreconditionFailed):
            # Another identical build may have won; only accept it after digest check.
            if marker.exists(client=client):
                existing = json.loads(marker.download_as_bytes().decode("utf-8"))
                if existing.get("sha256") == content_digest and _gcs_artifacts_match(bucket, prefix, artifacts, client):
                    return
        raise RuntimeError(f"GCS dataset publication failed; no commit marker was written: {exc}") from exc


def _web_compatible_image_path(path: Path, uri: str) -> tuple[Path, str]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(image.format)
            if extension:
                return path, extension
            converted = path.with_suffix(".converted.png")
            image.convert("RGB").save(converted, format="PNG", optimize=False)
            return converted, "png"
    except ImportError:  # pragma: no cover
        extension = Path(urlparse(uri).path).suffix.lstrip(".").lower()
        if extension not in {"jpg", "jpeg", "png", "webp"}:
            raise RuntimeError("Pillow is required to convert images to WebDataset supported formats")
        return path, "jpg" if extension == "jpeg" else extension


def _local_artifacts_match(root: Path, artifacts: Path) -> bool:
    expected_paths = [path for path in artifacts.rglob("*") if path.is_file()]
    expected = {path.relative_to(artifacts).as_posix() for path in expected_paths}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        return False
    return all(_sha256_file(root / name) == _sha256_file(artifacts / name) for name in expected)


def _gcs_artifacts_match(bucket: object, prefix: str, artifacts: Path, client: object) -> bool:
    for artifact_path in (path for path in artifacts.rglob("*") if path.is_file()):
        name = artifact_path.relative_to(artifacts).as_posix()
        blob = bucket.blob(f"{prefix}/{name}")
        if not blob.exists(client=client):
            return False
        blob.reload(client=client)
        if (blob.metadata or {}).get("sha256") != _sha256_file(artifact_path):
            return False
    return True


def _content_type(name: str) -> str:
    if name.endswith(".tar"):
        return "application/x-tar"
    if name.endswith(".json") or name.endswith(".jsonl"):
        return "application/json"
    return "application/octet-stream"


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _write_json_file(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(records: list[dict]) -> dict[str, int]:
    return {name: sum(record["split"] == name for record in records) for name in ("train", "validation", "test")}


def _join_uri(root: str, child: str) -> str:
    return root.rstrip("/") + "/" + child.lstrip("/")


def _check_object(spec: DatasetSpec, object_spec: ObjectSpec) -> None:
    if spec.object_slug != object_spec.slug:
        raise ValueError(f"Dataset object {spec.object_slug!r} does not match {object_spec.slug!r}")


class _UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def get_dataset_preview(spec: DatasetSpec, object_spec: ObjectSpec, **kwargs: object) -> PreviewResult:
    """Convenience alias kept for callers that prefer an explicit preview name."""
    return preview_dataset(spec, object_spec, **kwargs)
