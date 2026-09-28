"""Fail-closed verification for dataset artifacts before paid jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from defect_platform.contracts import DatasetVersion
from defect_platform.semantics import SemanticManifest, parse_manifest


def verify_dataset_version(version: DatasetVersion, expected_classes: list[str]) -> None:
    """Verify contract, commit marker, provenance, manifest, and every shard.

    Raises ``ValueError`` for contract/content mismatches and ``RuntimeError`` for
    unavailable or unreadable artifacts. Local paths and ``gs://`` URIs are
    supported; cloud dependencies are loaded only when required.
    """
    _verify_dataset_version(version, expected_classes)


def _verify_dataset_version(
    version: DatasetVersion,
    expected_classes: list[str],
    *,
    semantic_bytes: bytes | None = None,
) -> SemanticManifest:
    _require(version.manifest_uri == _join(version.root_uri, "manifest.json"), "manifest_uri does not match dataset root")
    _require(
        version.source_snapshot_uri == _join(version.root_uri, "source-snapshot.json"),
        "source_snapshot_uri does not match dataset root",
    )
    _require(version.semantic_manifest_uri == _join(version.root_uri, "semantics.json"),
             "semantic_manifest_uri does not match dataset root")
    try:
        commit = json.loads(_read(_join(version.root_uri, "_COMMIT.json")))
        manifest_bytes = _read(version.manifest_uri)
        manifest = json.loads(manifest_bytes)
        snapshot_uri = version.source_snapshot_uri
        source_bytes = _read(snapshot_uri)
        snapshot = json.loads(source_bytes)
        manifest_jsonl_bytes = _read(_join(version.root_uri, "manifest.jsonl"))
        if semantic_bytes is None:
            semantic_bytes = _read(version.semantic_manifest_uri)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Dataset {version.version_id!r} has missing or unreadable metadata: {exc}") from exc
    manifest_jsonl_digest = hashlib.sha256(manifest_jsonl_bytes).hexdigest()

    _require(commit.get("version_id") == version.version_id, "_COMMIT.json version_id does not match DatasetVersion")
    _require(commit.get("sha256") == version.sha256, "_COMMIT.json sha256 does not match DatasetVersion")
    _require(manifest.get("version_id") == version.version_id, "manifest version_id does not match DatasetVersion")
    _require(manifest.get("object_slug") == version.object_slug, "manifest object_slug does not match DatasetVersion")
    _require(manifest.get("classes") == expected_classes, "manifest classes do not match expected classes")
    _require(snapshot.get("object_slug") == version.object_slug, "source snapshot object_slug does not match DatasetVersion")
    _require(snapshot.get("classes") == expected_classes, "source snapshot classes do not match expected classes")
    _require(manifest.get("sample_counts") == version.sample_counts, "manifest sample counts do not match DatasetVersion")
    _require(
        manifest.get("manifest_sha256") == manifest_jsonl_digest,
        "manifest.jsonl checksum does not match manifest",
    )
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    _require(manifest.get("source_snapshot_sha256") == source_digest, "source snapshot checksum does not match manifest")
    semantic_file_digest = hashlib.sha256(semantic_bytes).hexdigest()
    _require(manifest.get("semantic_manifest_file_sha256") == semantic_file_digest,
             "semantic manifest file checksum does not match manifest")
    try:
        semantic = parse_manifest(semantic_bytes, expected_sha256=version.semantic_sha256)
    except Exception as exc:
        raise ValueError(f"invalid semantic manifest: {exc}") from exc
    _require(manifest.get("semantic_manifest_sha256") == semantic.sha256,
             "dataset manifest semantic checksum does not match semantic manifest")
    _require(version.semantic_sha256 == semantic.sha256, "DatasetVersion semantic checksum does not match manifest")
    _require(semantic.kind == "dataset", "semantic manifest kind is not dataset")
    _require(semantic.dataset_version_id == version.version_id, "semantic manifest version does not match DatasetVersion")
    _require(semantic.object_slug == version.object_slug, "semantic manifest object does not match DatasetVersion")
    _require(semantic.catalog.labels == expected_classes, "semantic catalog classes do not match expected classes")
    _require(semantic.source_snapshot_sha256 == source_digest, "semantic source checksum does not match snapshot")
    _require(semantic.label_mapping == snapshot.get("label_mapping", {}), "semantic label mapping does not match source snapshot")
    _require(semantic.split_policy == snapshot.get("split", {}), "semantic split policy does not match source snapshot")
    _require(semantic.catalog.model_dump(mode="json") == snapshot.get("class_catalog"),
             "semantic catalog does not match source snapshot")
    records = [json.loads(line) for line in manifest_jsonl_bytes.splitlines() if line]
    split_names = {"train", "validation", "test"}
    record_counts = {name: 0 for name in split_names}
    for record in records:
        _require(record.get("split") in split_names, "manifest.jsonl contains an invalid split name")
        _require(record.get("label") in semantic.catalog.labels, "manifest.jsonl contains an unknown class label")
        record_counts[record["split"]] += 1
    _require(manifest.get("sample_count") == len(records), "manifest sample_count does not match manifest.jsonl")
    _require(manifest.get("sample_counts") == record_counts, "manifest sample counts do not match manifest.jsonl")
    _require(version.sample_counts == record_counts, "DatasetVersion sample counts do not match manifest.jsonl")
    _require(set(version.sample_counts) == split_names, "DatasetVersion sample counts must define train/validation/test")
    assignments = [{"key": record["key"], "class_id": semantic.catalog.decode(
        semantic.catalog.encode(record["label"])).class_id, "split": record["split"]} for record in records]
    assignments_digest = hashlib.sha256(_json_bytes(assignments)).hexdigest()
    _require(semantic.split_assignments_sha256 == assignments_digest,
             "semantic split assignment checksum does not match manifest")

    shard_hashes = manifest.get("shards")
    _require(isinstance(shard_hashes, dict), "manifest shards must be a path-to-checksum mapping")
    _require(set(version.shard_uris) == split_names, "DatasetVersion shard_uris must define train/validation/test")
    contract_uris = [uri for values in version.shard_uris.values() for uri in values]
    _require(len(contract_uris) == len(set(contract_uris)), "DatasetVersion repeats a shard URI")
    contract_names = {_relative_shard_name(version.root_uri, uri) for uri in contract_uris}
    _require(contract_names == set(shard_hashes), "DatasetVersion shard URIs do not match manifest shards")
    for split, uris in version.shard_uris.items():
        for uri in uris:
            name = _relative_shard_name(version.root_uri, uri)
            _require(Path(name).name.startswith(f"{split}-"),
                     f"DatasetVersion {split} shard URI points to a different split: {name}")
    for name, expected_sha in shard_hashes.items():
        _validate_shard_name(name)
        actual_sha = _hash_uri(_join(version.root_uri, name))
        _require(actual_sha == expected_sha, f"shard checksum mismatch: {name}")

    manifest_file_digest = hashlib.sha256(manifest_bytes).hexdigest()
    expected_content_digest = hashlib.sha256(
        _json_bytes(
            {
                "manifest": manifest_file_digest,
                "manifest_jsonl": manifest_jsonl_digest,
                "source": source_digest,
                "shards": shard_hashes,
                "semantics": semantic_file_digest,
            }
        )
    ).hexdigest()
    _require(expected_content_digest == version.sha256, "dataset content digest does not match DatasetVersion")
    _require(commit.get("sha256") == expected_content_digest, "_COMMIT.json content digest verification failed")
    return semantic


def load_dataset_semantics(version: DatasetVersion) -> SemanticManifest:
    """Return the dataset semantic manifest only after verifying its binding."""
    try:
        classes = json.loads(_read(version.manifest_uri)).get("classes")
    except Exception as exc:
        raise RuntimeError(f"could not read dataset classes for semantic verification: {exc}") from exc
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError("dataset manifest classes are invalid")
    semantic_bytes = _read(version.semantic_manifest_uri or "")
    return _verify_dataset_version(version, expected_classes=classes, semantic_bytes=semantic_bytes)


def _read(uri: str) -> bytes:
    if uri.startswith("gs://"):
        client, bucket, path = _gcs_parts(uri)
        blob = client.bucket(bucket).blob(path)
        try:
            return blob.download_as_bytes()
        except Exception as exc:
            raise RuntimeError(f"could not read {uri}: {exc}") from exc
    path = _local_path(uri)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"could not read {uri}: {exc}") from exc


def _hash_uri(uri: str) -> str:
    digest = hashlib.sha256()
    if uri.startswith("gs://"):
        client, bucket, path = _gcs_parts(uri)
        blob = client.bucket(bucket).blob(path)
        try:
            with blob.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
        except Exception as exc:
            raise RuntimeError(f"could not checksum {uri}: {exc}") from exc
        return digest.hexdigest()
    path = _local_path(uri)
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RuntimeError(f"could not checksum {uri}: {exc}") from exc
    return digest.hexdigest()


def _gcs_parts(uri: str):
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - cloud extra
        raise RuntimeError("Verifying gs:// datasets requires the cloud extra") from exc
    parsed = urlparse(uri)
    if not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid GCS artifact URI: {uri}")
    return storage.Client(), parsed.netloc, parsed.path.strip("/")


def _local_path(uri: str) -> Path:
    if uri.startswith("file://"):
        return Path(urlparse(uri).path)
    if "://" in uri:
        raise ValueError(f"Unsupported dataset artifact URI: {uri}")
    return Path(uri).expanduser()


def _relative_shard_name(root_uri: str, uri: str) -> str:
    prefix = root_uri.rstrip("/") + "/"
    _require(uri.startswith(prefix), f"shard URI is outside dataset root: {uri}")
    return uri[len(prefix) :]


def _validate_shard_name(name: str) -> None:
    path = Path(name)
    _require(not path.is_absolute() and ".." not in path.parts and name.startswith("shards/"), f"invalid shard path in manifest: {name}")


def _join(root: str, name: str) -> str:
    return root.rstrip("/") + "/" + name.lstrip("/")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
