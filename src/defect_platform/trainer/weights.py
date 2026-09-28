"""Deterministic checksums for local model weight files and directories."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def artifact_sha256(path: str | Path) -> str:
    """Hash a file's bytes or a directory manifest in stable relative-path order.

    Directory digest is SHA256 over sorted ``relative_path NUL file_sha256 LF``
    entries. This makes a model repository (config plus checkpoint shards) a
    single verifiable artifact without depending on archive timestamps.
    """
    path = Path(path)
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(f"weight artifact does not exist: {path}")
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"weight artifact directory is empty: {path}")
    manifest = hashlib.sha256()
    for file_path in files:
        relative = file_path.relative_to(path).as_posix()
        file_digest = artifact_sha256(file_path)
        manifest.update(relative.encode("utf-8")); manifest.update(b"\0")
        manifest.update(file_digest.encode("ascii")); manifest.update(b"\n")
    return manifest.hexdigest()


def verify_artifact_sha256(path: str | Path, expected: str) -> str:
    """Verify a pinned SHA256 and return the actual digest."""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected or ""):
        raise ValueError("expected weight SHA256 must be exactly 64 hexadecimal characters")
    actual = artifact_sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"weight artifact checksum mismatch: expected {expected}, got {actual}")
    return actual


def write_bundle_integrity(directory: str | Path) -> None:
    """Write checksums for every portable model bundle member."""
    supplied_root = Path(directory)
    if supplied_root.is_symlink():
        raise ValueError("model bundle directory cannot be a symbolic link")
    root = supplied_root.resolve()
    entries = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("model bundle cannot contain symbolic links")
        if path.is_file() and path.relative_to(root).as_posix() != "integrity.json":
            rel = path.relative_to(root).as_posix()
            entries[rel] = artifact_sha256(path)
    if not entries:
        raise ValueError("model bundle is empty")
    payload = {"schema_version": 1, "files": entries}
    (root / "integrity.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def verify_bundle_integrity(directory: str | Path, expected_sha256: str | None = None) -> None:
    """Verify all bundle files and reject traversal, symlinks, and extras."""
    supplied_root = Path(directory)
    if supplied_root.is_symlink():
        raise ValueError("model bundle directory cannot be a symbolic link")
    root = supplied_root.resolve()
    manifest_path = root / "integrity.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"model bundle integrity manifest is missing or invalid: {exc}") from exc
    if manifest_path.is_symlink():
        raise ValueError("model bundle integrity manifest cannot be a symbolic link")
    if expected_sha256 is not None and hashlib.sha256(manifest_bytes).hexdigest() != expected_sha256.lower():
        raise ValueError("model bundle integrity manifest checksum mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model bundle integrity manifest is invalid: {exc}") from exc
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("files"), dict)):
        raise ValueError("invalid model bundle integrity manifest")
    expected = manifest["files"]
    actual_paths = set()
    for rel, digest in expected.items():
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts or not rel:
            raise ValueError("invalid path in model bundle integrity manifest")
        path = root / rel_path
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"unsafe or missing model bundle member: {rel}")
        actual_paths.add(rel_path.as_posix())
        verify_artifact_sha256(path, digest)
    present = {path.relative_to(root).as_posix() for path in root.rglob("*")
               if path.is_file() and path.relative_to(root).as_posix() != "integrity.json"}
    if present != actual_paths:
        raise ValueError("model bundle files do not match integrity manifest")
