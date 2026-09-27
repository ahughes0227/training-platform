"""Deterministic checksums for local model weight files and directories."""

from __future__ import annotations

import hashlib
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
