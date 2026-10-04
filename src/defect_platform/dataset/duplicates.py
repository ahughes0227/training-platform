"""Exact and perceptual duplicate detection for leakage-aware splits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    left: str
    right: str
    kind: str
    distance: int | None = None


@dataclass(frozen=True, slots=True)
class DuplicateReport:
    exact_pairs: tuple[DuplicatePair, ...]
    near_pairs: tuple[DuplicatePair, ...]
    errors: dict[str, str]

    @property
    def pairs(self) -> tuple[DuplicatePair, ...]:
        return self.exact_pairs + self.near_pairs


def find_duplicates(
    images: Iterable[tuple[str, bytes | Path]], *, max_hamming_distance: int = 6
) -> DuplicateReport:
    """Compare image bytes exactly, then compare 64-bit dHashes for near copies."""
    if max_hamming_distance < 0 or max_hamming_distance > 64:
        raise ValueError("max_hamming_distance must be between 0 and 64")
    exact: list[DuplicatePair] = []
    near: list[DuplicatePair] = []
    errors: dict[str, str] = {}
    by_content_hash: dict[str, str] = {}
    fingerprint_index = _BKTree()
    for uri, content in images:
        content_bytes = content.read_bytes() if isinstance(content, Path) else content
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        prior = by_content_hash.get(content_hash)
        if prior:
            exact.append(DuplicatePair(prior, uri, "exact"))
            continue
        by_content_hash[content_hash] = uri
        if max_hamming_distance == 0:
            continue  # Explicit exact-only mode; perceptual collisions are excluded.
        try:
            fingerprint, color_mean = _fingerprint(content_bytes)
        except Exception as exc:
            errors[uri] = f"could not decode image for near-duplicate check: {exc}"
            continue
        for other_uri, other_hash, other_mean in fingerprint_index.search(
            fingerprint, max_hamming_distance
        ):
            distance = (fingerprint ^ other_hash).bit_count()
            color_distance = (
                sum((left - right) ** 2 for left, right in zip(color_mean, other_mean)) ** 0.5
            )
            if distance <= max_hamming_distance and color_distance <= 80:
                near.append(DuplicatePair(other_uri, uri, "near", distance))
        fingerprint_index.add(fingerprint, (uri, fingerprint, color_mean))
    return DuplicateReport(tuple(exact), tuple(near), errors)


def _dhash(content: bytes) -> int:
    return _fingerprint(content)[0]


def _fingerprint(content: bytes) -> tuple[int, tuple[float, float, float]]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional data extra
        raise RuntimeError("Perceptual duplicate detection requires Pillow") from exc
    import io

    with Image.open(io.BytesIO(content)) as image:
        gray = image.convert("L").resize((9, 8))
        pixels = gray.tobytes()
        rgb = image.convert("RGB").resize((8, 8)).tobytes()
        color_mean = (
            sum(rgb[0::3]) / 64,
            sum(rgb[1::3]) / 64,
            sum(rgb[2::3]) / 64,
        )
    result = 0
    for row in range(8):
        for col in range(8):
            result = (result << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return result, color_mean


class _BKTree:
    """Metric index for Hamming-distance searches over 64-bit fingerprints."""

    def __init__(self) -> None:
        self.root: tuple[int, list[tuple[str, int, tuple[float, float, float]]], dict] | None = None

    def add(self, value: int, item: tuple[str, int, tuple[float, float, float]]) -> None:
        if self.root is None:
            self.root = (value, [item], {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(item)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (value, [item], {})
                return
            node = child

    def search(self, query: int, radius: int) -> list[tuple[str, int, tuple[float, float, float]]]:
        if self.root is None:
            return []
        matches = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (query ^ node[0]).bit_count()
            if distance <= radius:
                matches.extend(node[1])
            low, high = distance - radius, distance + radius
            pending.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return matches
