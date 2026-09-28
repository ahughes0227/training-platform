"""Canonical label mapping and reviewable exception reports."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable

from defect_platform.contracts import DatasetSpec, ObjectSpec
from defect_platform.semantics import ClassCatalog, catalog_for_object, validate_label_mapping

from .sources import LabelRow, read_label_source


@dataclass(frozen=True, slots=True)
class LabelException:
    source: str
    row_number: int | None
    image_uri: str
    raw_label: str
    normalized_label: str
    reason: str
    suggested_class: str | None = None
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreviewResult:
    row_count: int
    mapped_count: int
    exceptions: tuple[LabelException, ...]
    class_counts: dict[str, int]

    @property
    def ready(self) -> bool:
        return not self.exceptions

    def to_dict(self) -> dict:
        return {
            "row_count": self.row_count,
            "mapped_count": self.mapped_count,
            "ready": self.ready,
            "class_counts": self.class_counts,
            "exceptions": [asdict(item) for item in self.exceptions],
        }


def normalize_label(value: str) -> str:
    """Normalize spelling for lookup while retaining original source labels."""
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def map_label(raw_label: str, classes: Iterable[str], mapping: dict[str, str],
              catalog: ClassCatalog | None = None) -> tuple[str | None, str]:
    normalized = normalize_label(raw_label)
    class_list = list(classes)
    # Catalog aliases resolve to stable catalog positions, while the artifact
    # labels remain the catalog's ordered display labels.
    canonical_by_norm = {normalize_label(class_name): class_name for class_name in class_list}
    mapping_by_norm = {normalize_label(alias): target for alias, target in mapping.items()}
    target = mapping_by_norm.get(normalized)
    if target is not None:
        if catalog is not None:
            try:
                return catalog.decode(catalog.encode(target)).label, normalized
            except ValueError:
                return None, normalized
        return canonical_by_norm.get(normalize_label(target)), normalized
    if catalog is not None:
        try:
            return catalog.decode(catalog.encode(raw_label)).label, normalized
        except ValueError:
            return None, normalized
    return canonical_by_norm.get(normalized), normalized


def preview_dataset(
    spec: DatasetSpec,
    object_spec: ObjectSpec,
    *,
    rows: list[LabelRow] | None = None,
    bq_client: object | None = None,
) -> PreviewResult:
    """Resolve configured labels and return exceptions suitable for human review."""
    _check_object(spec, object_spec)
    catalog = catalog_for_object(object_spec)
    if rows is None:
        rows = [row for source in spec.sources for row in read_label_source(source, bq_client=bq_client)]
    exceptions: list[LabelException] = []
    counts = {name: 0 for name in object_spec.classes}
    mapped_count = 0
    seen_sample_ids: dict[str, LabelRow] = {}
    canonical_rows: dict[str, set[str]] = {}
    for row in rows:
        canonical, normalized = map_label(row.raw_label, object_spec.classes, spec.label_mapping, catalog)
        if canonical is None:
            candidates = _suggestions(normalized, object_spec.classes)
            exceptions.append(
                LabelException(
                    row.source,
                    row.row_number,
                    row.image_uri,
                    row.raw_label,
                    normalized,
                    "unmapped_label",
                    candidates[0] if candidates else None,
                    tuple(candidates),
                )
            )
            continue
        mapped_count += 1
        counts[canonical] += 1
        canonical_rows.setdefault(row.image_uri, set()).add(canonical)
        if row.sample_id:
            prior = seen_sample_ids.get(row.sample_id)
            if prior and prior.image_uri != row.image_uri:
                exceptions.append(
                    LabelException(
                        row.source,
                        row.row_number,
                        row.image_uri,
                        row.raw_label,
                        normalized,
                        "duplicate_sample_id",
                        candidates=(prior.image_uri,),
                    )
                )
            else:
                seen_sample_ids[row.sample_id] = row
    for uri, labels in canonical_rows.items():
        if len(labels) > 1:
            exceptions.append(
                LabelException(
                    "combined sources",
                    None,
                    uri,
                    ", ".join(sorted(labels)),
                    normalize_label(", ".join(sorted(labels))),
                    "conflicting_labels_for_same_image",
                    candidates=tuple(sorted(labels)),
                )
            )
    for class_name, count in counts.items():
        if count == 0:
            exceptions.append(
                LabelException(
                    "combined sources",
                    None,
                    "",
                    class_name,
                    normalize_label(class_name),
                    "class_has_no_examples",
                    suggested_class=class_name,
                )
            )
    return PreviewResult(len(rows), mapped_count, tuple(exceptions), counts)


def _suggestions(normalized: str, classes: list[str]) -> list[str]:
    # Suggestions are hints only. They never silently alter the canonical label.
    try:
        from difflib import get_close_matches

        normalized_classes = [normalize_label(value) for value in classes]
        matches = get_close_matches(normalized, normalized_classes, n=3, cutoff=0.55)
        class_names = {normalize_label(value): value for value in classes}
        return [class_names[match] for match in matches]
    except Exception:  # pragma: no cover - stdlib implementation is reliable
        return []


def _check_object(spec: DatasetSpec, object_spec: ObjectSpec) -> None:
    if spec.object_slug != object_spec.slug:
        raise ValueError(f"Dataset object {spec.object_slug!r} does not match {object_spec.slug!r}")
    catalog = catalog_for_object(object_spec)
    validate_label_mapping(catalog, spec.label_mapping)
    invalid_targets = []
    for target in spec.label_mapping.values():
        try:
            catalog.encode(target)
        except ValueError:
            invalid_targets.append(target)
    if invalid_targets:
        raise ValueError(f"Label mapping targets unknown classes: {', '.join(sorted(set(invalid_targets)))}")
