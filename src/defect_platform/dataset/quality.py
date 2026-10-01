"""Cleanlab-backed, fail-closed dataset quality checks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from defect_platform.contracts import CleanlabSpec, ObjectSpec


@dataclass(frozen=True, slots=True)
class QualityResult:
    report: dict[str, Any]
    report_bytes: bytes


def analyze_cleanlab(spec: CleanlabSpec, object_spec: ObjectSpec, samples: list[Any]) -> QualityResult:
    """Validate and analyze a prediction envelope against staged samples."""
    try:
        envelope = json.loads(_read_uri(spec.predictions_uri))
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Cannot read Cleanlab predictions {spec.predictions_uri!r}: {exc}") from exc
    if not isinstance(envelope, dict) or envelope.get("format_version") != 1:
        raise ValueError("Cleanlab predictions must be a format_version=1 JSON object")
    if envelope.get("object_slug") != object_spec.slug:
        raise ValueError("Cleanlab predictions object_slug does not match the dataset object")
    if envelope.get("classes") != object_spec.classes:
        raise ValueError("Cleanlab prediction classes must exactly match object class order")
    if envelope.get("out_of_sample") is not True:
        raise ValueError("Cleanlab predictions must declare out_of_sample=true")
    if envelope.get("key_field") != spec.key_field:
        raise ValueError("Cleanlab prediction key_field does not match dataset configuration")
    predictions = envelope.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Cleanlab predictions must contain a predictions list")  # noqa: TRY004
    expected_keys = [getattr(sample.row, spec.key_field) for sample in samples]
    if any(key is None for key in expected_keys):
        raise ValueError(f"Cleanlab key_field {spec.key_field!r} is missing from a dataset sample")
    expected = set(expected_keys)
    by_key: dict[str, list[float]] = {}
    for item in predictions:
        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
            raise ValueError("Each Cleanlab prediction must contain a string key")  # noqa: TRY004
        key = item["key"]
        if key in by_key:
            raise ValueError(f"Duplicate Cleanlab prediction key: {key!r}")
        probabilities = item.get("probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != len(object_spec.classes):
            raise ValueError(f"Cleanlab probabilities for {key!r} must contain one value per class")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > 1
               for value in probabilities):
            raise ValueError(f"Cleanlab probabilities for {key!r} must be finite values in [0, 1]")
        if abs(sum(probabilities) - 1.0) > 1e-5:
            raise ValueError(f"Cleanlab probabilities for {key!r} must sum to 1")
        by_key[key] = [float(value) for value in probabilities]
    missing = expected - set(by_key)
    extra = set(by_key) - expected
    if missing or extra:
        raise ValueError(f"Cleanlab prediction keys do not match dataset (missing={sorted(missing)!r}, extra={sorted(extra)!r})")
    try:
        import numpy as np
        from cleanlab.filter import find_label_issues
    except ImportError as exc:  # pragma: no cover - exercised in clean environments
        raise RuntimeError("Cleanlab checks require the data-quality extra") from exc
    labels = np.asarray([object_spec.classes.index(sample.label) for sample in samples], dtype=int)
    pred_probs = np.asarray([by_key[key] for key in expected_keys], dtype=float)
    # Dataset builds are bounded, reproducible jobs; avoid Cleanlab spawning an
    # unbounded process pool on a shared builder host.
    ranked = find_label_issues(labels, pred_probs, return_indices_ranked_by="self_confidence", n_jobs=1)
    issue_indices = [int(index) for index in ranked]
    issues = []
    for index in issue_indices:
        probabilities = pred_probs[index]
        label_index = int(labels[index])
        predicted_index = int(probabilities.argmax())
        issues.append({
            "key": expected_keys[index],
            "image_uri": samples[index].row.image_uri,
            "sample_id": samples[index].row.sample_id,
            "label": samples[index].label,
            "predicted_label": object_spec.classes[predicted_index],
            "label_probability": float(probabilities[label_index]),
            "issue_score": float(1.0 - probabilities[label_index]),
        })
    report = {
        "format_version": 1,
        "object_slug": object_spec.slug,
        "classes": object_spec.classes,
        "key_field": spec.key_field,
        "predictions_uri": spec.predictions_uri,
        "sample_count": len(samples),
        "issue_count": len(issues),
        "issue_fraction": len(issues) / len(samples) if samples else 0.0,
        "max_issue_fraction": spec.max_issue_fraction,
        "issues": issues,
    }
    report_bytes = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return QualityResult(report, report_bytes)


def _read_uri(uri: str) -> bytes:
    if uri.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - cloud extra
            raise RuntimeError("Reading gs:// Cleanlab predictions requires the cloud extra") from exc
        parsed = urlparse(uri)
        return storage.Client().bucket(parsed.netloc).blob(parsed.path.lstrip("/")).download_as_bytes()
    return Path(uri).expanduser().read_bytes()
