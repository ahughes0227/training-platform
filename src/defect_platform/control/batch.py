"""Deterministic experiment matrix planning for the control plane."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any

from defect_platform.contracts import ExperimentConfig, VertexJobConfig


def expand_matrix(matrix: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    """Expand a mapping of dimensions into stable Cartesian products.

    Dimension names are sorted, as are no values (their supplied order is
    meaningful and preserved). This makes plans reproducible across clients.
    """
    matrix = matrix or {}
    if not isinstance(matrix, dict):
        raise ValueError("matrix must be an object mapping dimensions to arrays")
    keys = sorted(matrix)
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("matrix dimensions must be non-empty strings")
    for key in keys:
        values = matrix[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"matrix dimension {key!r} must contain at least one value")
    if not keys:
        return [{}]
    return [dict(zip(keys, values)) for values in itertools.product(*(matrix[key] for key in keys))]


def apply_overrides(
    experiment: ExperimentConfig, job: VertexJobConfig, overrides: dict[str, Any]
) -> tuple[ExperimentConfig, VertexJobConfig]:
    """Apply ``experiment.`` and ``job.`` dimensions with strict field checks."""
    exp_updates: dict[str, Any] = {}
    job_updates: dict[str, Any] = {}
    for name, value in overrides.items():
        if name.startswith("experiment."):
            field = name.removeprefix("experiment.")
            if field not in ExperimentConfig.model_fields:
                raise ValueError(f"unknown experiment matrix field: {field}")
            exp_updates[field] = value
        elif name.startswith("job."):
            field = name.removeprefix("job.")
            if field not in VertexJobConfig.model_fields:
                raise ValueError(f"unknown job matrix field: {field}")
            job_updates[field] = value
        elif name in ExperimentConfig.model_fields:
            exp_updates[name] = value
        else:
            raise ValueError(f"matrix field must name an experiment or job field: {name}")
    # ``model_copy(update=...)`` intentionally skips validation in Pydantic;
    # re-parse each planned variant before it can be submitted.
    return (
        ExperimentConfig.model_validate({**experiment.model_dump(), **exp_updates}),
        VertexJobConfig.model_validate({**job.model_dump(), **job_updates}),
    )


def plan_matrix(
    experiment: ExperimentConfig,
    job: VertexJobConfig,
    *,
    matrix: dict[str, list[Any]] | None,
    idempotency_key: str,
) -> list[dict[str, Any]]:
    plans = []
    for index, overrides in enumerate(expand_matrix(matrix)):
        planned_experiment, planned_job = apply_overrides(experiment, job, overrides)
        encoded = json.dumps(overrides, sort_keys=True, separators=(",", ":"), default=str)
        suffix = hashlib.sha256(encoded.encode()).hexdigest()[:16]
        key = idempotency_key if not overrides else f"{idempotency_key}:{suffix}"
        plans.append(
            {
                "index": index,
                "idempotency_key": key,
                "overrides": overrides,
                "experiment": planned_experiment,
                "job": planned_job,
            }
        )
    return plans


# Explicit aliases make the planning API discoverable to CLI and agent callers.
expand_experiment_matrix = expand_matrix
plan_experiment_batch = plan_matrix
