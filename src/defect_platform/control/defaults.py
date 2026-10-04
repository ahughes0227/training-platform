"""Deterministic, reviewable defaults for agent-generated experiment plans."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_POLICY_ID = "platform-defaults@1"

# Semantic defaults already used by the typed contracts. Deployment-specific
# resources remain operator-profile values and are never selected by the model.
SEMANTIC_DEFAULTS: dict[str, Any] = {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "optimizer": "adamw",
    "loss": "cross_entropy",
    "focal_gamma": 2.0,
    "horizontal_flip_probability": 0.5,
    "seed": 42,
    "split": {"train": 0.7, "validation": 0.15, "test": 0.15, "seed": 42},
}


def resolve_defaults(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a detached effective default map with explicit overrides applied."""
    values = deepcopy(SEMANTIC_DEFAULTS)
    for key, value in (overrides or {}).items():
        if key not in values:
            raise ValueError(f"unsupported default override: {key}")
        values[key] = deepcopy(value)
    return values
