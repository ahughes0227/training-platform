"""Built-in workflow presets selected by ID and overridden by validated patches."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from defect_platform.contracts import Preset

BUILTIN_PRESET_IDS = (
    "baseline_classifier",
    "fine_tune_backbone",
    "low_cost_gpu",
    "production_ray",
)

_PARAMETERS: dict[str, dict[str, Any]] = {
    "baseline_classifier": {"unfreeze_last_n": 0, "optimizer": "adamw", "loss": "cross_entropy"},
    "fine_tune_backbone": {"unfreeze_last_n": 1, "optimizer": "adamw", "loss": "cross_entropy"},
    "low_cost_gpu": {"max_run_hours": 1, "accelerator_count": 1, "retry_limit": 0},
    "production_ray": {"serving_profile": "ray", "retry_limit": 2, "logging_profile": "production"},
}


def builtin_presets() -> tuple[Preset, ...]:
    return tuple(
        Preset(
            preset_id=name,
            version="1",
            display_name=name.replace("_", " ").title(),
            parameters=deepcopy(_PARAMETERS[name]),
        )
        for name in BUILTIN_PRESET_IDS
    )


def resolve_preset(preset_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if preset_id not in _PARAMETERS:
        raise ValueError(f"unknown workflow preset: {preset_id}")
    values = deepcopy(_PARAMETERS[preset_id])
    for key, value in (overrides or {}).items():
        if key not in values:
            raise ValueError(f"override is not permitted for preset {preset_id}: {key}")
        values[key] = deepcopy(value)
    return values
