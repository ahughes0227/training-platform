"""Deterministic model routing for bounded agent operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AgentOperation(StrEnum):
    EXTRACTION = "extraction"
    FIELD_MAPPING = "field_mapping"
    PATCH = "patch"
    AMBIGUITY = "ambiguity"
    ARCHITECTURE = "architecture"
    REVIEW = "review"


@dataclass(frozen=True)
class ModelRoutingPolicy:
    """Names are configuration only; the control plane still owns the action."""

    small_model: str
    large_model: str

    def model_for(self, operation: AgentOperation | str) -> str:
        operation = AgentOperation(operation)
        if operation in {
            AgentOperation.AMBIGUITY,
            AgentOperation.ARCHITECTURE,
            AgentOperation.REVIEW,
        }:
            return self.large_model
        return self.small_model


def route_model(operation: AgentOperation | str, *, policy: ModelRoutingPolicy) -> str:
    """Return the model selected by the explicit operation policy."""
    return policy.model_for(operation)
