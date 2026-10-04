#!/usr/bin/env python3
"""Export versioned JSON schemas from the authoritative Python module contracts."""

import argparse
import json
from pathlib import Path

from defect_platform.contracts import (
    CertifiedRuntime,
    ComponentCatalog,
    ComponentDefinition,
    DatasetSpec,
    DatasetVersion,
    ErrorRecord,
    ExperimentConfig,
    ExperimentMatrix,
    JsonPatchOperation,
    ModelRelease,
    ObjectSpec,
    OperatorProfile,
    Preset,
    RunRecord,
    SchemaRegistryEntry,
    StatusRecord,
    TelemetryEvent,
    VertexJobConfig,
)
from defect_platform.infrastructure_contract import InfrastructureCapabilities
from defect_platform.semantics import ClassCatalog, SemanticManifest

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "class-catalog": ClassCatalog,
    "object": ObjectSpec,
    "component-catalog": ComponentCatalog,
    "component-definition": ComponentDefinition,
    "preset": Preset,
    "operator-profile": OperatorProfile,
    "experiment-matrix": ExperimentMatrix,
    "status": StatusRecord,
    "error": ErrorRecord,
    "telemetry": TelemetryEvent,
    "json-patch-operation": JsonPatchOperation,
    "schema-registry-entry": SchemaRegistryEntry,
    "dataset-spec": DatasetSpec,
    "dataset-version": DatasetVersion,
    "experiment": ExperimentConfig,
    "certified-runtime": CertifiedRuntime,
    "vertex-job": VertexJobConfig,
    "run": RunRecord,
    "model-release": ModelRelease,
    "semantic-manifest": SemanticManifest,
    "infrastructure-capabilities": InfrastructureCapabilities,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "schemas"
    if args.write:
        directory.mkdir(exist_ok=True)
    for name, model in MODELS.items():
        target = directory / f"{name}.schema.json"
        content = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        if args.write:
            target.write_text(content)
        elif not target.exists() or target.read_text() != content:
            raise SystemExit(f"Contract schema is stale: {target}")
    print(f"Validated {len(MODELS)} generated module schemas.")


if __name__ == "__main__":
    main()
