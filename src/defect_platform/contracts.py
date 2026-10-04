"""Stable configuration and artifact contracts shared by all platform components."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .semantics import ClassCatalog, PreprocessingSpec, catalog_for_object


class StrictModel(BaseModel):
    # ``populate_by_name`` lets integrations use a wire alias (for example
    # ``from`` in JSON Patch) without changing the Python attribute name.
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, populate_by_name=True, allow_inf_nan=False
    )


class Reference(StrictModel):
    """Content-addressed reference to an authoritative platform record.

    References intentionally carry identity and integrity separately from the
    record they point at.  Consumers can therefore resolve a record and verify
    its bytes before using it, while local fixtures may omit the optional URI.
    """

    kind: str = Field(min_length=1, max_length=120)
    id: str = Field(min_length=1, max_length=300)
    version: str | None = Field(default=None, min_length=1, max_length=120)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    uri: str | None = Field(default=None, min_length=1)
    schema_version: int = Field(default=1, ge=1)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.id}:{self.version or '-'}"

    @property
    def fingerprint(self) -> str:
        return self.sha256

    @property
    def ref_id(self) -> str:
        return self.id


class ComponentRef(Reference):
    kind: Literal["component"] = "component"


class CatalogRef(Reference):
    kind: Literal["catalog"] = "catalog"


class PresetRef(Reference):
    kind: Literal["preset"] = "preset"


class OperatorProfileRef(Reference):
    kind: Literal["operator_profile"] = "operator_profile"


class ExperimentMatrixRef(Reference):
    kind: Literal["experiment_matrix"] = "experiment_matrix"


class StatusRef(Reference):
    kind: Literal["status"] = "status"


class ErrorRef(Reference):
    kind: Literal["error"] = "error"


class TelemetryRef(Reference):
    kind: Literal["telemetry"] = "telemetry"


class ComponentDefinition(StrictModel):
    """A catalog entry for a separately owned executable/service component."""

    component_id: str = Field(min_length=1, max_length=200)
    display_name: str = Field(min_length=1, max_length=300)
    owner: str = Field(min_length=1, max_length=200)
    kind: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=120)
    reference: ComponentRef
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComponentCatalog(StrictModel):
    """Versioned, immutable index of components available to the control plane."""

    catalog_id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=120)
    components: tuple[ComponentDefinition, ...] = ()
    catalog_ref: CatalogRef | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def unique_components(self) -> ComponentCatalog:
        ids = [component.component_id for component in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("component IDs must be unique")
        if self.catalog_ref and self.catalog_ref.id != self.catalog_id:
            raise ValueError("catalog reference ID must match catalog_id")
        if self.catalog_ref and self.catalog_ref.version not in (None, self.version):
            raise ValueError("catalog reference version must match catalog version")
        return self

    @property
    def sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"catalog_ref"})
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


class Preset(StrictModel):
    """Named parameters assembled from immutable component references."""

    preset_id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=120)
    display_name: str = Field(min_length=1, max_length=300)
    components: tuple[ComponentRef, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)
    preset_ref: PresetRef | None = None

    @model_validator(mode="after")
    def ref_matches_identity(self) -> Preset:
        if self.preset_ref and self.preset_ref.id != self.preset_id:
            raise ValueError("preset reference ID must match preset_id")
        if self.preset_ref and self.preset_ref.version not in (None, self.version):
            raise ValueError("preset reference version must match preset version")
        return self


class OperatorProfile(StrictModel):
    """Operator-owned limits and authority used by admission and dispatch."""

    profile_id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=120)
    display_name: str = Field(min_length=1, max_length=300)
    owner: str = Field(min_length=1, max_length=200)
    capabilities: tuple[str, ...] = ()
    limits: dict[str, float | int | str | bool] = Field(default_factory=dict)
    component_refs: tuple[ComponentRef, ...] = ()
    profile_ref: OperatorProfileRef | None = None
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def valid_expiry_and_ref(self) -> OperatorProfile:
        if self.expires_at and self.expires_at.tzinfo is None:
            raise ValueError("operator profile expiry must include a timezone")
        if self.profile_ref and self.profile_ref.id != self.profile_id:
            raise ValueError("operator profile reference ID must match profile_id")
        if self.profile_ref and self.profile_ref.version not in (None, self.version):
            raise ValueError("operator profile reference version must match profile version")
        return self


class ExperimentVariant(StrictModel):
    variant_id: str = Field(min_length=1, max_length=200)
    parameters: dict[str, Any] = Field(default_factory=dict)
    preset_ref: PresetRef | None = None


class ExperimentMatrix(StrictModel):
    """Finite, explicit experiment variants with shared immutable references."""

    matrix_id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=120)
    object_slug: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    base_experiment: Reference | None = None
    variants: tuple[ExperimentVariant, ...] = Field(min_length=1)
    operator_profile: OperatorProfileRef | None = None
    matrix_ref: ExperimentMatrixRef | None = None

    @model_validator(mode="after")
    def unique_variants_and_refs(self) -> ExperimentMatrix:
        ids = [variant.variant_id for variant in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("experiment variant IDs must be unique")
        if self.matrix_ref and self.matrix_ref.id != self.matrix_id:
            raise ValueError("matrix reference ID must match matrix_id")
        if self.matrix_ref and self.matrix_ref.version not in (None, self.version):
            raise ValueError("matrix reference version must match matrix version")
        return self


class LifecycleState(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class StatusRecord(StrictModel):
    """Durable, truthful status projection for a referenced operation."""

    status_id: str = Field(min_length=1, max_length=200)
    state: LifecycleState
    revision: int = Field(default=1, ge=1)
    observed_at: datetime
    message: str = ""
    next_action: str | None = None
    references: tuple[Reference, ...] = ()
    error_ref: ErrorRef | None = None

    @model_validator(mode="after")
    def error_matches_state(self) -> StatusRecord:
        if self.state is LifecycleState.FAILED and self.error_ref is None:
            raise ValueError("failed status requires an error reference")
        if self.state is not LifecycleState.FAILED and self.error_ref is not None:
            raise ValueError("error reference is only valid for failed status")
        return self


class ErrorOwner(StrEnum):
    DATA = "data"
    TRAINER = "trainer"
    RUNTIME = "runtime"
    CONTROL = "control"
    INFRASTRUCTURE = "infrastructure"
    TRACKING = "tracking"
    SERVING = "serving"
    UNKNOWN = "unknown"


class ErrorRecord(StrictModel):
    """Failure taxonomy record preserving ownership and uncertainty."""

    error_id: str = Field(min_length=1, max_length=200)
    code: str = Field(min_length=1, max_length=120)
    owner: ErrorOwner
    message: str = Field(min_length=1, max_length=4000)
    stage: str = Field(min_length=1, max_length=120)
    retryable: bool = False
    uncertain: bool = False
    occurred_at: datetime
    command: str | None = None
    config_ref: Reference | None = None
    image_digest: str | None = None
    job_name: str | None = None
    log_ref: str | None = None
    error_ref: ErrorRef | None = None

    @model_validator(mode="after")
    def ref_matches_identity(self) -> ErrorRecord:
        if self.error_ref and self.error_ref.id != self.error_id:
            raise ValueError("error reference ID must match error_id")
        return self


class TelemetrySeverity(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TelemetryEvent(StrictModel):
    """Versioned structured event suitable for stdout, OTLP, or Loki adapters."""

    event_id: str = Field(min_length=1, max_length=200)
    schema_version: int = Field(default=1, ge=1)
    occurred_at: datetime
    severity: TelemetrySeverity = TelemetrySeverity.INFO
    component: str = Field(min_length=1, max_length=200)
    event: str = Field(min_length=1, max_length=200)
    message: str = ""
    context: dict[str, str] = Field(default_factory=dict)
    references: tuple[Reference, ...] = ()
    error: ErrorRecord | None = None
    telemetry_ref: TelemetryRef | None = None

    @model_validator(mode="after")
    def error_severity(self) -> TelemetryEvent:
        if self.error and self.severity not in {
            TelemetrySeverity.ERROR,
            TelemetrySeverity.CRITICAL,
        }:
            raise ValueError("telemetry errors require ERROR or CRITICAL severity")
        return self


class JsonPatchOperation(StrictModel):
    """One RFC 6902 operation; values remain JSON-compatible opaque data."""

    op: Literal["add", "remove", "replace", "move", "copy", "test"]
    path: str = Field(pattern=r"^(|(?:/(?:[^~/]|~[01])*))+$$")
    value: Any = None
    from_: str | None = Field(default=None, alias="from")

    @model_validator(mode="after")
    def required_members(self) -> JsonPatchOperation:
        if self.op in {"add", "replace", "test"} and "value" not in self.model_fields_set:
            raise ValueError(f"{self.op} operation requires value")
        if self.op in {"move", "copy"} and not self.from_:
            raise ValueError(f"{self.op} operation requires from")
        if self.op in {"add", "remove", "replace", "test"} and self.from_ is not None:
            raise ValueError(f"{self.op} operation cannot contain from")
        return self


class JsonPatchError(ValueError):
    """Raised when a patch is malformed or cannot be applied."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _decode_pointer(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise JsonPatchError("JSON Pointer must start with '/'")
    parts = []
    for token in pointer[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise JsonPatchError("invalid JSON Pointer escape")
        parts.append(token.replace("~1", "/").replace("~0", "~"))
    return parts


def validate_json_patch(
    patch: Sequence[Mapping[str, Any] | JsonPatchOperation],
) -> list[JsonPatchOperation]:
    """Parse and validate a patch without applying it."""
    try:
        operations = [
            item
            if isinstance(item, JsonPatchOperation)
            else JsonPatchOperation.model_validate(item)
            for item in patch
        ]
    except Exception as exc:
        raise JsonPatchError(str(exc)) from exc
    for operation in operations:
        _decode_pointer(operation.path)
        if operation.from_ is not None:
            _decode_pointer(operation.from_)
    return operations


def _resolve_parent(document: Any, tokens: list[str], *, create: bool = False) -> tuple[Any, str]:
    if not tokens:
        raise JsonPatchError("root path has no parent")
    current = document
    for token in tokens[:-1]:
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError) as exc:
                raise JsonPatchError("array path does not exist") from exc
        elif isinstance(current, dict) and token in current:
            current = current[token]
        else:
            raise JsonPatchError("object path does not exist")
    return current, tokens[-1]


def _get_value(document: Any, tokens: list[str]) -> Any:
    current = document
    for token in tokens:
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError) as exc:
                raise JsonPatchError("array path does not exist") from exc
        elif isinstance(current, dict) and token in current:
            current = current[token]
        else:
            raise JsonPatchError("object path does not exist")
    return current


def apply_json_patch(document: Any, patch: Sequence[Mapping[str, Any] | JsonPatchOperation]) -> Any:
    """Apply RFC 6902 operations atomically to a deep copy of ``document``."""
    operations = validate_json_patch(patch)
    result = deepcopy(document)
    for operation in operations:
        tokens = _decode_pointer(operation.path)
        if operation.op == "test":
            if _get_value(result, tokens) != operation.value:
                raise JsonPatchError(f"test operation failed at {operation.path!r}")
            continue
        if operation.op in {"move", "copy"}:
            source_tokens = _decode_pointer(operation.from_ or "")
            value = deepcopy(_get_value(result, source_tokens))
            if operation.op == "move":
                source_parent, source_key = _resolve_parent(result, source_tokens)
                if isinstance(source_parent, list):
                    del source_parent[int(source_key)]
                elif isinstance(source_parent, dict):
                    del source_parent[source_key]
                else:
                    raise JsonPatchError("move source parent is not a container")
        if not tokens:
            if operation.op in {"remove"}:
                raise JsonPatchError("cannot remove the document root")
            result = deepcopy(operation.value if operation.op in {"add", "replace"} else value)
            continue
        parent, key = _resolve_parent(result, tokens)
        if isinstance(parent, list):
            if key == "-" and operation.op == "add":
                parent.append(deepcopy(operation.value if operation.op == "add" else value))
            else:
                try:
                    index = int(key)
                except ValueError as exc:
                    raise JsonPatchError("array index is invalid") from exc
                if operation.op == "add":
                    if index < 0 or index > len(parent):
                        raise JsonPatchError("array index is out of range")
                    parent.insert(index, deepcopy(operation.value))
                elif operation.op == "remove":
                    if index < 0 or index >= len(parent):
                        raise JsonPatchError("array index is out of range")
                    del parent[index]
                else:
                    if index < 0 or index >= len(parent):
                        raise JsonPatchError("array index is out of range")
                    parent[index] = deepcopy(
                        operation.value if operation.op == "replace" else value
                    )
        elif isinstance(parent, dict):
            if operation.op == "add" or operation.op == "replace":
                if operation.op == "replace" and key not in parent:
                    raise JsonPatchError("replace path does not exist")
                parent[key] = deepcopy(operation.value)
            elif operation.op == "remove":
                if key not in parent:
                    raise JsonPatchError("remove path does not exist")
                del parent[key]
            elif operation.op in {"move", "copy"}:
                parent[key] = value
            else:
                raise JsonPatchError("unsupported operation")
        else:
            raise JsonPatchError("patch parent is not a container")
    return result


T = TypeVar("T", bound=BaseModel)


class SchemaRegistryEntry(StrictModel):
    schema_id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=120)
    model_name: str = Field(min_length=1, max_length=300)
    schema: dict[str, Any]
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @model_validator(mode="after")
    def schema_fingerprint(self) -> SchemaRegistryEntry:
        expected = hashlib.sha256(_canonical_json(self.schema)).hexdigest()
        if self.sha256.lower() != expected:
            raise ValueError("schema fingerprint does not match schema")
        return self


class SchemaRegistry:
    """In-process registry for versioned Pydantic schemas and validators."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], type[BaseModel]] = {}
        self._entries: dict[tuple[str, str], SchemaRegistryEntry] = {}

    def register(
        self, schema_id: str, version: str | type[T] = "1", model: type[T] | None = None
    ) -> SchemaRegistryEntry:
        # Accept both register("id", "v1", Model) and the convenient
        # register("id", Model) form used by local adapters.
        if model is None and isinstance(version, type) and issubclass(version, BaseModel):
            model, version = version, "1"
        if model is None:
            raise TypeError("register requires a Pydantic model")
        version = str(version)
        key = (schema_id, version)
        schema = model.model_json_schema()
        entry = SchemaRegistryEntry(
            schema_id=schema_id,
            version=version,
            model_name=model.__name__,
            schema=schema,
            sha256=hashlib.sha256(_canonical_json(schema)).hexdigest(),
        )
        prior = self._entries.get(key)
        if prior and prior.sha256 != entry.sha256:
            raise ValueError(
                f"schema {schema_id}@{version} is already registered with different content"
            )
        self._models[key] = model
        self._entries[key] = entry
        return entry

    def register_entry(self, entry: SchemaRegistryEntry) -> None:
        key = (entry.schema_id, entry.version)
        prior = self._entries.get(key)
        if prior and prior.sha256 != entry.sha256:
            raise ValueError(f"schema {entry.schema_id}@{entry.version} is already registered")
        self._entries[key] = entry

    def get(self, schema_id: str, version: str) -> SchemaRegistryEntry:
        try:
            return self._entries[(schema_id, version)]
        except KeyError as exc:
            raise KeyError(f"unknown schema {schema_id}@{version}") from exc

    def validate(self, schema_id: str, version: str, value: Any) -> BaseModel:
        try:
            model = self._models[(schema_id, version)]
        except KeyError as exc:
            raise KeyError(f"no validator registered for schema {schema_id}@{version}") from exc
        return model.model_validate(value)

    def entries(self) -> tuple[SchemaRegistryEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))


# Compatibility names used by adapters that call these records envelopes or
# references.  Keeping aliases here avoids parallel model definitions at the
# control boundary.
RecordRef = Reference
ArtifactRef = Reference
SchemaRef = Reference
PatchOperation = JsonPatchOperation
ComponentCatalogRecord = ComponentCatalog
PresetRecord = Preset
OperatorProfileRecord = OperatorProfile
ExperimentMatrixRecord = ExperimentMatrix
Status = StatusRecord
Error = ErrorRecord
TelemetryRecord = TelemetryEvent
apply_patch = apply_json_patch
validate_patch = validate_json_patch


class SourceKind(StrEnum):
    CSV = "csv"
    BIGQUERY = "bigquery"


class LabelSource(StrictModel):
    kind: SourceKind
    location: str  # local/gs:// CSV path or fully qualified BigQuery table
    image_uri_column: str = "image_uri"
    label_column: str = "label"
    sample_id_column: str | None = None
    group_id_column: str | None = None
    image_root: str | None = None


class ObjectSpec(StrictModel):
    slug: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    display_name: str
    description: str = ""
    classes: list[str] = Field(min_length=2)
    class_catalog: ClassCatalog | None = None

    @model_validator(mode="after")
    def unique_classes(self) -> ObjectSpec:
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("class names must be unique")
        catalog_for_object(self)
        return self


class SplitSpec(StrictModel):
    train: float = 0.7
    validation: float = 0.15
    test: float = 0.15
    seed: int = 42

    @model_validator(mode="after")
    def valid_total(self) -> SplitSpec:
        if min(self.train, self.validation, self.test) <= 0:
            raise ValueError("all split fractions must be positive")
        if abs(self.train + self.validation + self.test - 1) > 1e-6:
            raise ValueError("split fractions must sum to one")
        return self


class LabelReviewEvidence(StrictModel):
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime
    accepted_mapping_additions: dict[str, str]
    preview_before_review: list[dict[str, str | int | None]]


class CleanlabSpec(StrictModel):
    """External out-of-sample probabilities used for dataset quality checks."""

    predictions_uri: str
    key_field: Literal["image_uri", "sample_id"] = "image_uri"
    max_issue_fraction: float = Field(default=0.0, ge=0, le=1)


class DatasetSpec(StrictModel):
    object_slug: str
    sources: list[LabelSource] = Field(min_length=1)
    label_mapping: dict[str, str] = Field(default_factory=dict)
    label_review: LabelReviewEvidence | None = None
    cleanlab: CleanlabSpec | None = None
    split: SplitSpec = Field(default_factory=SplitSpec)
    output_uri: str  # gs://bucket/prefix
    shard_max_samples: int = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def review_matches_mapping(self) -> DatasetSpec:
        if self.label_review and any(
            self.label_mapping.get(alias) != target
            for alias, target in self.label_review.accepted_mapping_additions.items()
        ):
            raise ValueError("accepted label review mappings must match the dataset mapping")
        return self


class DatasetVersion(StrictModel):
    version_id: str
    object_slug: str
    root_uri: str
    manifest_uri: str
    shard_uris: dict[str, list[str]]
    sample_counts: dict[str, int]
    sha256: str
    source_snapshot_uri: str
    semantic_manifest_uri: str | None = None
    semantic_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def semantic_reference_pair(self):
        if (self.semantic_manifest_uri is None) != (self.semantic_sha256 is None):
            raise ValueError("semantic manifest URI and checksum must be supplied together")
        return self


class ModelSpec(StrictModel):
    backbone: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    weights_uri: str
    weights_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    image_size: int = Field(default=256, gt=0)
    hidden_dim: int = Field(default=256, gt=0)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    unfreeze_last_n: int = Field(default=0, ge=0)
    preprocessing: PreprocessingSpec | None = None

    @model_validator(mode="after")
    def preprocessing_shape(self):
        if self.preprocessing and self.preprocessing.image_size != self.image_size:
            raise ValueError("preprocessing image size must match model image size")
        return self


class ExperimentConfig(StrictModel):
    experiment_id: str
    object_slug: str
    dataset_version_id: str
    runtime_id: str
    model: ModelSpec
    epochs: int = Field(default=10, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    optimizer: Literal["adamw", "sgd"] = "adamw"
    loss: Literal["cross_entropy", "focal"] = "cross_entropy"
    focal_gamma: float = Field(default=2.0, ge=0)
    horizontal_flip_probability: float = Field(default=0.5, ge=0, le=1)
    class_weights: dict[str, float] = Field(default_factory=dict)
    seed: int = 42
    max_review_error_rate: float | None = Field(default=None, ge=0, lt=1)
    catalog_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    # Optional references preserve compatibility with existing project files
    # while allowing the control plane to bind an exact component/preset and
    # operator profile when those records are available.
    component_refs: tuple[ComponentRef, ...] = ()
    preset_ref: PresetRef | None = None
    operator_profile_ref: OperatorProfileRef | None = None
    matrix_ref: ExperimentMatrixRef | None = None

    @model_validator(mode="after")
    def positive_class_weights(self) -> ExperimentConfig:
        if any(weight <= 0 for weight in self.class_weights.values()):
            raise ValueError("class weights must be positive")
        return self


class VertexJobConfig(StrictModel):
    project: str
    region: str
    machine_type: str
    accelerator_type: str
    accelerator_count: int = Field(ge=1)
    service_account: str
    staging_uri: str
    network: str | None = None
    max_run_hours: float = Field(gt=0)
    estimated_hourly_usd: float = Field(ge=0)
    max_run_cost_usd: float = Field(gt=0)
    retry_limit: int = Field(default=0, ge=0)
    logging_profile: str = "default"
    serving_profile: str | None = None
    runtime_settings: dict[str, Any] = Field(default_factory=dict)


class ValidationResults(StrictModel):
    trainer: bool = False
    container_gpu: bool = False
    vertex_gpu: bool = False
    gcs_read: bool = False
    gcs_write: bool = False


class CertifiedRuntime(StrictModel):
    runtime_id: str
    source_commit: str
    image_tag: str
    image_digest: str
    runtime_version: str
    python_version: str
    pytorch_version: str
    cuda_version: str
    validation: ValidationResults
    certified: bool = False
    certified_at: datetime | None = None

    @model_validator(mode="after")
    def exact_certification(self) -> CertifiedRuntime:
        if self.certified:
            if "@sha256:" not in self.image_digest:
                raise ValueError("certification requires an immutable image digest")
            if not all(self.validation.model_dump().values()) or self.certified_at is None:
                raise ValueError("all validation stages and certified_at are required")
        return self


class RunState(StrEnum):
    PENDING = "pending"
    PREPARING = "preparing"
    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class RunRecord(StrictModel):
    run_id: str
    object_slug: str
    experiment_id: str
    dataset_version_id: str
    runtime_id: str
    state: RunState
    created_at: datetime
    vertex_job_name: str | None = None
    mlflow_run_id: str | None = None
    output_uri: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    logs_uri: str | None = None
    catalog_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    dataset_semantic_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    runtime_image_digest: str | None = None
    runtime_source_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    status_ref: StatusRef | None = None
    error_ref: ErrorRef | None = None
    telemetry_refs: tuple[TelemetryRef, ...] = ()


class ModelRelease(StrictModel):
    object_slug: str
    model_name: str
    model_version: str
    release_id: str
    run_id: str
    dataset_version_id: str
    runtime_id: str
    serving_image_digest: str = Field(pattern=r"^[^\s]+@sha256:[a-f0-9]{64}$")
    state: Literal["staged", "promoted", "rolled_back"]
    approved_by: str | None = None
    approved_at: datetime | None = None
    catalog_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_semantic_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    bundle_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class PlatformConfig(StrictModel):
    gcp_project: str | None = None
    gcp_region: str | None = None
    mlflow_tracking_uri: str | None = None
    litellm_model: str | None = None
    otlp_endpoint: str | None = None
    state_database_url: str | None = None
    workflows_name: str | None = None
    ray_service_namespace: str = "defect-serving"
