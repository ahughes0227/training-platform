"""Stable configuration and artifact contracts shared by all platform components."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .semantics import ClassCatalog, PreprocessingSpec, catalog_for_object


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


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
    # A zero estimate would make every cost cap pass, so a real price is required.
    estimated_hourly_usd: float = Field(gt=0)
    max_run_cost_usd: float = Field(gt=0)


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
