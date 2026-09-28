"""Versioned semantic contracts shared by dataset, training, and serving modules.

Fingerprints bind declared meaning; they do not authenticate a human reviewer or
establish ground-truth accuracy. Operational consumers resolve accepted catalogs
from an operator-controlled store.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256 = r"^[a-f0-9]{64}$"


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return " ".join(re.sub(r"[^\w]+", " ", value).split())


def canonical_bytes(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


class SemanticRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ClassDefinition(SemanticRecord):
    class_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    label: str = Field(min_length=1)
    definition: str = ""
    aliases: list[str] = Field(default_factory=list)
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)


class ClassCatalog(SemanticRecord):
    schema_version: Literal[1] = 1
    catalog_id: str = Field(min_length=1)
    object_slug: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    version: int = Field(default=1, gt=0)
    classes: list[ClassDefinition] = Field(min_length=2)
    review_status: Literal["draft", "reviewed", "legacy"] = "draft"
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def unambiguous(self):
        ids = [item.class_id for item in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("catalog class IDs must be unique")
        names: dict[str, str] = {}
        for item in self.classes:
            for name in (item.label, item.class_id, *item.aliases):
                key = normalize_name(name)
                if not key:
                    raise ValueError("class labels/aliases must contain a meaningful name")
                if key in names and names[key] != item.class_id:
                    raise ValueError(f"ambiguous normalized class label/alias: {name}")
                names[key] = item.class_id
        if self.review_status == "reviewed":
            if not self.reviewed_by or not self.reviewed_by.strip() or self.reviewed_at is None:
                raise ValueError("reviewed catalogs require reviewer and review timestamp")
            if any(not item.definition.strip() for item in self.classes):
                raise ValueError("reviewed catalogs require an explicit definition for each class")
        return self

    @property
    def labels(self) -> list[str]:
        return [item.label for item in self.classes]

    @property
    def class_ids(self) -> list[str]:
        return [item.class_id for item in self.classes]

    @property
    def sha256(self) -> str:
        return fingerprint(self)

    def encode(self, value: str) -> int:
        key = normalize_name(value)
        for index, item in enumerate(self.classes):
            if key in {normalize_name(x) for x in (item.class_id, item.label, *item.aliases)}:
                return index
        raise ValueError(f"unknown catalog class: {value}")

    def decode(self, index: int) -> ClassDefinition:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.classes):
            raise ValueError("class index is outside the catalog")
        return self.classes[index]


def legacy_catalog(object_slug: str, classes: list[str]) -> ClassCatalog:
    """Explicitly mark an old string vocabulary; never invent reviewed definitions."""
    return ClassCatalog(catalog_id=f"{object_slug}-classes", object_slug=object_slug,
        review_status="legacy", classes=[
            ClassDefinition(class_id="cls-" + hashlib.sha256(
                (object_slug + "\0" + normalize_name(label)).encode()).hexdigest()[:16], label=label)
            for label in classes])


def catalog_for_object(object_spec: Any) -> ClassCatalog:
    supplied = getattr(object_spec, "class_catalog", None)
    catalog = ClassCatalog.model_validate(supplied) if supplied is not None else legacy_catalog(
        object_spec.slug, list(object_spec.classes))
    if catalog.object_slug != object_spec.slug or catalog.labels != list(object_spec.classes):
        raise ValueError("object classes/order do not match its class catalog")
    return catalog


def validate_label_mapping(catalog: ClassCatalog, mapping: dict[str, str]) -> None:
    names: dict[str, int] = {}
    for item in catalog.classes:
        target = catalog.encode(item.class_id)
        for name in (item.class_id, item.label, *item.aliases):
            names[normalize_name(name)] = target
    for alias, label in mapping.items():
        index = catalog.encode(label)
        key = normalize_name(alias)
        if not key:
            raise ValueError("label mapping aliases must contain a meaningful name")
        if key in names and names[key] != index:
            raise ValueError(f"conflicting normalized label mapping: {alias}")
        names[key] = index


class PreprocessingSpec(SemanticRecord):
    schema_version: Literal[1] = 1
    image_size: int = Field(gt=0)
    channels: Literal["RGB"] = "RGB"
    interpolation: Literal["bilinear"] = "bilinear"
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @model_validator(mode="after")
    def finite_normalization(self):
        if not all(math.isfinite(x) for x in (*self.normalization_mean, *self.normalization_std)):
            raise ValueError("normalization must be finite")
        if min(self.normalization_std) <= 0:
            raise ValueError("normalization standard deviations must be positive")
        return self


class TrainingPolicy(SemanticRecord):
    feature_pooling: Literal["mean_spatial_patches"] = "mean_spatial_patches"
    exclude_cls_and_register_tokens: Literal[True] = True
    selection_metrics: tuple[Literal["mcc"], Literal["macro_f1"]] = ("mcc", "macro_f1")
    calibration_split: Literal["validation"] = "validation"
    evaluation_split: Literal["test"] = "test"
    optimizer: Literal["adamw", "sgd"]
    loss: Literal["cross_entropy", "focal"]
    focal_gamma: float = Field(ge=0)
    horizontal_flip_probability: float = Field(ge=0, le=1)
    class_weights: dict[str, float] = Field(default_factory=dict)
    seed: int
    max_review_error_rate: float | None = Field(default=None, ge=0, lt=1)

    @model_validator(mode="after")
    def finite_values(self):
        values = [self.focal_gamma, self.horizontal_flip_probability, *self.class_weights.values()]
        if self.max_review_error_rate is not None:
            values.append(self.max_review_error_rate)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("training policy numbers must be finite")
        if any(value <= 0 for value in self.class_weights.values()):
            raise ValueError("class weights must be positive")
        return self


class SemanticManifest(SemanticRecord):
    schema_version: Literal[1] = 1
    kind: Literal["dataset", "model"]
    object_slug: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    catalog: ClassCatalog
    dataset_version_id: str = Field(min_length=1)
    label_mapping: dict[str, str] = Field(default_factory=dict)
    split_policy: dict[str, float | int] = Field(default_factory=dict)
    source_snapshot_sha256: str | None = Field(default=None, pattern=SHA256)
    split_assignments_sha256: str | None = Field(default=None, pattern=SHA256)
    dataset_sha256: str | None = Field(default=None, pattern=SHA256)
    parent_semantic_sha256: str | None = Field(default=None, pattern=SHA256)
    preprocessing: PreprocessingSpec | None = None
    training_policy: TrainingPolicy | None = None
    runtime_id: str | None = None
    runtime_image_digest: str | None = None
    runtime_source_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    weights_sha256: str | None = Field(default=None, pattern=SHA256)
    experiment_sha256: str | None = Field(default=None, pattern=SHA256)

    @model_validator(mode="after")
    def coherent(self):
        if self.object_slug != self.catalog.object_slug:
            raise ValueError("manifest object does not match catalog")
        if (self.runtime_image_digest is None) != (self.runtime_source_commit is None):
            raise ValueError("runtime digest and source commit must be supplied together")
        if self.runtime_image_digest is not None and not re.fullmatch(
                r"[^\s]+@sha256:[a-f0-9]{64}", self.runtime_image_digest):
            raise ValueError("model runtime requires an immutable image digest")
        validate_label_mapping(self.catalog, self.label_mapping)
        if self.kind == "model" and any(item is None for item in (
                self.preprocessing, self.training_policy, self.runtime_id,
                self.weights_sha256, self.experiment_sha256, self.parent_semantic_sha256)):
            raise ValueError("model semantic manifests require complete training policy and lineage")
        if self.kind == "dataset" and any(item is not None for item in (
                self.preprocessing, self.training_policy, self.experiment_sha256)):
            raise ValueError("dataset manifest cannot declare model policy")
        if self.split_policy:
            if set(self.split_policy) != {"train", "validation", "test", "seed"}:
                raise ValueError("split policy must identify train/validation/test fractions and seed")
            fractions = [self.split_policy[k] for k in ("train", "validation", "test")]
            if not all(math.isfinite(x) and x > 0 for x in fractions) or abs(sum(fractions)-1) > 1e-6:
                raise ValueError("invalid semantic split fractions")
        if self.training_policy and self.training_policy.class_weights and set(
                self.training_policy.class_weights) != set(self.catalog.labels):
            raise ValueError("class weights must match the catalog labels")
        return self

    @property
    def sha256(self) -> str:
        return fingerprint(self)


def make_dataset_manifest(object_spec: Any, dataset_spec: Any, version_id: str, *,
                          source_sha256: str | None = None,
                          split_assignments_sha256: str | None = None) -> SemanticManifest:
    catalog = catalog_for_object(object_spec)
    if dataset_spec.object_slug != object_spec.slug:
        raise ValueError("dataset object does not match catalog object")
    return SemanticManifest(kind="dataset", object_slug=object_spec.slug, catalog=catalog,
        dataset_version_id=version_id, label_mapping=dict(dataset_spec.label_mapping),
        split_policy=dataset_spec.split.model_dump(), source_snapshot_sha256=source_sha256,
        split_assignments_sha256=split_assignments_sha256)


def preprocessing_for_config(config: Any) -> PreprocessingSpec:
    supplied = getattr(config.model, "preprocessing", None)
    spec = PreprocessingSpec.model_validate(supplied) if supplied is not None else PreprocessingSpec(
        image_size=config.model.image_size)
    if spec.image_size != config.model.image_size:
        raise ValueError("preprocessing image size does not match model image size")
    return spec


def policy_for_config(config: Any) -> TrainingPolicy:
    fields = ("optimizer", "loss", "focal_gamma", "horizontal_flip_probability",
              "class_weights", "seed", "max_review_error_rate")
    return TrainingPolicy(**{name: getattr(config, name) for name in fields})


def make_training_manifest(dataset: SemanticManifest, config: Any, *,
                           dataset_sha256: str | None = None,
                           runtime_image_digest: str | None = None,
                           runtime_source_commit: str | None = None) -> SemanticManifest:
    if dataset.kind != "dataset" or dataset.object_slug != config.object_slug or (
            dataset.dataset_version_id != config.dataset_version_id):
        raise ValueError("experiment object/dataset identity does not match semantic manifest")
    values = dataset.model_dump()
    values.update(kind="model", parent_semantic_sha256=dataset.sha256,
        dataset_sha256=dataset_sha256, preprocessing=preprocessing_for_config(config),
        training_policy=policy_for_config(config), runtime_id=config.runtime_id,
        runtime_image_digest=runtime_image_digest, runtime_source_commit=runtime_source_commit,
        weights_sha256=config.model.weights_sha256.lower(),
        experiment_sha256=fingerprint(config))
    return SemanticManifest.model_validate(values)


def assert_classes_match(manifest: SemanticManifest, classes: list[str]) -> None:
    if manifest.catalog.labels != list(classes):
        raise ValueError("class order does not match semantic catalog")


def assert_training_compatible(manifest: SemanticManifest, config: Any,
                               classes: list[str]) -> None:
    assert_classes_match(manifest, classes)
    if manifest.kind != "model" or manifest.object_slug != config.object_slug or (
            manifest.dataset_version_id != config.dataset_version_id):
        raise ValueError("training semantic identity mismatch")
    expected = (preprocessing_for_config(config), policy_for_config(config), config.runtime_id,
                config.model.weights_sha256.lower(), fingerprint(config))
    actual = (manifest.preprocessing, manifest.training_policy, manifest.runtime_id,
              manifest.weights_sha256, manifest.experiment_sha256)
    if actual != expected:
        raise ValueError("training/preprocessing policy does not match semantic manifest")
    pinned = getattr(config, "catalog_sha256", None)
    if pinned and pinned != manifest.catalog.sha256:
        raise ValueError("experiment catalog fingerprint mismatch")


def serialize_manifest(manifest: SemanticManifest) -> bytes:
    return canonical_bytes({"manifest": manifest.model_dump(mode="json"), "sha256": manifest.sha256})


def parse_manifest(payload: bytes | str, expected_sha256: str | None = None) -> SemanticManifest:
    envelope = json.loads(payload)
    if not isinstance(envelope, dict) or set(envelope) != {"manifest", "sha256"}:
        raise ValueError("invalid semantic manifest envelope")
    manifest = SemanticManifest.model_validate(envelope["manifest"])
    if envelope["sha256"] != manifest.sha256 or (
            expected_sha256 is not None and expected_sha256 != manifest.sha256):
        raise ValueError("semantic manifest fingerprint mismatch")
    return manifest


def write_manifest(path: str | Path, manifest: SemanticManifest) -> None:
    Path(path).write_bytes(serialize_manifest(manifest))


def read_manifest(path: str | Path, expected_sha256: str | None = None) -> SemanticManifest:
    return parse_manifest(Path(path).read_bytes(), expected_sha256)
