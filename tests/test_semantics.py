import json
from typing import Any, cast
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from defect_platform.catalog_store import DirectoryCatalogStore
from defect_platform.cli import _render_object_templates, app
from defect_platform.contracts import (
    DatasetSpec,
    ExperimentConfig,
    LabelSource,
    ModelSpec,
    ObjectSpec,
    SourceKind,
)
from defect_platform.semantics import (
    ClassCatalog,
    ClassDefinition,
    PreprocessingSpec,
    assert_training_compatible,
    make_dataset_manifest,
    make_training_manifest,
    parse_manifest,
    serialize_manifest,
)


def reviewed_catalog():
    return ClassCatalog(
        catalog_id="valve-defects",
        object_slug="valve",
        review_status="reviewed",
        reviewed_by="fixture-operator",
        reviewed_at=datetime(2026, 9, 28, tzinfo=UTC),
        classes=[
            ClassDefinition(
                class_id="crack",
                label="crack",
                definition="A visible material separation",
                aliases=["fracture"],
                positive_examples=["gs://examples/crack.png"],
                negative_examples=["gs://examples/scratch.png"],
            ),
            ClassDefinition(
                class_id="dent", label="dent", definition="An inward surface deformation"
            ),
        ],
    )


def experiment():
    return ExperimentConfig(
        experiment_id="e-1",
        object_slug="valve",
        dataset_version_id="ds-1",
        runtime_id="rt-1",
        catalog_sha256=reviewed_catalog().sha256,
        model=ModelSpec(weights_uri="gs://weights/dino", weights_sha256="a" * 64, image_size=32),
    )


def dataset_semantics():
    obj = ObjectSpec(
        slug="valve",
        display_name="Valve",
        classes=["crack", "dent"],
        class_catalog=reviewed_catalog(),
    )
    spec = DatasetSpec(
        object_slug="valve",
        sources=[LabelSource(kind=SourceKind.CSV, location="labels.csv")],
        output_uri="gs://datasets/valve",
    )
    return make_dataset_manifest(
        obj, spec, "ds-1", source_sha256="b" * 64, split_assignments_sha256="c" * 64
    )


def test_catalog_rejects_ambiguous_aliases_and_invalid_indices():
    values = reviewed_catalog().model_dump(mode="json")
    values["classes"][1]["aliases"] = ["FRACTURE"]
    with pytest.raises(ValueError, match="ambiguous normalized"):
        ClassCatalog.model_validate(values)
    for value in (True, -1, 2, 1.5):
        with pytest.raises(ValueError, match="outside the catalog"):
            reviewed_catalog().decode(cast(Any, value))
    assert reviewed_catalog().decode(reviewed_catalog().encode("FRACTURE")).class_id == "crack"


def test_manifest_round_trip_and_policy_rejection():
    config = experiment()
    value = make_training_manifest(dataset_semantics(), config, dataset_sha256="d" * 64)
    reloaded = parse_manifest(serialize_manifest(value), expected_sha256=value.sha256)
    assert_training_compatible(reloaded, config, ["crack", "dent"])
    for update in (
        {"optimizer": "sgd"},
        {"seed": 7},
        {"learning_rate": 0.1},
        {"catalog_sha256": "f" * 64},
    ):
        changed = ExperimentConfig.model_validate({**config.model_dump(mode="json"), **update})
        with pytest.raises(ValueError):
            assert_training_compatible(reloaded, changed, ["crack", "dent"])
    with pytest.raises(ValueError):
        assert_training_compatible(reloaded, config, ["dent", "crack"])
    values = json.loads(serialize_manifest(value))
    values["manifest"]["preprocessing"]["normalization_mean"][0] = 0.1
    with pytest.raises(ValueError, match="fingerprint"):
        parse_manifest(json.dumps(values), expected_sha256=value.sha256)


def test_normalization_rejects_nonfinite_or_nonpositive_values():
    for std in ((0, 1, 1), (float("nan"), 1, 1), (float("inf"), 1, 1)):
        with pytest.raises(ValueError):
            PreprocessingSpec(image_size=32, normalization_std=std)


def test_approved_catalog_store_is_immutable_and_version_bound(tmp_path):
    store = DirectoryCatalogStore(tmp_path / "approved")
    catalog = reviewed_catalog()
    path = store.publish(catalog)
    assert store.publish(catalog) == path
    assert store.get_catalog("valve", catalog.sha256) == catalog
    assert store.get_catalog("valve", "f" * 64) is None
    changed = catalog.model_dump(mode="json")
    changed["classes"][0]["definition"] = "A break in the coating only"
    with pytest.raises(ValueError, match="immutable"):
        store.publish(ClassCatalog.model_validate(changed))
    changed["version"] = 2
    assert store.publish(ClassCatalog.model_validate(changed))
    values = json.loads(Path(path).read_text())
    values["catalog"]["classes"][0]["label"] = "scratch"
    Path(path).write_text(json.dumps(values))
    with pytest.raises(ValueError, match="fingerprint"):
        store.get_catalog("valve", catalog.sha256)


def test_store_rejects_unreviewed_and_escaping_paths(tmp_path):
    store = DirectoryCatalogStore(tmp_path / "approved")
    values = reviewed_catalog().model_dump(mode="json")
    values.update(review_status="draft", reviewed_by=None, reviewed_at=None)
    with pytest.raises(ValueError, match="only reviewed"):
        store.publish(ClassCatalog.model_validate(values))
    for slug, digest in (("../valve", "a" * 64), ("valve", "../../x")):
        with pytest.raises(ValueError):
            store.get_catalog(slug, digest)
    (tmp_path / "approved").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "approved" / "valve").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes"):
        store.get_catalog("valve", reviewed_catalog().sha256)


def test_catalog_review_requires_explicit_approval_and_definitions(tmp_path):
    obj = ObjectSpec(slug="valve", display_name='Valve "quoted"', classes=["crack", "dent"])
    project = tmp_path / "project"
    _render_object_templates(project, obj)
    config = project / "object.yaml"
    rendered = ObjectSpec.model_validate(yaml.safe_load(config.read_text()))
    assert rendered.class_catalog is not None
    assert rendered.class_catalog.review_status == "draft"
    assert rendered.display_name == obj.display_name
    args = [
        "object",
        "review-catalog",
        str(config),
        "--reviewer",
        "operator",
        "--store",
        str(tmp_path / "catalogs"),
    ]
    runner = CliRunner()
    assert runner.invoke(app, args).exit_code != 0
    assert runner.invoke(app, args + ["--approve"]).exit_code != 0
    values = rendered.model_dump(mode="json")
    for item in values["class_catalog"]["classes"]:
        item["definition"] = f"Human definition of {item['label']}"
    config.write_text(yaml.safe_dump(values))
    result = runner.invoke(app, args + ["--approve"])
    assert result.exit_code == 0, result.output
    accepted = ObjectSpec.model_validate(yaml.safe_load(config.read_text())).class_catalog
    assert accepted is not None
    stored = DirectoryCatalogStore(tmp_path / "catalogs").get_catalog("valve", accepted.sha256)
    assert stored is not None
    assert stored == accepted


def test_gcs_catalog_publication_uses_create_only_and_reads_exact_digest():
    class Conflict(Exception):
        code = 412

    class Missing(Exception):
        code = 404

    class Blob:
        def __init__(self, key, objects):
            self.key, self.objects = key, objects

        def upload_from_string(self, data, *, content_type, if_generation_match):
            assert if_generation_match == 0 and content_type == "application/json"
            if self.key in self.objects:
                raise Conflict()
            self.objects[self.key] = data

        def download_as_bytes(self):
            if self.key not in self.objects:
                raise Missing()
            return self.objects[self.key]

    class Storage:
        def __init__(self):
            self.objects = {}

        def bucket(self, name):
            assert name == "catalogs"
            return self

        def blob(self, key):
            return Blob(key, self.objects)

    client = Storage()
    store = DirectoryCatalogStore("gs://catalogs/approved", client=client)
    catalog = reviewed_catalog()
    store.publish(catalog)
    store.publish(catalog)
    assert store.get_catalog("valve", catalog.sha256) == catalog
    assert store.get_catalog("valve", "0" * 64) is None
