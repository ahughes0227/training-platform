"""Trainer semantic rejection, bundle integrity, and prediction identity tests."""

import json

import pytest

from defect_platform.contracts import ExperimentConfig, ModelSpec
from defect_platform.semantics import (
    PreprocessingSpec,
    SemanticManifest,
    assert_classes_match,
    assert_training_compatible,
    legacy_catalog,
    make_training_manifest,
)
from defect_platform.trainer import runner, training
from defect_platform.trainer.weights import (
    artifact_sha256,
    verify_bundle_integrity,
    write_bundle_integrity,
)


def _inputs(tmp_path):
    cfg = ExperimentConfig(experiment_id="semantics", object_slug="fixture",
        dataset_version_id="v1", runtime_id="local",
        model=ModelSpec(weights_uri=str(tmp_path), weights_sha256="a" * 64))
    dataset = SemanticManifest(kind="dataset", object_slug="fixture",
        catalog=legacy_catalog("fixture", ["scratch", "dent"]), dataset_version_id="v1")
    return cfg, dataset


def test_class_order_rejected_before_torch_or_data_access(tmp_path, monkeypatch):
    cfg, dataset = _inputs(tmp_path)
    monkeypatch.setattr(training, "_torch_modules", lambda: pytest.fail("torch initialized"))
    with pytest.raises(ValueError, match="class order"):
        training._train_experiment_impl(cfg, tmp_path, tmp_path / "out",
            ["dent", "scratch"], semantic_manifest=dataset, dataset_sha256="b" * 64)


def test_dataset_content_identity_required_before_training(tmp_path, monkeypatch):
    cfg, dataset = _inputs(tmp_path)
    monkeypatch.setattr(training, "_torch_modules", lambda: pytest.fail("torch initialized"))
    with pytest.raises(ValueError, match="dataset content SHA256"):
        training._train_experiment_impl(cfg, tmp_path, tmp_path / "out",
            ["scratch", "dent"], semantic_manifest=dataset)


def test_checkpoint_class_order_must_match_catalog(tmp_path):
    cfg, dataset = _inputs(tmp_path)
    manifest = make_training_manifest(dataset, cfg)
    with pytest.raises(ValueError, match="class order"):
        assert_classes_match(manifest, ["dent", "scratch"])


@pytest.mark.parametrize("label", [-1, 2, True, 1.0, "02", "unknown"])
def test_invalid_training_labels_are_rejected(label):
    with pytest.raises((TypeError, ValueError)):
        training._resolve_target(label, {"scratch": 0, "dent": 1}, 2)


@pytest.mark.parametrize("refs", [
    {"dataset_semantic_sha256": "0" * 64, "catalog_sha256": "1" * 64},
    {"dataset_semantic_sha256": "bad", "catalog_sha256": "1" * 64},
    {"dataset_semantic_sha256": "0" * 64, "catalog_sha256": "1" * 64, "unexpected": "x"},
    {"dataset_semantic_sha256": "0" * 64},
])
def test_runner_rejects_bad_semantic_refs_before_training_or_upload(tmp_path, monkeypatch, refs):
    from defect_platform.contracts import DatasetVersion

    config, semantic = _inputs(tmp_path)
    dataset = DatasetVersion(version_id="v1", object_slug="fixture", root_uri=str(tmp_path),
        manifest_uri=str(tmp_path / "manifest.json"), shard_uris={}, sample_counts={},
        sha256="b" * 64, source_snapshot_uri=str(tmp_path / "source-snapshot.json"))
    request = {"experiment": config.model_dump(mode="json"),
        "dataset": dataset.model_dump(mode="json"), "classes": semantic.catalog.labels,
        "semantic_refs": refs, "output_uri": str(tmp_path / "results")}
    monkeypatch.setattr(runner, "load_dataset_semantics", lambda _dataset: semantic)
    monkeypatch.setattr(runner, "train_experiment", lambda *_a, **_kw: pytest.fail("training called"))
    monkeypatch.setattr(runner, "_upload_tree", lambda *_a, **_kw: pytest.fail("upload called"))
    with pytest.raises(ValueError, match="semantic_refs"):
        runner.run_request(request, tmp_path / "local-output")
    assert not (tmp_path / "local-output").exists()


@pytest.mark.parametrize("change", [
    {"horizontal_flip_probability": 0.0},
    {"optimizer": "sgd"},
    {"seed": 7},
])
def test_training_manifest_fingerprint_binds_preprocessing_and_policy(tmp_path, change):
    cfg, dataset = _inputs(tmp_path)
    baseline = make_training_manifest(dataset, cfg)
    altered = cfg.model_copy(update=change)
    changed = make_training_manifest(dataset, altered)
    assert baseline.sha256 != changed.sha256
    assert baseline.parent_semantic_sha256 == changed.parent_semantic_sha256
    with pytest.raises(ValueError, match="training/preprocessing policy"):
        assert_training_compatible(baseline, altered, dataset.catalog.labels)


def test_training_manifest_fingerprint_binds_preprocessing_spec(tmp_path):
    cfg, dataset = _inputs(tmp_path)
    baseline = make_training_manifest(dataset, cfg)
    changed_model = cfg.model.model_copy(update={"preprocessing": PreprocessingSpec(image_size=256,
        normalization_mean=(0.4, 0.4, 0.4))})
    changed = make_training_manifest(dataset, cfg.model_copy(update={"model": changed_model}))
    assert baseline.preprocessing != changed.preprocessing
    assert baseline.sha256 != changed.sha256
    with pytest.raises(ValueError, match="training/preprocessing policy"):
        assert_training_compatible(baseline, cfg.model_copy(update={"model": changed_model}),
                                   dataset.catalog.labels)


def test_bundle_integrity_detects_artifact_tampering_and_unlisted_files(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.pt").write_bytes(b"safe tensor checkpoint")
    (model / "semantics.json").write_text('{"manifest": {}, "sha256": "abc"}')
    write_bundle_integrity(model)
    digest = artifact_sha256(model / "integrity.json")
    verify_bundle_integrity(model, digest)
    with pytest.raises(ValueError, match="integrity manifest checksum mismatch"):
        verify_bundle_integrity(model, "0" * 64)
    (model / "model.pt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_bundle_integrity(model)
    write_bundle_integrity(model)
    (model / "unexpected.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="do not match"):
        verify_bundle_integrity(model)


def test_integrity_manifest_rejects_path_traversal(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "integrity.json").write_text(json.dumps({"schema_version": 1,
        "files": {"../escape": "0" * 64}}))
    with pytest.raises(ValueError, match="invalid path"):
        verify_bundle_integrity(model)


def test_named_cls_members_stream_without_generic_unpickling(tmp_path):
    import io

    wds = pytest.importorskip("webdataset")
    from PIL import Image

    from defect_platform.semantics import normalize_name

    shard = tmp_path / "named.tar"
    encoded = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(encoded, format="PNG")
    with wds.TarWriter(str(shard)) as sink:
        sink.write({"__key__": "named", "png": encoded.getvalue(), "cls": b"scratch\n",
                    "json": json.dumps({"label": "scratch"}).encode(),
                    "pkl": b"this must never be unpickled"})
    stream = training._iterable_dataset([str(shard)], {normalize_name("scratch"): 0,
        normalize_name("dent"): 1}, 2, PreprocessingSpec(image_size=8), False)
    tensor, target = next(iter(stream))
    assert target == 0
    assert tuple(tensor.shape) == (3, 8, 8)


def test_exact_runtime_revision_changes_model_semantic_fingerprint(tmp_path):
    config, dataset = _inputs(tmp_path)
    baseline = make_training_manifest(dataset, config, dataset_sha256="b"*64,
        runtime_image_digest="repo/trainer@sha256:"+"c"*64, runtime_source_commit="d"*40)
    altered = make_training_manifest(dataset, config, dataset_sha256="b"*64,
        runtime_image_digest="repo/trainer@sha256:"+"e"*64, runtime_source_commit="f"*40)
    assert baseline.sha256 != altered.sha256
    assert baseline.runtime_image_digest.endswith("c"*64)
    assert baseline.training_policy.feature_pooling == "mean_spatial_patches"
    with pytest.raises(ValueError, match="supplied together"):
        make_training_manifest(dataset, config, runtime_source_commit="d"*40)
