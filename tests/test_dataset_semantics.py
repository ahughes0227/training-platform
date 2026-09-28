from __future__ import annotations

import io
import json
from dataclasses import replace

import pytest
from PIL import Image

from defect_platform.contracts import (
    DatasetSpec,
    LabelSource,
    ObjectSpec,
    SourceKind,
    SplitSpec,
)
from defect_platform.dataset import (
    LabelRow,
    build_dataset,
    load_dataset_semantics,
    verify_dataset_version,
)
from defect_platform.dataset import verification as dataset_verification
from defect_platform.semantics import ClassCatalog, ClassDefinition


def png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (24, 24), color).save(output, format="PNG")
    return output.getvalue()


def dataset_config(tmp_path, *, mapping=None):
    return DatasetSpec(
        object_slug="panel",
        sources=[LabelSource(kind=SourceKind.CSV, location="unused.csv")],
        label_mapping=mapping or {},
        split=SplitSpec(train=0.6, validation=0.2, test=0.2, seed=19),
        output_uri=str(tmp_path / "datasets"),
        shard_max_samples=2,
    )


def rows(tmp_path, first_label="fracture"):
    colors = [(240, 10, 10), (220, 20, 10), (10, 10, 240), (10, 20, 220), (20, 10, 240)]
    labels = [first_label, first_label, "dent", "dent", "fracture"]
    result = []
    for index, (color, label) in enumerate(zip(colors, labels)):
        path = tmp_path / f"image-{index}.png"
        path.write_bytes(png(color))
        result.append(LabelRow(str(path), label, sample_id=f"s-{index}", source="fixture", row_number=index + 2))
    return result


def catalog(*, definition="A visible crack in the panel"):
    return ClassCatalog(
        catalog_id="panel-defects", object_slug="panel", version=1,
        classes=[
            ClassDefinition(class_id="fracture", label="fracture", definition=definition, aliases=["split"]),
            ClassDefinition(class_id="dent", label="dent", definition="A local inward deformation", aliases=["pit"]),
        ],
    )


def test_catalog_aliases_drive_labels_and_manifest_loader(tmp_path):
    obj = ObjectSpec(slug="panel", display_name="Panel", classes=["fracture", "dent"], class_catalog=catalog())
    spec = dataset_config(tmp_path, mapping={"surface break": "split"})
    source_rows = rows(tmp_path)
    source_rows[0] = replace(source_rows[0], raw_label="SPLIT")

    version = build_dataset(spec, obj, rows=source_rows, near_duplicate_distance=0)
    semantic = load_dataset_semantics(version)

    assert semantic.catalog.class_ids == ["fracture", "dent"]
    assert semantic.catalog.encode("pit") == 1
    assert semantic.label_mapping == {"surface break": "split"}
    assert semantic.source_snapshot_sha256
    assert semantic.split_assignments_sha256
    assert verify_dataset_version(version, obj.classes) is None
    envelope = json.loads((tmp_path / "datasets" / version.version_id / "semantics.json").read_text())
    assert envelope["sha256"] == version.semantic_sha256 == semantic.sha256


def test_normalized_alias_mapping_conflict_is_rejected(tmp_path):
    obj = ObjectSpec(slug="panel", display_name="Panel", classes=["fracture", "dent"], class_catalog=catalog())
    spec = dataset_config(tmp_path, mapping={"SPLIT": "dent"})

    with pytest.raises(ValueError, match="conflicting normalized label mapping"):
        build_dataset(spec, obj, rows=rows(tmp_path), near_duplicate_distance=0)


def test_catalog_order_must_match_object_class_order():
    with pytest.raises(ValueError, match="do not match its class catalog"):
        ObjectSpec(slug="panel", display_name="Panel", classes=["dent", "fracture"], class_catalog=catalog())


def test_verifier_rejects_shards_routed_to_the_wrong_split(tmp_path):
    obj = ObjectSpec(slug="panel", display_name="Panel", classes=["fracture", "dent"], class_catalog=catalog())
    version = build_dataset(dataset_config(tmp_path), obj, rows=rows(tmp_path), near_duplicate_distance=0)
    swapped = dict(version.shard_uris)
    swapped["train"], swapped["test"] = swapped["test"], swapped["train"]

    with pytest.raises(ValueError, match="points to a different split"):
        verify_dataset_version(version.model_copy(update={"shard_uris": swapped}), obj.classes)


def test_loader_uses_the_exact_semantic_bytes_it_verifies(tmp_path, monkeypatch):
    obj = ObjectSpec(slug="panel", display_name="Panel", classes=["fracture", "dent"], class_catalog=catalog())
    version = build_dataset(dataset_config(tmp_path), obj, rows=rows(tmp_path), near_duplicate_distance=0)
    semantic_uri = version.semantic_manifest_uri
    read = dataset_verification._read
    semantic_reads = 0

    def read_with_changed_second_semantic_read(uri):
        nonlocal semantic_reads
        if uri == semantic_uri:
            semantic_reads += 1
            if semantic_reads > 1:
                return b'{"manifest":{},"sha256":"' + b"0" * 64 + b'"}'
        return read(uri)

    monkeypatch.setattr(dataset_verification, "_read", read_with_changed_second_semantic_read)
    loaded = load_dataset_semantics(version)

    assert loaded.sha256 == version.semantic_sha256
    assert semantic_reads == 1


def test_catalog_meaning_changes_dataset_identity(tmp_path):
    source_rows = rows(tmp_path)
    first_obj = ObjectSpec(slug="panel", display_name="Panel", classes=["fracture", "dent"], class_catalog=catalog())
    second_obj = ObjectSpec(slug="panel", display_name="Panel", classes=["fracture", "dent"],
                            class_catalog=catalog(definition="A separation in the panel coating"))

    first = build_dataset(dataset_config(tmp_path), first_obj, rows=source_rows, near_duplicate_distance=0)
    second = build_dataset(dataset_config(tmp_path), second_obj, rows=source_rows, near_duplicate_distance=0)

    assert first.version_id != second.version_id


@pytest.mark.parametrize("tamper", ["payload", "checksum"])
def test_semantic_manifest_tampering_is_rejected(tmp_path, tamper):
    obj = ObjectSpec(slug="panel", display_name="Panel", classes=["fracture", "dent"], class_catalog=catalog())
    version = build_dataset(dataset_config(tmp_path), obj, rows=rows(tmp_path), near_duplicate_distance=0)
    path = tmp_path / "datasets" / version.version_id / "semantics.json"
    envelope = json.loads(path.read_text())
    if tamper == "payload":
        envelope["manifest"]["label_mapping"]["forged"] = "dent"
    else:
        envelope["sha256"] = "0" * 64
    path.write_text(json.dumps(envelope))

    with pytest.raises(ValueError, match="semantic manifest"):
        load_dataset_semantics(version)
