from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from defect_platform.contracts import DatasetSpec, LabelReviewEvidence, LabelSource, ObjectSpec, SourceKind, SplitSpec
from defect_platform.dataset import (
    LabelRow,
    build_dataset,
    find_duplicates,
    preview_dataset,
    read_label_source,
    verify_dataset_version,
)
from defect_platform.dataset.splits import assign_splits


def png(color: tuple[int, int, int], *, size: int = 24) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (size, size), color).save(output, format="PNG")
    return output.getvalue()


def config(tmp_path, *, mapping=None, shard_size=2):
    obj = ObjectSpec(slug="panel", display_name="Panel", classes=["crack", "dent"])
    spec = DatasetSpec(
        object_slug="panel",
        sources=[LabelSource(kind=SourceKind.CSV, location="unused.csv")],
        label_mapping=mapping or {},
        split=SplitSpec(train=0.6, validation=0.2, test=0.2, seed=19),
        output_uri=str(tmp_path / "datasets"),
        shard_max_samples=shard_size,
    )
    return spec, obj


def test_csv_adapter_resolves_relative_images_and_optional_ids(tmp_path):
    manifest = tmp_path / "labels.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "type", "batch"])
        writer.writeheader()
        writer.writerow({"path": "images/a.png", "type": "Crack", "batch": "b-1"})
    source = LabelSource(
        kind="csv",
        location=str(manifest),
        image_uri_column="path",
        label_column="type",
        group_id_column="batch",
    )

    rows = read_label_source(source)

    assert rows[0].image_uri == str(tmp_path / "images" / "a.png")
    assert rows[0].raw_label == "Crack"
    assert rows[0].group_id == "b-1"
    assert rows[0].row_number == 2


def test_csv_adapter_reports_missing_columns_and_cells(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("image_uri\nimg.png\n", encoding="utf-8")
    source = LabelSource(kind="csv", location=str(path))
    with pytest.raises(ValueError, match="lacks columns: label"):
        read_label_source(source)

    path.write_text("image_uri,label\nimg.png,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing label"):
        read_label_source(source)


def test_bigquery_adapter_quotes_columns_and_maps_rows():
    class Query:
        def __init__(self, sql):
            self.sql = sql

        def result(self):
            return [{"uri": "gs://images/a.png", "kind": "Dent", "lot": "lot-7"}]

    class Client:
        sql = None

        def query(self, sql):
            self.sql = sql
            return Query(sql)

    client = Client()
    source = LabelSource(
        kind="bigquery",
        location="project-1.defects.labels",
        image_uri_column="uri",
        label_column="kind",
        group_id_column="lot",
    )
    rows = read_label_source(source, bq_client=client)
    assert rows[0].image_uri == "gs://images/a.png"
    assert rows[0].group_id == "lot-7"
    assert "`uri`" in client.sql
    with pytest.raises(ValueError, match="project.dataset.table"):
        read_label_source(source.model_copy(update={"location": "not-a-table"}), bq_client=client)


def test_label_preview_normalizes_mapping_and_reports_unresolved_suggestions(tmp_path):
    spec, obj = config(tmp_path, mapping={"split": "crack", "dent-like": "dent"})
    rows = [
        LabelRow("a.png", "SPLIT", source="labels.csv", row_number=2),
        LabelRow("b.png", "dnt", source="labels.csv", row_number=3),
    ]
    preview = preview_dataset(spec, obj, rows=rows)
    assert not preview.ready
    assert preview.class_counts == {"crack": 1, "dent": 0}
    exception = preview.exceptions[0]
    assert exception.reason == "unmapped_label"
    assert exception.suggested_class == "dent"
    assert preview.to_dict()["exceptions"][0]["raw_label"] == "dnt"


def test_preview_detects_conflicting_same_image_and_sample_ids(tmp_path):
    spec, obj = config(tmp_path)
    rows = [
        LabelRow("same.png", "crack", sample_id="id-a", source="one", row_number=1),
        LabelRow("same.png", "dent", sample_id="id-a", source="two", row_number=1),
        LabelRow("else.png", "dent", sample_id="id-a", source="two", row_number=2),
    ]
    reasons = {item.reason for item in preview_dataset(spec, obj, rows=rows).exceptions}
    assert reasons == {"conflicting_labels_for_same_image", "duplicate_sample_id"}


def test_duplicate_detector_finds_exact_and_near_images():
    first = png((255, 0, 0))
    exact = first
    altered = png((254, 0, 0))
    report = find_duplicates([("a", first), ("b", exact), ("c", altered)])
    assert [(pair.left, pair.right) for pair in report.exact_pairs] == [("a", "b")]
    assert any(pair.left == "a" and pair.right == "c" for pair in report.near_pairs)


def test_group_aware_splits_are_deterministic_and_keep_groups_together():
    labels = ["a"] * 5 + ["b"] * 5
    groups = ["batch-a"] * 2 + [f"a-{i}" for i in range(3)] + [f"b-{i}" for i in range(5)]
    split_spec = SplitSpec(train=0.6, validation=0.2, test=0.2, seed=41)
    first = assign_splits(labels, groups, split_spec)
    assert first == assign_splits(labels, groups, split_spec)
    by_group = {}
    for group, split in zip(groups, first):
        assert by_group.setdefault(group, split) == split
    assert set(first) == {"train", "validation", "test"}


def test_builder_creates_webdataset_provenance_checksums_and_idempotent_version(tmp_path):
    spec, obj = config(tmp_path, shard_size=1)
    colors = [(255, 0, 0), (240, 0, 0), (0, 255, 0), (0, 240, 0), (0, 0, 255), (0, 0, 240)]
    labels = ["crack", "crack", "crack", "dent", "dent", "dent"]
    rows = []
    for index, (color, label) in enumerate(zip(colors, labels)):
        path = tmp_path / f"image-{index}.png"
        path.write_bytes(png(color))
        rows.append(LabelRow(str(path), label, sample_id=f"sample-{index}", source="unit.csv", row_number=index + 2))

    version = build_dataset(spec, obj, rows=rows, near_duplicate_distance=0)

    root = tmp_path / "datasets" / version.version_id
    commit = json.loads((root / "_COMMIT.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    snapshot = json.loads((root / "source-snapshot.json").read_text())
    assert commit["sha256"] == version.sha256
    assert manifest["version_id"] == version.version_id
    assert sum(version.sample_counts.values()) == len(rows)
    assert len(snapshot["rows"]) == len(rows)
    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in manifest["shards"].items())
    all_shards = [path for split in version.shard_uris.values() for path in split]
    assert len(all_shards) == len(rows)
    for shard in all_shards:
        with tarfile.open(shard) as tar:
            members = tar.getnames()
            assert len(members) == 3
            assert any(name.endswith(".cls") for name in members)
            assert any(name.endswith(".json") for name in members)
    second = build_dataset(spec, obj, rows=rows, near_duplicate_distance=0)
    assert second.version_id == version.version_id
    assert second.sha256 == version.sha256


def test_review_evidence_is_pinned_to_dataset_version(tmp_path):
    spec, obj = config(tmp_path, mapping={"split": "crack"})
    crack, dent = tmp_path / "crack.png", tmp_path / "dent.png"
    crack.write_bytes(png((200, 20, 20)))
    dent.write_bytes(png((20, 20, 200)))
    review = LabelReviewEvidence(
        reviewer="quality-owner", reviewed_at=datetime.now(UTC),
        accepted_mapping_additions={"split": "crack"},
        preview_before_review=[{"source": "labels.csv", "row_number": 2,
                                "image_uri": str(crack), "raw_label": "split",
                                "suggested_class": "crack", "reason": "unmapped_label"}],
    )
    spec = spec.model_copy(update={"label_review": review})
    rows = [LabelRow(str(crack), "split"), LabelRow(str(dent), "dent")]
    first = build_dataset(spec, obj, rows=rows, near_duplicate_distance=0)
    snapshot = json.loads((tmp_path / "datasets" / first.version_id / "source-snapshot.json").read_text())
    assert snapshot["label_review"]["reviewer"] == "quality-owner"
    verify_dataset_version(first, expected_classes=obj.classes)
    second = build_dataset(spec.model_copy(update={"label_review": review.model_copy(update={"reviewer": "other-owner"})}),
                           obj, rows=rows, near_duplicate_distance=0)
    assert second.version_id != first.version_id


def test_builder_refuses_unresolved_labels_and_conflicting_exact_duplicates(tmp_path):
    spec, obj = config(tmp_path)
    path_a, path_b = tmp_path / "a.png", tmp_path / "b.png"
    payload = png((100, 10, 10))
    path_a.write_bytes(payload)
    path_b.write_bytes(payload)
    with pytest.raises(ValueError, match="review exception"):
        build_dataset(spec, obj, rows=[LabelRow(str(path_a), "unknown")])
    with pytest.raises(ValueError, match="conflicting labels"):
        build_dataset(
            spec,
            obj,
            rows=[LabelRow(str(path_a), "crack"), LabelRow(str(path_b), "dent")],
        )


def test_builder_refuses_overwrite_of_existing_version(tmp_path):
    spec, obj = config(tmp_path)
    data = png((200, 80, 20))
    image_path = tmp_path / "single.png"
    image_path.write_bytes(data)
    other_path = tmp_path / "other.png"
    other_path.write_bytes(png((20, 80, 200)))
    rows = [LabelRow(str(image_path), "crack"), LabelRow(str(other_path), "dent")]
    first = build_dataset(spec, obj, rows=rows, near_duplicate_distance=0)
    root = tmp_path / "datasets" / first.version_id
    (root / "manifest.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite immutable"):
        build_dataset(spec, obj, rows=rows, near_duplicate_distance=0)


def test_verify_dataset_version_accepts_complete_immutable_build(tmp_path):
    spec, obj = config(tmp_path)
    crack, dent = tmp_path / "crack.png", tmp_path / "dent.png"
    crack.write_bytes(png((200, 20, 20)))
    dent.write_bytes(png((20, 20, 200)))
    version = build_dataset(
        spec,
        obj,
        rows=[LabelRow(str(crack), "crack"), LabelRow(str(dent), "dent")],
        near_duplicate_distance=0,
    )

    assert verify_dataset_version(version, expected_classes=obj.classes) is None


@pytest.mark.parametrize(
    ("artifact", "mutation", "message"),
    [
        ("_COMMIT.json", "sha", "_COMMIT.json sha256"),
        ("manifest.json", "classes", "manifest classes"),
        ("source-snapshot.json", "alter", "source snapshot checksum"),
        ("manifest.jsonl", "truncate", "manifest.jsonl checksum"),
        ("shard", "truncate", "shard checksum"),
        ("shard", "delete", "could not checksum"),
    ],
)
def test_verify_dataset_version_fails_closed_on_missing_or_corrupt_artifact(tmp_path, artifact, mutation, message):
    spec, obj = config(tmp_path)
    crack, dent = tmp_path / "crack.png", tmp_path / "dent.png"
    crack.write_bytes(png((200, 20, 20)))
    dent.write_bytes(png((20, 20, 200)))
    version = build_dataset(
        spec,
        obj,
        rows=[LabelRow(str(crack), "crack"), LabelRow(str(dent), "dent")],
        near_duplicate_distance=0,
    )
    root = tmp_path / "datasets" / version.version_id
    if artifact == "shard":
        shard = Path(next(uri for uris in version.shard_uris.values() for uri in uris))
        path = shard
    else:
        path = root / artifact
    if mutation == "sha":
        marker = json.loads(path.read_text())
        marker["sha256"] = "0" * 64
        path.write_text(json.dumps(marker))
    elif mutation == "classes":
        manifest = json.loads(path.read_text())
        manifest["classes"] = ["other"]
        path.write_text(json.dumps(manifest))
    elif mutation == "delete":
        path.unlink()
    elif mutation == "alter":
        snapshot = json.loads(path.read_text())
        snapshot["format_version"] = 999
        path.write_text(json.dumps(snapshot))
    else:
        path.write_bytes(path.read_bytes()[:8])

    with pytest.raises((ValueError, RuntimeError), match=message):
        verify_dataset_version(version, expected_classes=obj.classes)
