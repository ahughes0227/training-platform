"""A CPU-sized demo problem for proving a goal end to end on a small machine.

``write_demo`` creates, under one directory, everything ``defect goal start``
needs: a synthetic three-class crop dataset built through the real dataset
builder, a miniature randomly initialised DINOv3 backbone (no download, no
GPU), and a ``goal.yaml``. Each run trains in seconds on a laptop CPU.

The reachable demo uses visibly different classes. The unreachable demo makes
two classes indistinguishable, so the goal ends in a "why not" report.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..contracts import DatasetSpec, ExperimentConfig, LabelSource, ModelSpec, ObjectSpec
from ..semantics import ClassCatalog, ClassDefinition

CLASSES = ("ok", "scratch", "dent")
IMAGE_SIZE = 32
SAMPLES_PER_CLASS = 50
# Mean colour of each class's crops. In the unreachable demo scratch and dent
# share a colour, so no model can separate them.
COLOURS = {"ok": (200, 200, 200), "scratch": (220, 50, 50), "dent": (50, 50, 220)}


def _write_images(directory: Path, unreachable: bool, seed: int) -> list:
    from PIL import Image

    from ..dataset import LabelRow

    rng = random.Random(seed)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(SAMPLES_PER_CLASS * len(CLASSES)):
        label = CLASSES[index % len(CLASSES)]
        colour = COLOURS["scratch" if unreachable and label == "dent" else label]
        image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        image.putdata([tuple(max(0, min(255, channel + rng.randint(-45, 45))) for channel in colour)
                       for _ in range(IMAGE_SIZE * IMAGE_SIZE)])
        path = directory / f"crop-{index:03d}.png"
        image.save(path)
        rows.append(LabelRow(str(path), label, sample_id=f"crop-{index:03d}",
                             source="demo", row_number=index + 1))
    return rows


def _write_backbone(directory: Path, seed: int) -> tuple[str, str]:
    import torch
    import transformers

    from ..trainer.weights import artifact_sha256

    torch.manual_seed(seed)
    config = transformers.AutoConfig.for_model(
        "dinov3_vit", hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, num_register_tokens=2, patch_size=16, image_size=IMAGE_SIZE)
    transformers.AutoModel.from_config(config).save_pretrained(directory)
    return str(directory), artifact_sha256(directory)


def write_demo(directory: str | Path, *, unreachable: bool = False, seed: int = 7,
               gcs_prefix: str | None = None) -> Path:
    """Write the demo dataset, backbone and goal file; return the goal file path.

    With ``gcs_prefix`` the dataset is published to GCS and the backbone is
    uploaded there, so Vertex AI jobs can read both.
    """
    from ..dataset import build_dataset

    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog = ClassCatalog(
        catalog_id="demo-crops-v1", object_slug="demo-crops", version=1,
        review_status="reviewed", reviewed_by="goal-demo", reviewed_at=datetime.now(UTC),
        classes=[ClassDefinition(class_id=name, label=name, definition=f"a synthetic {name} crop")
                 for name in CLASSES])
    object_spec = ObjectSpec(slug="demo-crops", display_name="Demo crops", classes=list(CLASSES),
                             class_catalog=catalog)
    rows = _write_images(root / "images", unreachable, seed)
    dataset = build_dataset(
        DatasetSpec(object_slug="demo-crops",
                    sources=[LabelSource(kind="csv", location=str(root / "labels.csv"))],
                    output_uri=f"{gcs_prefix.rstrip('/')}/datasets" if gcs_prefix
                    else str(root / "datasets")),
        object_spec, rows=rows, near_duplicate_distance=0)
    (root / "dataset-version.yaml").write_text(
        yaml.safe_dump(dataset.model_dump(mode="json"), sort_keys=False))
    weights_uri, weights_sha256 = _write_backbone(root / "backbone", seed)
    if gcs_prefix:
        from .goal_vertex import upload_directory

        # Content-addressed, so a rerun never replaces weights an earlier run used.
        weights_uri = f"{gcs_prefix.rstrip('/')}/backbones/{weights_sha256[:16]}"
        upload_directory(root / "backbone", weights_uri)
    experiment = ExperimentConfig(
        experiment_id="baseline", object_slug="demo-crops", dataset_version_id=dataset.version_id,
        runtime_id="local", epochs=3, batch_size=16, learning_rate=1e-3, seed=seed,
        model=ModelSpec(weights_uri=weights_uri, weights_sha256=weights_sha256,
                        image_size=IMAGE_SIZE, hidden_dim=16, dropout=0))
    goal = {
        "goal_id": "demo-unreachable" if unreachable else "demo",
        "object_slug": "demo-crops",
        "description": "Name the defect in a synthetic crop.",
        "primary_metric": "mcc", "min_primary": 0.8, "min_class_recall": 0.7,
        "max_runs": 6, "plateau_runs": 3, "min_validation_support": 5,
    }
    goal_file = root / "goal.yaml"
    goal_file.write_text(yaml.safe_dump({
        "goal": goal, "dataset": "dataset-version.yaml",
        "experiment": experiment.model_dump(mode="json"),
    }, sort_keys=False))
    return goal_file
