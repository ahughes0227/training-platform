"""WebDataset trainer for single-label defect crops."""

from __future__ import annotations

import importlib
import json
import logging
import math
import random
import re
from pathlib import Path
from typing import Any

from ..contracts import ExperimentConfig
from ..semantics import (
    SemanticManifest,
    assert_training_compatible,
    make_training_manifest,
    normalize_name,
    parse_manifest,
    serialize_manifest,
    write_manifest,
)
from .metrics import calibrate_abstention, evaluate_predictions
from .model import build_model

logger = logging.getLogger(__name__)


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("training requires PyTorch") from exc
    return torch, nn


def _resolve_target(label: Any, class_to_idx: dict[str, int], num_classes: int) -> int:
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    if isinstance(label, str):
        normalized = normalize_name(label)
        if normalized in class_to_idx:
            return class_to_idx[normalized]
        if re.fullmatch(r"0|[1-9][0-9]*", label):
            target = int(label)
        else:
            raise ValueError(f"unknown class label: {label}")
    elif isinstance(label, int) and not isinstance(label, bool):
        target = label
    else:
        raise TypeError("class label must be a canonical string or integer class index")
    if not 0 <= target < num_classes:
        raise ValueError(f"class index {target} is outside [0, {num_classes})")
    return target


def _gcs_open(url: str, mode: str = "rb", bufsize: int = 8192, **_):
    """Stream a gs:// shard with the storage client; the trainer image has no gsutil."""
    if mode != "rb":
        raise ValueError(f"{url}: shards are opened read-only")
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError("Reading gs:// shards requires google-cloud-storage") from exc
    bucket, _, key = url[5:].partition("/")
    return storage.Client().bucket(bucket).blob(key).open("rb")


def _iterable_dataset(shards: list[str], class_to_idx: dict[str, int], num_classes: int, preprocessing,
                      training: bool, horizontal_flip_probability: float = 0.5):
    """Build a WebDataset stream; records need an image member and label/class."""
    try:
        import webdataset as wds
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("WebDataset training requires webdataset, Pillow and torchvision") from exc
    from .preprocessing import image_transform
    if any(str(shard).startswith("gs://") for shard in shards):
        # webdataset's default gs:// handler shells out to gsutil. Its package
        # exports a gopen function that hides the module, so import it by name.
        importlib.import_module("webdataset.gopen").gopen_schemes["gs"] = _gcs_open
    transform = image_transform(preprocessing, training=training,
                                horizontal_flip_probability=horizontal_flip_probability)

    def convert(sample):
        image_bytes = next((value for key, value in sample.items()
                            if key.lower().lstrip(".") in {"jpg", "jpeg", "png", "webp"}
                            or key.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))), None)
        meta_bytes = sample.get("json") or sample.get("cls.json")
        if image_bytes is None:
            raise ValueError("WebDataset sample has no supported image member")
        metadata = json.loads(meta_bytes) if isinstance(meta_bytes, (str, bytes)) else (meta_bytes or {})
        label = metadata.get("label", metadata.get("class"))
        if label is None:
            label = sample.get("label", sample.get("cls"))
        target = _resolve_target(label, class_to_idx, num_classes)
        if isinstance(image_bytes, Image.Image):
            tensor = transform(image_bytes.convert("RGB"))
        else:
            with Image.open(__import__("io").BytesIO(image_bytes)) as im:
                tensor = transform(im.convert("RGB"))
        return tensor, target

    # Parse only our image/JSON/label members. Generic decoding coerces named
    # .cls labels to int and may deserialize arbitrary pickle members.
    return wds.WebDataset(shards, shardshuffle=100 if training else False).map(convert)


def _train_experiment_impl(config: ExperimentConfig, dataset_root: str | Path,
                           output_dir: str | Path, classes: list[str],
                           shard_uris: dict[str, list[str]] | None = None,
                           semantic_manifest: SemanticManifest | None = None,
                           dataset_sha256: str | None = None,
                           mlflow_tracking_uri: str | None = None,
                           mlflow_run_id: str | None = None,
                           runtime_image_digest: str | None = None,
                           runtime_source_commit: str | None = None) -> dict[str, Any]:
    """Train from immutable WebDataset shards and persist an evaluated checkpoint.

    ``dataset_root`` can be local or gs://. Multi-GPU training uses PyTorch
    DataParallel on the single Vertex machine as specified by accelerator_count.
    """
    # Resolve and validate semantics before loading model weights or reading shards.
    if semantic_manifest is None:
        raise ValueError("verified dataset semantic manifest is required for training")
    if not isinstance(semantic_manifest, SemanticManifest):
        semantic_manifest = SemanticManifest.model_validate(semantic_manifest)
    if semantic_manifest.kind != "dataset":
        raise ValueError("training requires a verified dataset semantic manifest")
    if dataset_sha256 is None:
        raise ValueError("verified dataset content SHA256 is required for model semantics")
    semantic_manifest = parse_manifest(serialize_manifest(semantic_manifest))
    training_manifest = make_training_manifest(semantic_manifest, config,
        dataset_sha256=dataset_sha256, runtime_image_digest=runtime_image_digest,
        runtime_source_commit=runtime_source_commit)
    assert_training_compatible(training_manifest, config, classes)
    torch, nn = _torch_modules()
    import numpy as np
    from torch.utils.data import DataLoader
    seed = config.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("classes must contain at least two unique names")
    class_weights = config.class_weights
    if class_weights and set(class_weights) != set(classes):
        missing = sorted(set(classes) - set(class_weights))
        unknown = sorted(set(class_weights) - set(classes))
        raise ValueError(f"class_weights keys must exactly match classes; missing={missing}, unknown={unknown}")
    if any(not math.isfinite(float(weight)) or float(weight) <= 0 for weight in class_weights.values()):
        raise ValueError("class weights must be finite and positive")
    if config.optimizer not in {"adamw", "sgd"}:
        raise ValueError(f"unsupported optimizer: {config.optimizer}")
    if config.loss not in {"cross_entropy", "focal"}:
        raise ValueError(f"unsupported loss: {config.loss}")
    if not 0 <= config.horizontal_flip_probability <= 1:
        raise ValueError("horizontal_flip_probability must be in [0, 1]")
    if shard_uris is None:
        root = str(dataset_root).rstrip("/")
        shard_uris = {split: [f"{root}/{split}-*.tar"] for split in ("train", "validation", "test")}
    class_to_idx = {}
    for index, item in enumerate(semantic_manifest.catalog.classes):
        for label in (item.label, item.class_id, *item.aliases):
            class_to_idx[normalize_name(label)] = index
    for alias, label in semantic_manifest.label_mapping.items():
        class_to_idx[normalize_name(alias)] = semantic_manifest.catalog.encode(label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes), config.model.hidden_dim, config.model.dropout,
                        config.model.unfreeze_last_n, config.model.backbone, config.model.weights_uri,
                        config.model.weights_sha256)
    visible_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if visible_gpus > 1:
        model = nn.DataParallel(model)
    model.to(device)
    trainable_parameters = (p for p in model.parameters() if p.requires_grad)
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate)
    else:
        optimizer = torch.optim.SGD(trainable_parameters, lr=config.learning_rate, momentum=0.9)
    weights_tensor = None
    if class_weights:
        weights_tensor = torch.tensor([class_weights[name] for name in classes],
                                      dtype=torch.float32, device=device)

    def criterion(logits, labels):
        if config.loss == "cross_entropy":
            return nn.functional.cross_entropy(logits, labels, weight=weights_tensor)
        per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
        probabilities = torch.exp(-per_sample)
        focal = (1 - probabilities).pow(config.focal_gamma) * per_sample
        if weights_tensor is not None:
            sample_weights = weights_tensor[labels]
            return (focal * sample_weights).sum() / sample_weights.sum()
        return focal.mean()

    loaders = {
        split: DataLoader(_iterable_dataset(shard_uris.get(split, []), class_to_idx, len(classes),
                                            training_manifest.preprocessing, split == "train",
                                            config.horizontal_flip_probability),
                          batch_size=config.batch_size, num_workers=0,
                          pin_memory=device.type == "cuda")
        for split in ("train", "validation", "test")
    }

    def pass_split(split: str, optimize: bool = False):
        model.train(optimize)
        all_y: list[int] = []; all_pred: list[int] = []; all_probs: list[list[float]] = []
        losses = []
        for images, labels in loaders[split]:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            with torch.set_grad_enabled(optimize):
                logits = model(images)
                loss = criterion(logits, labels)
                if optimize:
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            probs = torch.softmax(logits.detach(), dim=1).cpu()
            all_y.extend(labels.detach().cpu().tolist())
            all_pred.extend(probs.argmax(dim=1).tolist())
            all_probs.extend(probs.tolist()); losses.append(float(loss.item()))
        if not all_y:
            raise ValueError(f"{split} split contains no examples")
        return evaluate_predictions(all_y, all_pred, classes), all_y, all_probs, sum(losses) / len(losses)

    best_key = (-2.0, -1.0); best_epoch = 0; best_state = None; history = []
    logger.info("training_started", extra={"stage": "training_started"})
    for epoch in range(1, config.epochs + 1):
        logger.info("training_epoch_started epoch=%d", epoch,
                    extra={"stage": "training_epoch_started"})
        _train_metric, _, _, train_loss = pass_split("train", optimize=True)
        val_metric, val_y, val_probs, val_loss = pass_split("validation")
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "validation_mcc": val_metric.mcc, "validation_macro_f1": val_metric.macro_f1})
        logger.info("training_epoch_completed epoch=%d train_loss=%.6f validation_loss=%.6f validation_mcc=%.6f validation_macro_f1=%.6f",
                    epoch, train_loss, val_loss, val_metric.mcc, val_metric.macro_f1,
                    extra={"stage": "training_epoch_completed"})
        if val_metric.selection_key > best_key:
            best_key, best_epoch = val_metric.selection_key, epoch
            unwrapped = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {k: v.detach().cpu().clone() for k, v in unwrapped.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    unwrapped.load_state_dict(best_state)
    val_eval, val_y, val_probs, _ = pass_split("validation")
    test_eval, _, _, _ = pass_split("test")
    threshold = None
    if config.max_review_error_rate is not None:
        threshold = calibrate_abstention(val_probs, val_y, config.max_review_error_rate)
        # A target validation cannot satisfy is reported as evidence, not an
        # exception: the caller needs the trained model and its metrics to
        # explain why the goal was not met. Serving abstains on everything
        # until the threshold is reviewed, so no uncalibrated prediction ships.
        if not threshold["target_met"]:
            logger.warning("abstention_target_unmet", extra={
                "stage": "abstention_target_unmet",
                "max_review_error_rate": config.max_review_error_rate})
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(model_dir / "semantics.json", training_manifest)
    checkpoint_path = model_dir / "model.pt"
    model_config = config.model.model_dump()
    # Bundle the backbone config and parameters with the checkpoint so inference
    # does not depend on the original HF repository or a worker-local cache.
    backbone = getattr(unwrapped, "backbone", None)
    if backbone is not None and hasattr(backbone, "save_pretrained"):
        backbone.save_pretrained(model_dir / "backbone")
        model_config["weights_uri"] = "backbone"
        from .weights import artifact_sha256
        model_config["weights_sha256"] = artifact_sha256(model_dir / "backbone")
    else:
        raise ValueError("portable model export requires a backbone with save_pretrained")
    (model_dir / "class_mapping.json").write_text(json.dumps({"classes": classes}, indent=2))
    (model_dir / "inference_config.json").write_text(json.dumps({
        "model": model_config, "confidence_threshold":
        threshold["confidence_threshold"] if threshold else 0.0,
        "image_size": config.model.image_size,
    }, indent=2))
    training_config = {"optimizer": config.optimizer, "loss": config.loss,
                       "focal_gamma": config.focal_gamma,
                       "horizontal_flip_probability": config.horizontal_flip_probability,
                       "class_weights": config.class_weights, "seed": config.seed,
                       "max_review_error_rate": config.max_review_error_rate}
    torch.save({"state_dict": best_state, "classes": classes,
                "model": model_config, "confidence_threshold":
                threshold["confidence_threshold"] if threshold else 0.0,
                "training_config": training_config, "best_epoch": best_epoch,
                "experiment": config.model_dump(mode="json"),
                "semantic_sha256": training_manifest.sha256}, checkpoint_path)
    from .weights import write_bundle_integrity
    write_bundle_integrity(model_dir)
    report = {"classes": classes, "best_epoch": best_epoch, "history": history,
              "validation": val_eval.__dict__, "test": test_eval.__dict__,
              "training_config": training_config,
              "abstention": threshold, "checkpoint": str(checkpoint_path),
              "semantic_sha256": training_manifest.sha256,
              "device": str(device), "visible_gpu_count": visible_gpus}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    if mlflow_tracking_uri:
        report["mlflow_run_id"] = _log_mlflow(mlflow_tracking_uri, config, report,
                                              checkpoint_path, mlflow_run_id)
        (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    logger.info("training_completed", extra={"stage": "training_completed",
                 "best_epoch": best_epoch, "validation_mcc": val_eval.mcc,
                 "validation_macro_f1": val_eval.macro_f1})
    return report


def train_experiment(config: ExperimentConfig, dataset_root: str | Path,
                     output_dir: str | Path, classes: list[str],
                     shard_uris: dict[str, list[str]] | None = None,
                     mlflow_tracking_uri: str | None = None,
                     mlflow_run_id: str | None = None,
                     semantic_manifest: SemanticManifest | None = None,
                     dataset_sha256: str | None = None,
                     runtime_image_digest: str | None = None,
                     runtime_source_commit: str | None = None) -> dict[str, Any]:
    """Run a training job with traceable object, dataset and run log context."""
    from ..telemetry import bind_context, reset_context
    token = bind_context(run_id=mlflow_run_id or config.experiment_id,
                         experiment_id=config.experiment_id, object_slug=config.object_slug,
                         dataset_version_id=config.dataset_version_id, step="trainer")
    try:
        return _train_experiment_impl(config, dataset_root, output_dir, classes,
                                      shard_uris, semantic_manifest, dataset_sha256,
                                      mlflow_tracking_uri, mlflow_run_id,
                                      runtime_image_digest, runtime_source_commit)
    finally:
        reset_context(token)


def _log_mlflow(tracking_uri: str, config: ExperimentConfig, report: dict, checkpoint: Path,
                mlflow_run_id: str | None = None) -> str:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow logging requested but mlflow is not installed") from exc
    from ..mlflow_auth import mlflow_tracking_auth

    # The controller may have created the run before submitting Vertex. Resume
    # that exact run while the Cloud Run IAM token is present for every API call.
    with mlflow_tracking_auth(tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
        run_context = (mlflow.start_run(run_id=mlflow_run_id) if mlflow_run_id else
                       mlflow.start_run(run_name=config.experiment_id))
        with run_context as run:
            mlflow.log_params({"object_slug": config.object_slug,
                               "dataset_version_id": config.dataset_version_id,
                               "runtime_id": config.runtime_id,
                               "backbone": config.model.backbone,
                               "epochs": config.epochs, "batch_size": config.batch_size,
                               "optimizer": config.optimizer, "loss": config.loss,
                               "focal_gamma": config.focal_gamma,
                               "horizontal_flip_probability": config.horizontal_flip_probability,
                               "class_weights": json.dumps(config.class_weights, sort_keys=True)})
            # Only validation metrics are logged as run metrics. Held-out test
            # results stay in the evaluation artifact so no comparison or search
            # can rank runs on them; they are read once for a chosen candidate.
            mlflow.log_metrics({"validation_mcc": report["validation"]["mcc"],
                                "validation_macro_f1": report["validation"]["macro_f1"]})
            for epoch in report.get("history") or ():
                # Per-epoch validation scores, so a search can stop a weak trial
                # while it is still running rather than only after it finishes.
                mlflow.log_metrics({"validation_mcc_epoch": epoch["validation_mcc"],
                                    "validation_macro_f1_epoch": epoch["validation_macro_f1"]},
                                   step=epoch["epoch"])
            mlflow.log_artifacts(str(checkpoint.parent), artifact_path="model")
            report["mlflow_run_id"] = run.info.run_id
            (checkpoint.parent.parent / "evaluation.json").write_text(json.dumps(report, indent=2))
            mlflow.log_artifact(str(checkpoint.parent.parent / "evaluation.json"))
            return run.info.run_id
