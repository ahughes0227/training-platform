"""Typed run evidence read from MLflow for the Analysis plane.

The trainer logs ``evaluation.json`` to every MLflow run.  This module parses
that report, plus an optional ``flagged_cases.json`` artifact, into a frozen
``RunEvidence`` snapshot.  Only validation-split results are retained: the
Analysis plane proposes the next experiment from them, so reading test results
here would leak the held-out split into model selection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from ..contracts import ExperimentConfig, StrictModel

EVALUATION_ARTIFACT = "evaluation.json"
FLAGGED_CASES_ARTIFACT = "flagged_cases.json"
# Trainer-logged settings that must match the experiment the caller supplies.
_EXPERIMENT_KEYS = frozenset({"optimizer", "loss", "focal_gamma", "horizontal_flip_probability",
                              "class_weights", "seed", "max_review_error_rate"})


class ClassMetrics(StrictModel):
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    support: int = Field(ge=0)


class SplitEvaluation(StrictModel):
    mcc: float = Field(ge=-1, le=1)
    macro_f1: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    per_class: dict[str, ClassMetrics]
    confusion: tuple[tuple[int, ...], ...]


class AbstentionCalibration(StrictModel):
    # An unmet target reports a threshold above 1.0 so serving abstains on
    # everything; see trainer.metrics.calibrate_abstention.
    confidence_threshold: float = Field(ge=0)
    review_rate: float = Field(ge=0, le=1)
    accepted_error_rate: float = Field(ge=0, le=1)
    validation_count: int = Field(gt=0)
    accepted_count: int | None = Field(default=None, ge=0)
    target_met: bool = True


class FlaggedCase(StrictModel):
    """One validation sample a reviewer or the trainer singled out."""

    sample_id: str = Field(min_length=1)
    true_class: str
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    reason: Literal["misclassified", "low_confidence", "suspected_label_error"]


class RunEvidence(StrictModel):
    mlflow_run_id: str = Field(min_length=1)
    experiment: ExperimentConfig
    classes: tuple[str, ...] = Field(min_length=2)
    validation: SplitEvaluation
    abstention: AbstentionCalibration | None = None
    flagged_cases: tuple[FlaggedCase, ...] = ()
    report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    model_config: ClassVar = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def consistent_classes(self) -> RunEvidence:
        known = set(self.classes)
        if set(self.validation.per_class) != known:
            raise ValueError("per-class metrics do not match the run's classes")
        n = len(self.classes)
        if len(self.validation.confusion) != n or any(len(row) != n
                                                      for row in self.validation.confusion):
            raise ValueError("confusion matrix shape does not match the run's classes")
        for case in self.flagged_cases:
            if case.true_class not in known or case.predicted_class not in known:
                raise ValueError(f"flagged case {case.sample_id} names an unknown class")
        if any(name not in known for name in self.experiment.class_weights):
            raise ValueError("experiment class weights name an unknown class")
        return self

    def confused_with(self, class_name: str, limit: int = 3) -> list[tuple[str, int]]:
        """Most frequent wrong predictions for ``class_name`` from validation."""
        i = self.classes.index(class_name)
        pairs = [(self.classes[j], count) for j, count in enumerate(self.validation.confusion[i])
                 if j != i and count]
        return sorted(pairs, key=lambda pair: (-pair[1], pair[0]))[:limit]


def report_fingerprint(report: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def evidence_from_report(report: dict[str, Any], experiment: ExperimentConfig, *,
                         mlflow_run_id: str | None = None,
                         flagged_cases: list[dict[str, Any]] | None = None) -> RunEvidence:
    """Build run evidence from the trainer's ``evaluation.json`` report."""
    run_id = mlflow_run_id or report.get("mlflow_run_id")
    if not run_id:
        raise ValueError("evaluation report has no MLflow run id")
    validation = report.get("validation")
    if not isinstance(validation, dict):
        raise TypeError("evaluation report has no validation results")
    logged = report.get("training_config") or {}
    mismatched = sorted(key for key, value in logged.items()
                        if key in _EXPERIMENT_KEYS and getattr(experiment, key) != value)
    if mismatched:
        raise ValueError(f"experiment config does not match the run's logged {mismatched}")
    return RunEvidence(
        mlflow_run_id=run_id, experiment=experiment, classes=tuple(report["classes"]),
        validation=SplitEvaluation.model_validate(validation),
        abstention=report.get("abstention"),
        flagged_cases=tuple(FlaggedCase.model_validate(case) for case in flagged_cases or ()),
        report_sha256=report_fingerprint(report),
    )


ArtifactDownloader = Callable[[str, str], Path | None]


class MlflowRunReader:
    """Read-only access to a training run's evaluation artifacts in MLflow.

    ``download(run_id, artifact_path)`` returns a local file path, or ``None``
    when the optional artifact does not exist.  The default downloader imports
    MLflow lazily so the analysis plane has no hard cloud dependency.
    """

    def __init__(self, tracking_uri: str | None = None, *,
                 download: ArtifactDownloader | None = None):
        if download is None and not tracking_uri:
            raise ValueError("an MLflow tracking URI or artifact downloader is required")
        self.tracking_uri = tracking_uri
        self._download = download or self._mlflow_download

    def _mlflow_download(self, run_id: str, artifact_path: str) -> Path | None:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("reading run evidence requires mlflow") from exc
        from ..mlflow_auth import mlflow_tracking_auth

        assert self.tracking_uri is not None
        with mlflow_tracking_auth(self.tracking_uri):
            mlflow.set_tracking_uri(self.tracking_uri)
            names = {item.path for item in mlflow.MlflowClient().list_artifacts(run_id)}
            if artifact_path not in names:
                return None
            return Path(mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=artifact_path, tracking_uri=self.tracking_uri))

    def read(self, mlflow_run_id: str, experiment: ExperimentConfig) -> RunEvidence:
        report_path = self._download(mlflow_run_id, EVALUATION_ARTIFACT)
        if report_path is None:
            raise FileNotFoundError(f"MLflow run {mlflow_run_id} has no {EVALUATION_ARTIFACT}")
        report = json.loads(Path(report_path).read_text())
        if report.get("mlflow_run_id") not in (None, mlflow_run_id):
            raise ValueError("evaluation report belongs to a different MLflow run")
        flagged_path = self._download(mlflow_run_id, FLAGGED_CASES_ARTIFACT)
        flagged = json.loads(Path(flagged_path).read_text()) if flagged_path else None
        if isinstance(flagged, dict):
            flagged = flagged.get("cases", [])
        return evidence_from_report(report, experiment, mlflow_run_id=mlflow_run_id,
                                    flagged_cases=flagged)
