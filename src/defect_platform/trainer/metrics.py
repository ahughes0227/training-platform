"""Selection metrics and validation-derived abstention calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Evaluation:
    mcc: float
    macro_f1: float
    accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]

    @property
    def selection_key(self) -> tuple[float, float]:
        return self.mcc, self.macro_f1


def evaluate_predictions(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> Evaluation:
    """Compute multiclass MCC, macro F1 and classwise counts without sklearn."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("truth and prediction arrays must have the same nonzero length")
    n = len(class_names)
    if n < 2 or any(i < 0 or i >= n for i in [*y_true, *y_pred]):
        raise ValueError("class indices must fit class_names, which needs at least two classes")
    cm = [[0 for _ in range(n)] for _ in range(n)]
    for truth, pred in zip(y_true, y_pred):
        cm[truth][pred] += 1
    supports = [sum(row) for row in cm]
    predicted = [sum(cm[i][j] for i in range(n)) for j in range(n)]
    per_class: dict[str, dict[str, float]] = {}
    f1s = []
    for i, name in enumerate(class_names):
        tp, support, pred_count = cm[i][i], supports[i], predicted[i]
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": float(support)}
    total = len(y_true)
    trace = sum(cm[i][i] for i in range(n))
    numerator = trace * total - sum(supports[i] * predicted[i] for i in range(n))
    denominator_left = total * total - sum(v * v for v in supports)
    denominator_right = total * total - sum(v * v for v in predicted)
    denominator = (denominator_left * denominator_right) ** 0.5
    mcc = numerator / denominator if denominator else 0.0
    return Evaluation(mcc, sum(f1s) / n, trace / total, per_class, cm)


def calibrate_abstention(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    max_review_error_rate: float,
) -> dict[str, float | int | bool]:
    """Choose the lowest confidence cutoff meeting the validation error limit.

    Review rate is the fraction abstained among all validation samples. Ties are
    handled conservatively; callers must provide validation data, never test data.

    When no cutoff meets the limit, ``target_met`` is false and the threshold is
    above every possible confidence, so serving abstains on every prediction
    rather than accepting all of them. The caller decides what an unmet target
    means for its goal; this function never silently reports success.
    """
    if not 0 <= max_review_error_rate < 1:
        raise ValueError("max_review_error_rate must be in [0, 1)")
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probabilities and labels must have the same nonzero length")
    rows = [list(row) for row in probabilities]
    if any(not row or any(p < 0 or p > 1 for p in row) for row in rows):
        raise ValueError("probabilities must be nonempty and in [0, 1]")
    ranked = sorted(
        ((max(row), int(max(range(len(row)), key=row.__getitem__) == label)) for row, label in zip(rows, labels)),
        reverse=True,
    )
    errors = 0
    best: tuple[float, int, int] | None = None
    for accepted, (confidence, correct) in enumerate(ranked, start=1):
        errors += 1 - correct
        if errors / accepted <= max_review_error_rate:
            best = (confidence, accepted, errors)
    if best is None:
        # Strictly above any probability, so `confidence < threshold` abstains on all.
        return {
            "confidence_threshold": math.nextafter(1.0, 2.0),
            "review_rate": 1.0,
            "accepted_error_rate": 0.0,
            "accepted_count": 0,
            "validation_count": len(labels),
            "target_met": False,
        }
    threshold, count, mistakes = best
    return {
        "confidence_threshold": float(threshold),
        "review_rate": (len(labels) - count) / len(labels),
        "accepted_error_rate": mistakes / count,
        "accepted_count": count,
        "validation_count": len(labels),
        "target_met": True,
    }
