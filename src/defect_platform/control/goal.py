"""A manager's goal and the one objective every goal decision uses.

A ``Goal`` states what a delivered model must achieve on validation data and
when to stop trying. ``evaluate_goal`` scores a finished run against it, so
the goal runner, the deliverables and the status view all agree on whether a
run met the goal and which run is best.

Test-split results never enter this module: selection uses validation only,
and the chosen candidate's test results are read once, when it is delivered.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from ..analysis.evidence import RunEvidence
from ..contracts import StrictModel


class Goal(StrictModel):
    goal_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    object_slug: str
    description: str = ""
    primary_metric: Literal["mcc", "macro_f1"] = "mcc"
    min_primary: float = Field(ge=-1, le=1)
    min_class_recall: float | None = Field(default=None, gt=0, le=1)
    max_review_rate: float | None = Field(default=None, ge=0, le=1)
    max_runs: int = Field(default=8, ge=1)
    deadline: datetime | None = None
    # Stop once this many runs in a row fail to beat the best primary metric
    # by min_improvement: more runs are unlikely to reach the target.
    plateau_runs: int = Field(default=3, ge=1)
    min_improvement: float = Field(default=0.01, ge=0)
    # Below this many validation samples a weak class is reported as a data
    # need instead of retrained: tuning cannot fix too little evidence.
    min_validation_support: int = Field(default=30, ge=1)

    @model_validator(mode="after")
    def deadline_has_timezone(self) -> Goal:
        if self.deadline is not None and self.deadline.tzinfo is None:
            raise ValueError("goal deadline must include a timezone")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True)
                              .encode()).hexdigest()


class Shortfall(StrictModel):
    target: str
    required: float
    achieved: float | None


class GoalScore(StrictModel):
    met: bool
    primary: float
    macro_f1: float
    shortfalls: tuple[Shortfall, ...] = ()

    @property
    def rank_key(self) -> tuple[bool, float, float]:
        return (self.met, self.primary, self.macro_f1)


def evaluate_goal(goal: Goal, evidence: RunEvidence) -> GoalScore:
    """Score one run's validation evidence against the goal."""
    validation = evidence.validation
    primary = getattr(validation, goal.primary_metric)
    shortfalls = []
    if primary < goal.min_primary:
        shortfalls.append(Shortfall(target=f"validation {goal.primary_metric}",
                                    required=goal.min_primary, achieved=primary))
    if goal.min_class_recall is not None:
        for name in evidence.classes:
            recall = validation.per_class[name].recall
            if recall < goal.min_class_recall:
                shortfalls.append(Shortfall(target=f"{name} recall",
                                            required=goal.min_class_recall, achieved=recall))
    if goal.max_review_rate is not None:
        abstention = evidence.abstention
        if abstention is None or not abstention.target_met:
            # No usable threshold means every prediction goes to review.
            shortfalls.append(Shortfall(target="review rate", required=goal.max_review_rate,
                                        achieved=abstention.review_rate if abstention else None))
        elif abstention.review_rate > goal.max_review_rate:
            shortfalls.append(Shortfall(target="review rate", required=goal.max_review_rate,
                                        achieved=abstention.review_rate))
    return GoalScore(met=not shortfalls, primary=primary, macro_f1=validation.macro_f1,
                     shortfalls=tuple(shortfalls))
