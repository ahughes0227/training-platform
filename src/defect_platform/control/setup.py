"""One-shot, schema-constrained assistance for turning an idea into a reviewed run draft."""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from defect_platform.contracts import LabelSource, ObjectSpec, StrictModel


class SetupDraft(StrictModel):
    object: ObjectSpec | None = None
    sources: list[LabelSource] = Field(default_factory=list)
    notes: str = ""
    clarifying_questions: list[str] = Field(default_factory=list)
    review_items: list[str] = Field(default_factory=list)
    experiment: dict[str, Any] = Field(default_factory=dict)


def propose_setup(request: str, *, model: str | None = None, completion=None) -> SetupDraft:
    """Ask LiteLLM for a typed draft. Inject `completion` in tests/offline environments."""
    if completion is None:
        try:
            from litellm import completion as litellm_completion
        except ImportError as exc:
            raise RuntimeError("setup assistance requires the optional agent dependencies") from exc
        completion = litellm_completion
    if not model:
        raise ValueError("configure a LiteLLM model before asking setup questions")
    schema = SetupDraft.model_json_schema()
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Draft a supervised defect-classification setup. Ask concise clarifying questions "
                "for missing required information. Never invent source locations, class labels, "
                "or credentials. Set object to null when its name or classes are unknown. "
                "Put uncertain mappings or assumptions in review_items. Return "
                "JSON matching the supplied schema; do not start jobs or claim validation.\n"
                + json.dumps(schema))},
            {"role": "user", "content": request},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("setup model returned an empty response")
    draft = SetupDraft.model_validate_json(content)
    if not draft.sources and "provide a labeled data source" not in draft.clarifying_questions:
        draft.clarifying_questions.append("Provide a labeled data source (CSV or BigQuery table).")
    return draft
