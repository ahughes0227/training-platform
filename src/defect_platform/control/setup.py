"""Schema-constrained setup assistance and stateful intake sessions.

The model only proposes a JSON Patch.  The control service owns the draft,
revision, expiry, and validation; callers never need to send the complete
conversation or draft back to the service.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from pydantic import ConfigDict, Field

from defect_platform.contracts import LabelSource, ObjectSpec, StrictModel

try:
    # Shared contract utility when available; this keeps setup patches aligned
    # with control and other module boundaries.
    from defect_platform.contracts import JsonPatchError
    from defect_platform.contracts import apply_json_patch as _contract_apply_json_patch
except ImportError:  # pragma: no cover - compatibility with older checkouts
    JsonPatchError = ValueError
    _contract_apply_json_patch = None


class SetupDraft(StrictModel):
    object: ObjectSpec | None = None
    sources: list[LabelSource] = Field(default_factory=list)
    notes: str = ""
    clarifying_questions: list[str] = Field(default_factory=list)
    review_items: list[str] = Field(default_factory=list)
    experiment: dict[str, Any] = Field(default_factory=dict)


SETUP_SCHEMA_REF = "urn:defect-platform:setup-draft:v1"
INTAKE_PATCH_SCHEMA_REF = "urn:defect-platform:intake-patch:v1"


def setup_schema_checksum() -> str:
    payload = json.dumps(
        SetupDraft.model_json_schema(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def intake_patch_schema_checksum() -> str:
    payload = json.dumps(
        IntakePatch.model_json_schema(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class IntakeSession(StrictModel):
    """The server-owned setup draft and its optimistic-concurrency fence."""

    session_id: str
    revision: int = Field(default=0, ge=0)
    schema_ref: str = SETUP_SCHEMA_REF
    draft: SetupDraft = Field(default_factory=SetupDraft)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


class IntakeTurn(StrictModel):
    """A compact user turn.  ``expected_revision`` prevents lost updates."""

    message: str = Field(min_length=1, max_length=12000)
    expected_revision: int = Field(ge=0)


class IntakePatch(StrictModel):
    """A model response and the validated state produced by its patch."""

    model_config = ConfigDict(extra="forbid")
    patch: list[dict[str, Any]] = Field(default_factory=list)
    # ``None`` means the model omitted the field; an empty list/string is an
    # explicit instruction to clear previously stored metadata.
    clarifying_questions: list[str] | None = None
    review_items: list[str] | None = None
    notes: str | None = None


class IntakeResponse(StrictModel):
    session: IntakeSession
    schema_id: str = SETUP_SCHEMA_REF
    schema_sha256: str = ""
    response_schema_id: str = INTAKE_PATCH_SCHEMA_REF
    response_schema_sha256: str = ""
    patch: list[dict[str, Any]] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    review_items: list[str] = Field(default_factory=list)
    notes: str = ""


def _pointer_tokens(path: str) -> list[str]:
    if path == "":
        return []
    if not path.startswith("/"):
        raise ValueError("JSON Patch path must start with '/'")
    return [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]


def _container(document: Any, tokens: list[str]):
    current = document
    for token in tokens:
        if isinstance(current, dict):
            if token not in current:
                raise ValueError(f"JSON Patch path does not exist: /{'/'.join(tokens)}")
            current = current[token]
        elif isinstance(current, list):
            if token == "-" or not token.isdigit() or int(token) >= len(current):
                raise ValueError("JSON Patch list path is out of range")
            current = current[int(token)]
        else:
            raise ValueError("JSON Patch path traverses a scalar")
    return current


def apply_json_patch(document: dict[str, Any], patch: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the RFC 6902 subset used by setup, without an optional dependency."""
    if _contract_apply_json_patch is not None:
        try:
            result = _contract_apply_json_patch(document, patch)
        except JsonPatchError as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(result, dict):
            raise ValueError("setup patch must produce a JSON object")
        return result
    result = copy.deepcopy(document)
    for operation in patch:
        if not isinstance(operation, dict) or operation.get("op") not in {
            "add",
            "remove",
            "replace",
            "test",
        }:
            raise ValueError("setup patch operation must be add, remove, replace, or test")
        op = operation["op"]
        path = operation.get("path")
        if not isinstance(path, str):
            raise ValueError("setup patch operation requires a path")
        tokens = _pointer_tokens(path)
        if op == "test":
            actual = result if not tokens else _container(result, tokens)
            if actual != operation.get("value"):
                raise ValueError(f"JSON Patch test failed at {path}")
            continue
        if not tokens:
            if op == "remove":
                raise ValueError("cannot remove the setup draft root")
            if "value" not in operation:
                raise ValueError(f"JSON Patch {op} requires a value")
            result = copy.deepcopy(operation["value"])
            continue
        parent = _container(result, tokens[:-1])
        key = tokens[-1]
        if isinstance(parent, dict):
            if (op == "remove" or op == "replace") and key not in parent:
                raise ValueError(f"JSON Patch path does not exist: {path}")
            if op == "remove":
                del parent[key]
            else:
                if "value" not in operation:
                    raise ValueError(f"JSON Patch {op} requires a value")
                parent[key] = copy.deepcopy(operation["value"])
        elif isinstance(parent, list):
            if key == "-":
                if op != "add":
                    raise ValueError("'-' is valid only for JSON Patch add")
                index = len(parent)
            elif key.isdigit():
                index = int(key)
                if index > len(parent) or (op != "add" and index == len(parent)):
                    raise ValueError("JSON Patch list path is out of range")
            else:
                raise ValueError("JSON Patch list path is out of range")
            if op == "remove":
                del parent[index]
            elif op == "add":
                if "value" not in operation:
                    raise ValueError("JSON Patch add requires a value")
                parent.insert(index, copy.deepcopy(operation["value"]))
            else:
                if "value" not in operation:
                    raise ValueError("JSON Patch replace requires a value")
                parent[index] = copy.deepcopy(operation["value"])
        else:
            raise ValueError("JSON Patch path traverses a scalar")
    if not isinstance(result, dict):
        raise ValueError("setup patch must produce a JSON object")
    return result


def apply_setup_patch(draft: SetupDraft, patch: list[dict[str, Any]]) -> SetupDraft:
    """Apply and validate a patch before it can be persisted."""
    return SetupDraft.model_validate(apply_json_patch(draft.model_dump(mode="json"), patch))


def _response_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as exc:
        raise ValueError("setup model returned an invalid response") from exc
    if not content:
        raise ValueError("setup model returned an empty response")
    return content


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
            {
                "role": "system",
                "content": (
                    "Draft a supervised defect-classification setup. Ask concise clarifying questions "
                    "for missing required information. Never invent source locations, class labels, "
                    "or credentials. Set object to null when its name or classes are unknown. "
                    "Put uncertain mappings or assumptions in review_items. Return "
                    "JSON matching the supplied schema; do not start jobs or claim validation. "
                    "schema_ref=" + SETUP_SCHEMA_REF + "\n" + json.dumps(schema)
                ),
            },
            {"role": "user", "content": request},
        ],
        response_format={"type": "json_object"},
    )
    draft = SetupDraft.model_validate_json(_response_content(response))
    if not draft.sources and "provide a labeled data source" not in draft.clarifying_questions:
        draft.clarifying_questions.append("Provide a labeled data source (CSV or BigQuery table).")
    return draft


def _completion_for(model: str | None, completion=None):
    if completion is not None:
        return completion
    try:
        from litellm import completion as litellm_completion
    except ImportError as exc:
        raise RuntimeError("setup assistance requires the optional agent dependencies") from exc
    if not model:
        raise ValueError("configure a LiteLLM model before asking setup questions")
    return litellm_completion


def propose_setup_turn(
    draft: SetupDraft,
    message: str,
    *,
    model: str | None = None,
    completion=None,
    session_id: str = "unknown",
    revision: int = 0,
    missing_fields: list[str] | None = None,
    include_schema: bool = False,
    budget=None,
) -> IntakePatch:
    """Ask for a patch against one server-side draft.

    The request intentionally contains only the current draft and one compact
    turn.  The server applies and validates the patch before returning it.
    """
    if not message.strip():
        raise ValueError("setup turn cannot be empty")
    completion = _completion_for(model, completion)
    schema_hint = (
        "The first turn includes the draft and response schemas in the user payload."
        if include_schema
        else "The schema is available from the schema registry; do not expect it inline."
    )
    wire_payload = {
        "session_id": session_id,
        "revision": revision,
        "missing_fields": missing_fields
        if missing_fields is not None
        else draft.clarifying_questions,
        "latest_answers": message,
        "schema_ref": SETUP_SCHEMA_REF,
        "schema_sha256": setup_schema_checksum(),
        "response_schema_ref": INTAKE_PATCH_SCHEMA_REF,
        "response_schema_sha256": intake_patch_schema_checksum(),
        **(
            {
                "schema": SetupDraft.model_json_schema(),
                "response_schema": IntakePatch.model_json_schema(),
            }
            if include_schema
            else {}
        ),
    }
    if budget is not None:
        from defect_platform.telemetry import AgentTelemetry

        prompt_tokens = max(1, (len(json.dumps(wire_payload, separators=(",", ":"))) + 3) // 4)
        budget.validate(AgentTelemetry(prompt_tokens=prompt_tokens, context_tokens=prompt_tokens))
    response = completion(
        model=cast(str, model),
        messages=[
            {
                "role": "system",
                "content": (
                    "Update a supervised defect-classification setup using RFC 6902 JSON Patch. "
                    "Return only JSON matching this response schema. Use paths against the SetupDraft "
                    "object (object, sources, notes, clarifying_questions, review_items, experiment). "
                    "Never invent source locations, class labels, credentials, or validation. Use add, "
                    "replace, remove, and test only. Keep questions concise and put assumptions in "
                    "review_items. " + schema_hint + " "
                    "schema_ref=" + SETUP_SCHEMA_REF
                ),
            },
            {"role": "user", "content": json.dumps(wire_payload, separators=(",", ":"))},
        ],
        response_format={"type": "json_object"},
    )
    from defect_platform.telemetry import response_telemetry

    with response_telemetry(
        response,
        prompt_tokens=prompt_tokens if budget is not None else None,
        context_tokens=prompt_tokens if budget is not None else 0,
        budget=budget,
    ):
        result = IntakePatch.model_validate_json(_response_content(response))
    # Validate at the boundary even though the caller generally applies this
    # immediately.  This keeps an invalid model patch from becoming a response.
    apply_setup_patch(draft, result.patch)
    return result


def new_intake_session(
    session_id: str, *, ttl_seconds: int = 3600, now: datetime | None = None
) -> IntakeSession:
    if ttl_seconds <= 0 or ttl_seconds > 7 * 24 * 3600:
        raise ValueError("intake session TTL must be between 1 second and 7 days")
    now = now or datetime.now(UTC)
    return IntakeSession(
        session_id=session_id,
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(seconds=ttl_seconds),
    )


class IntakeCoordinator:
    """Application service joining the setup agent to a durable session store."""

    def __init__(
        self,
        store,
        *,
        model: str | None = None,
        completion=None,
        ttl_seconds: int = 3600,
        routing_policy=None,
        budget=None,
    ):
        self.store = store
        self.model = model
        self.completion = completion
        self.ttl_seconds = ttl_seconds
        self.routing_policy = routing_policy
        if budget is None:
            from defect_platform.telemetry import TokenBudget

            budget = TokenBudget.from_env()
        self.budget = budget

    @staticmethod
    def _response(session: IntakeSession, result: IntakePatch) -> IntakeResponse:
        return IntakeResponse(
            session=session,
            patch=result.patch,
            schema_sha256=setup_schema_checksum(),
            response_schema_sha256=intake_patch_schema_checksum(),
            clarifying_questions=session.draft.clarifying_questions,
            review_items=session.draft.review_items,
            notes=session.draft.notes,
        )

    def start(
        self, message: str, *, model: str | None = None, ttl_seconds: int | None = None
    ) -> IntakeResponse:
        if ttl_seconds is not None and (ttl_seconds <= 0 or ttl_seconds > 7 * 24 * 3600):
            raise ValueError("intake session TTL must be between 1 second and 7 days")
        effective_ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        session = new_intake_session(str(uuid.uuid4()), ttl_seconds=effective_ttl)
        self.store.create(session)
        return self.turn(
            session.session_id,
            message,
            expected_revision=0,
            model=model,
            include_schema=True,
            operation="extraction",
        )

    def get(self, session_id: str) -> IntakeSession:
        session = self.store.get(session_id)
        if session is None:
            raise KeyError(f"intake session not found or expired: {session_id}")
        return session

    def turn(
        self,
        session_id: str,
        message: str,
        *,
        expected_revision: int,
        model: str | None = None,
        include_schema: bool = False,
        operation: str = "patch",
    ) -> IntakeResponse:
        session = self.get(session_id)
        if session.revision != expected_revision:
            from defect_platform.control.store import IntakeSessionConflictError

            raise IntakeSessionConflictError(
                f"intake session revision conflict: expected {expected_revision}, current {session.revision}",
                session,
            )
        effective_model = model
        if effective_model is None and self.routing_policy is not None:
            effective_model = self.routing_policy.model_for(operation)
        if effective_model is None:
            effective_model = self.model
        result = propose_setup_turn(
            session.draft,
            message,
            model=effective_model,
            completion=self.completion,
            session_id=session.session_id,
            revision=session.revision,
            missing_fields=session.draft.clarifying_questions,
            include_schema=include_schema,
            budget=self.budget,
        )
        now = datetime.now(UTC)
        draft = apply_setup_patch(session.draft, result.patch)
        metadata = {}
        if result.clarifying_questions is not None:
            metadata["clarifying_questions"] = result.clarifying_questions
        if result.review_items is not None:
            metadata["review_items"] = result.review_items
        if result.notes is not None:
            metadata["notes"] = result.notes
        updated = session.model_copy(
            update={
                "revision": session.revision + 1,
                "draft": draft.model_copy(update=metadata),
                "updated_at": now,
            }
        )
        saved = self.store.update(updated, expected_revision)
        return self._response(saved, result)

    def patch(
        self, session_id: str, patch: list[dict[str, Any]], *, expected_revision: int
    ) -> IntakeResponse:
        session = self.get(session_id)
        if session.revision != expected_revision:
            from defect_platform.control.store import IntakeSessionConflictError

            raise IntakeSessionConflictError(
                f"intake session revision conflict: expected {expected_revision}, current {session.revision}",
                session,
            )
        draft = apply_setup_patch(session.draft, patch)
        now = datetime.now(UTC)
        saved = self.store.update(
            session.model_copy(
                update={"revision": session.revision + 1, "draft": draft, "updated_at": now}
            ),
            expected_revision,
        )
        return self._response(saved, IntakePatch(patch=patch))
