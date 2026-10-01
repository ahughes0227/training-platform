"""Plane-agent boundaries, handoffs, and feedback.

The durable platform services remain authoritative.  This module supplies the
small, transport-neutral protocol that plane agents use at those boundaries:
immutable handoff envelopes, deterministic gate decisions, and append-only
feedback.  It deliberately contains no cloud or model-provider integration.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pydantic import Field, model_validator

from .contracts import StrictModel


class Plane(StrEnum):
    DATA = "data"
    TRAINING = "training"
    ANALYSIS = "analysis"
    CONTROL = "control"
    RELEASE = "release"
    BUSINESS = "business"


class HandoffState(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class ValidationIssue(StrictModel):
    code: str = Field(min_length=1, max_length=120)
    layer: str = Field(pattern=r"^(structure|integrity|meaning|authority)$")
    message: str = Field(min_length=1, max_length=2000)
    evidence: tuple[str, ...] = ()
    owner: Plane | None = None


class PlaneAgentContext(StrictModel):
    """The only context a plane agent may receive from the host.

    ``context`` is intentionally an opaque, plane-local mapping.  A caller
    must use ``with_context`` to create a new immutable snapshot; it cannot
    mutate another plane's context in place.
    """

    agent_id: str = Field(min_length=1, max_length=200)
    plane: Plane
    capabilities: frozenset[str] = frozenset()
    context: dict[str, Any] = Field(default_factory=dict)

    model_config: ClassVar = {"frozen": True, "extra": "forbid"}

    def can(self, capability: str) -> bool:
        return capability in self.capabilities

    def with_context(self, **values: Any) -> PlaneAgentContext:
        return self.model_copy(update={"context": {**self.context, **values}})

    def without_context(self, *keys: str) -> PlaneAgentContext:
        removed = set(keys)
        return self.model_copy(update={"context": {k: v for k, v in self.context.items()
                                                     if k not in removed}})


class HandoffEnvelope(StrictModel):
    """An immutable, content-addressed object crossing a plane boundary."""

    handoff_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_plane: Plane
    target_plane: Plane
    object_type: str = Field(min_length=1, max_length=200)
    revision: int = Field(default=1, ge=1)
    parent_fingerprint: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    payload: dict[str, Any]
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_fingerprints: tuple[str, ...] = ()
    authority_scope: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config: ClassVar = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def valid_content_address(self) -> HandoffEnvelope:
        if self.source_plane == self.target_plane:
            raise ValueError("a handoff must cross two different planes")
        expected = payload_fingerprint(self.payload)
        if self.payload_sha256 != expected:
            raise ValueError("handoff payload fingerprint does not match payload")
        if self.revision == 1 and self.parent_fingerprint is not None:
            raise ValueError("the first handoff revision cannot have a parent")
        if self.revision > 1 and self.parent_fingerprint is None:
            raise ValueError("repaired handoffs must reference their parent fingerprint")
        return self

    @property
    def fingerprint(self) -> str:
        value = self.model_dump(mode="json", exclude={"handoff_id", "payload_sha256"})
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class GateDecision(StrictModel):
    state: HandoffState
    receiver: Plane
    handoff_id: str
    revision: int
    issues: tuple[ValidationIssue, ...] = ()
    accepted_fingerprint: str | None = None
    required_action: str | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    decided_by: str = Field(min_length=1)

    @model_validator(mode="after")
    def state_matches_issues(self) -> GateDecision:
        if self.state == HandoffState.ACCEPTED and self.issues:
            raise ValueError("accepted handoffs cannot contain validation issues")
        if self.state != HandoffState.ACCEPTED and not self.issues:
            raise ValueError("rejected or review handoffs require validation issues")
        if self.state == HandoffState.ACCEPTED and not self.accepted_fingerprint:
            raise ValueError("accepted handoffs require the accepted fingerprint")
        return self


class HandoffLedger(Protocol):
    def append(self, envelope: HandoffEnvelope, decision: GateDecision) -> None: ...
    def decisions(self, handoff_id: str) -> list[GateDecision]: ...


class SQLiteHandoffLedger:
    """Append-only local ledger; production may provide a database-backed adapter."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS plane_handoffs (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                handoff_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                envelope TEXT NOT NULL,
                decision TEXT NOT NULL,
                UNIQUE(handoff_id, revision)
            )""")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def append(self, envelope: HandoffEnvelope, decision: GateDecision) -> None:
        if decision.handoff_id != envelope.handoff_id or decision.revision != envelope.revision:
            raise ValueError("ledger decision does not match handoff revision")
        with self._db() as db:
            try:
                db.execute("INSERT INTO plane_handoffs(handoff_id,revision,envelope,decision) VALUES (?,?,?,?)",
                           (envelope.handoff_id, envelope.revision, envelope.model_dump_json(),
                            decision.model_dump_json()))
            except sqlite3.IntegrityError as exc:
                raise ValueError("handoff revision is already recorded") from exc

    def decisions(self, handoff_id: str) -> list[GateDecision]:
        with self._db() as db:
            rows = db.execute("SELECT decision FROM plane_handoffs WHERE handoff_id=? ORDER BY revision",
                              (handoff_id,)).fetchall()
        return [GateDecision.model_validate_json(row[0]) for row in rows]


def payload_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def make_handoff(*, source_plane: Plane, target_plane: Plane, object_type: str,
                 payload: dict[str, Any], authority_scope: tuple[str, ...] = (),
                 evidence_fingerprints: tuple[str, ...] = (), revision: int = 1,
                 parent: HandoffEnvelope | None = None) -> HandoffEnvelope:
    if revision == 1 and parent is not None:
        raise ValueError("a first revision cannot have a parent")
    if revision > 1 and parent is None:
        raise ValueError("a repaired revision requires a parent")
    if parent is not None:
        if parent.source_plane != source_plane or parent.target_plane != target_plane:
            raise ValueError("repair must stay within the original plane boundary")
        if revision != parent.revision + 1:
            raise ValueError("repair revisions must increment by one")
    return HandoffEnvelope(source_plane=source_plane, target_plane=target_plane,
                           object_type=object_type, revision=revision,
                           parent_fingerprint=parent.fingerprint if parent else None,
                           payload=payload, payload_sha256=payload_fingerprint(payload),
                           authority_scope=authority_scope,
                           evidence_fingerprints=evidence_fingerprints)


def gate_handoff(envelope: HandoffEnvelope, receiver: Plane, context: PlaneAgentContext,
                 *, validator: Callable[[dict[str, Any]], tuple[ValidationIssue, ...]] | None = None,
                 ledger: HandoffLedger | None = None) -> GateDecision:
    """Validate a handoff and append the decision before returning it."""
    issues: list[ValidationIssue] = []
    # Pydantic's frozen model prevents field reassignment, but nested mappings
    # can still be mutated by an untrusted adapter. Recheck the content address
    # at the actual boundary so such mutation becomes an explicit rejection.
    if payload_fingerprint(envelope.payload) != envelope.payload_sha256:
        issues.append(ValidationIssue(code="PAYLOAD_MUTATED", layer="integrity",
                                      message="handoff payload changed after it was fingerprinted",
                                      owner=receiver))
    if envelope.target_plane != receiver:
        issues.append(ValidationIssue(code="TARGET_MISMATCH", layer="authority",
                                      message=f"handoff targets {envelope.target_plane.value}, not {receiver.value}",
                                      owner=receiver))
    if context.plane != receiver:
        issues.append(ValidationIssue(code="CONTEXT_MISMATCH", layer="authority",
                                      message="agent context does not belong to the receiving plane",
                                      owner=receiver))
    if not context.can("receive_handoff"):
        issues.append(ValidationIssue(code="CAPABILITY_DENIED", layer="authority",
                                      message="agent lacks the receive_handoff capability", owner=receiver))
    if validator is not None:
        issues.extend(validator(envelope.payload))
    state = HandoffState.ACCEPTED if not issues else HandoffState.REJECTED
    decision = GateDecision(state=state, receiver=receiver, handoff_id=envelope.handoff_id,
                            revision=envelope.revision, issues=tuple(issues),
                            accepted_fingerprint=envelope.fingerprint if not issues else None,
                            required_action=None if not issues else "create a new immutable repair revision",
                            decided_by=context.agent_id)
    if ledger is not None:
        ledger.append(envelope, decision)
    return decision
