# Plane agents and contract feedback

The platform is operated through six bounded plane agents:

| Plane | Owns | May receive | Must not bypass |
| --- | --- | --- | --- |
| Data | Catalogs, sources, labels, immutable dataset versions | Object and label evidence | Review authority and immutable publication |
| Training | Experiment execution and model artifacts | Verified dataset and runtime records | Runtime certification and control admission |
| Analysis | Evaluation, calibration, diagnostics, and telemetry interpretation | Training results and serving observations | Dataset meaning or release authority |
| Control | Admission, budgets, queueing, dispatch, retries, and reconciliation | Accepted data/training requests | Dataset, runtime, or release policy |
| Release | Eligibility, promotion, rollback, and serving identity | Accepted candidate and analysis evidence | Human/policy approval requirements |
| Business | Policy-bounded operational decisions and workflow actions | Accepted evidence from every plane | Any receiving-plane gate |

`defect_platform.planes` is the transport-neutral boundary protocol. A sender
creates an immutable `HandoffEnvelope`; the receiver calls `gate_handoff` with
its `PlaneAgentContext`. The gate checks target and agent identity/capability,
then runs the receiver's semantic validator. Every result is a `GateDecision`.

Rejected and review-required handoffs include a stable code, validation layer,
owner, explanation, and evidence references. Repair is performed by creating a
new envelope revision with the rejected envelope's fingerprint as its parent.
The original record cannot be edited or accepted retroactively. The SQLite
ledger is suitable for local operation and implements the same append-only
interface that a production Cloud SQL adapter can provide.

The existing control, dataset, runtime, release, and serving services remain
authoritative for paid work and durable state. Plane agents must use narrow
skills or MCP adapters that call those services; they must not implement a
second admission or release policy. Long-running runs continue independently
of agent sessions.

## Boundary acceptance rules

Every receiver must reject a handoff when any of these fail:

1. Structure: the object cannot be parsed or does not satisfy its contract.
2. Integrity: fingerprints, revisions, or evidence do not match.
3. Meaning: the object does not satisfy the receiving plane's semantic policy.
4. Authority: the agent, approval, capability, or freshness scope is invalid.

The Business agent can initiate a cross-plane request, but a rejection cannot
be overridden by that agent. Policy and exception records are the human control
surface for future automation.
