# ChatGPT skill layer

The repository skill layer gives ChatGPT a conversational interface over the
existing Training Service authorities. It does not replace dataset, run,
runtime, release, or approval records. The catalog is in
[`skills/manifest.yaml`](../skills/manifest.yaml), and the read-only project and
readiness projections are implemented in
[`defect_platform.skills`](../src/defect_platform/skills.py).

## Directions

**Onboarding** presents the welcome screen, discovers the selected role, and
offers the opt-in local readiness check. Readiness is reported as `ready`,
`blocked`, or `unconfigured`; an unconfigured cloud endpoint is not treated as
a failed check and no check claims to authenticate the user.

**User skills** operate on a Project facade. A project binds an object, an
immutable dataset version, a DINOv3 architecture family and head variant, and
its parameterized experiments. Existing files and control-plane stores remain
authoritative. `List Projects` derives a canonical lifecycle state and shows a
blocker and next action.

**Developer skills** change the platform within the ownership boundaries in
`AGENTS.md`, `docs/PLANE_AGENTS.md`, and `scripts/role_guard.py`. Runtime
certification remains separate from normal experiment execution.

**Internal skills** are agent-only helpers for authority reads, state
derivation, handoffs, evidence, reconciliation, integrity checks, and bounded
recovery. They must preserve uncertain state instead of inferring success.

## Project state

The derived lifecycle is:

`draft` → `data_ingestion` → `dataset_ready` → `experiment_ready` → `training` →
`evaluation` → `release_ready` → `deployed`.

`failed`, `canceled`, and `archived` are exceptional or terminal states. Every
summary includes the source record used for derivation, a blocker when one is
known, and the next useful action.

## Python entry points

```python
from defect_platform.skills import FilesystemProjectReader, check_local_readiness

projects = FilesystemProjectReader("projects").list_projects()
readiness = check_local_readiness(root=".")
```

These entry points are read-only. Mutating skills must call the existing CLI,
control API, dataset builder, runtime release flow, or release ledger rather
than writing parallel authority records.
