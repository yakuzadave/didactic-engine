# Design Patterns, Anti-Patterns, and Architectural Decisions (Index)

This is an **index page** that links to smaller documents.

Rationale: splitting these topics across multiple files makes them easier to
reference (and easier for tooling to load selectively).

See also:
- [01_ARCHITECTURE.md](01_ARCHITECTURE.md)
- [02_KEY_FLOWS.md](02_KEY_FLOWS.md)
- [06_API_REFERENCE.md](06_API_REFERENCE.md)

---

## Design patterns

Patterns used intentionally across the codebase:
- [07_DESIGN_PATTERNS.md](07_DESIGN_PATTERNS.md)

---

## Anti-patterns and risks

Known risks, footguns, and recommended mitigations:
- [08_ANTI_PATTERNS.md](08_ANTI_PATTERNS.md)

---

## Architectural decisions (ADR-style)

ADR index:
- [adr/README.md](adr/README.md)

---

## Architectural “north stars” (invariants)

These are the assumptions the code is built around:

1. **Audio is processed as mono 1D arrays** inside the pipeline.
2. **Configuration is immutable** and drives deterministic outputs.
3. **Optional tools should not prevent partial progress**.
4. **Outputs are artifacts-first**: write files + datasets + summaries.

If a future change violates one of these, it should be an explicit decision (and ideally captured as an ADR).
