# Anti-Patterns, Risks, and Footguns (didactic-engine)

This document lists anti-patterns and architectural risks observed in the current codebase.

These are not “bad code” judgments; they’re things that tend to bite later as the project grows.

See also:
- [03_DEBUGGING.md](03_DEBUGGING.md)
- [07_DESIGN_PATTERNS_AND_DECISIONS.md](07_DESIGN_PATTERNS_AND_DECISIONS.md) (index)

---

## 1) Two overlapping pipelines in one repository

**What:** The repo contains both `didactic_engine/` and `music_etl/` with overlapping concepts.

**Why it matters:**
- bugfixes can be applied to one pipeline and not the other
- new contributors may not know which is “the” supported path

**Mitigations:**
- keep `docs/06_API_REFERENCE.md` and `docs/00_README_DEV.md` explicit about the primary entrypoint
- consider extracting shared utilities to a shared module or removing duplication over time

---

## 2) Duplicate chunking implementations

**What:** Bar chunking logic appears in:
- `didactic_engine.bar_chunker` (pydub/file helpers)
- `AudioPipeline._process_stems()` (pipeline-integrated chunking + features)

**Risk:** behavior and performance diverge.

**Mitigation idea:** unify behind a single chunking implementation that supports:
- in-memory slicing
- optional persistence of chunk WAVs
- feature extraction hooks

Tracked as an architectural decision:
- ADR: [0006-consolidate-chunking.md](adr/0006-consolidate-chunking.md)

---

## 3) Results schema is a free-form dict

**What:** `AudioPipeline.run()` returns `dict[str, Any]`.

**Risk:**
- callsites can drift (missing keys, different naming)
- docstrings and reports can fall out of sync

**Mitigations:**
- document the schema (see `docs/06_API_REFERENCE.md`)
- optionally introduce a typed results dataclass later if the schema stabilizes

---

## 4) Potential report contract drift

**What:** `export_full_report()` documents keys that may not exactly match current pipeline output keys.

**Mitigation:**
- keep `export_full_report` tolerant (it already uses `.get()`)
- document “expected keys” based on actual pipeline outputs

---

## 5) WSL performance footgun: heavy I/O on `/mnt/<drive>`

**What:** Writing many chunk WAVs to Windows-mounted paths can be dramatically slower under WSL.

**Symptoms:**
- chunk-heavy runs feel much slower on WSL vs native Linux

**Mitigations:**
- write outputs under WSL filesystem (`/home/...`) and copy results back
- use `--no-chunk-wavs` where possible (compute features without writing per-bar WAVs)

---

## 6) Subprocess brittleness for external tools

**What:** External CLIs can fail due to PATH issues, version mismatches, or changed output layout.

**Mitigations:**
- centralize discovery and error messages (already done)
- log command + stderr on failure (already captured in exceptions)

---

## 7) Performance cliffs in “feature-per-chunk” workloads

**What:** Per-bar feature extraction multiplies work: bars × stems.

Mitigations:
- allow disabling chunk persistence (`--no-chunk-wavs`)
- consider caching / batching if the feature set grows
