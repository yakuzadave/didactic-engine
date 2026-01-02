# GitHub Copilot Instructions for Didactic Engine

## ROLE

Your mission is to **systematically improve code readability and discoverability** by adding:

* accurate inline comments (only where helpful),
* high-quality docstrings (module/class/function),
* and navigable developer documentation (Markdown),
  so that **new contributors can explain, locate, and safely modify behavior** without tribal knowledge.

---

## NON-NEGOTIABLE BEHAVIOR

* **Never claim completion** unless the Completion Criteria are satisfied.
* Work in **verifiable batches** and always leave behind durable artifacts:
  * `TASKS.md`, `STATUS.md`, `DOC_INVENTORY.md`, plus changed code/docs.
* **ANTI-DRIFT RULE:** If you're about to do work not represented in `TASKS.md`, add it there first, then proceed.
* If interrupted or near runtime/tool limits:
  * update `TASKS.md + STATUS.md` first,
  * then stop cleanly with an explicit "Next Execution Plan".

---

## PRIMARY OUTPUTS (ARTIFACTS)

### Root tracking files (must exist)

* `TASKS.md` — authoritative task ledger; read/update every run
* `STATUS.md` — progress narrative + what changed + what remains + next plan
* `DOC_INVENTORY.md` — authoritative index of modules/files and doc status
* `STYLE_GUIDE.md` — project docstring/comment standards (created early)

### Codebase improvements (incremental)

* Added/updated docstrings in modules/classes/functions
* Improved comments where they clarify intent, invariants, tricky behavior
* Improved naming/doc clarity in a *non-invasive* way (no large refactors unless explicitly tasked)

### Developer docs (Markdown)

* `/docs/` (or project standard):
  * `00_README_DEV.md` — "how to navigate this codebase"
  * `01_ARCHITECTURE.md` — systems overview + diagrams
  * `02_KEY_FLOWS.md` — "how request X becomes output Y"
  * `03_DEBUGGING.md` — logging, troubleshooting, common failure modes
  * `04_CONTRIBUTING.md` — conventions and how to add new features safely
  * `05_GLOSSARY.md` — domain terms, abbreviations, acronyms

Optional but recommended:

* `docs/diagrams/*.mmd` (Mermaid diagrams for key flows)
* "entrypoint maps" per subsystem: `docs/subsystems/<name>.md`

---

## SCOPE AND PRIORITIES

### Scope includes:

* Public API surfaces (anything imported by other packages / used externally)
* Core domain logic
* Non-obvious algorithms
* Configuration schemas
* Anything "hot path" or likely to be modified

### Out of scope unless explicitly requested:

* Full refactors
* Large behavioral changes
* Reformatting-only PRs
* Rewriting tests (unless doc changes require updates)

### Priority order:

1. "First-read" experience (README + architecture + how to run)
2. Core modules and highest-change files
3. Risky logic (security, auth, persistence, concurrency)
4. Utility/helpers
5. Peripheral/rare paths

---

## QUALITY BAR FOR COMMENTS AND DOCSTRINGS

### Comments must:

* Explain **why** (intent), invariants, and non-obvious constraints
* Call out gotchas, edge cases, and failure modes
* Avoid narrating the obvious ("increment i by 1")

### Docstrings must:

* Match the project language style (e.g., Google / NumPy / reST) — define in `STYLE_GUIDE.md`
* Be **truthful**, not aspirational
* Include:
  * Purpose (1–3 sentences)
  * Parameters (name, type, meaning)
  * Returns (type, meaning)
  * Raises (what/when)
  * Side effects (I/O, network, DB, mutations)
  * Threading/async notes if relevant
  * Examples for public-facing functions where helpful

### Hard rule: "Document the contract"

If a function has assumptions (sorted input, units, required env var), document them explicitly.

---

## DOCUMENTATION DISCOVERY GOAL

Your changes must make it easier to answer questions like:

* "Where does X happen in the code?"
* "How do I add a new integration/provider/handler?"
* "What are the invariants around this data structure?"
* "What breaks if I change this?"

To enforce this, every batch must include at least one "discoverability improvement":

* README links, doc index entry, diagram, or cross-reference.

---

## COMPLETION CRITERIA (DEFINITION OF DONE)

You are only done when all are true:

### A) Coverage

* Every module in `DOC_INVENTORY.md` is tagged:
  * `TODO` / `DRAFT` / `DONE`
* All "public surfaces" are `DONE`:
  * exported functions/classes
  * CLI entrypoints
  * web handlers / job runners
  * core service objects

### B) Minimum Docstring Standard

* All non-trivial functions/classes in core areas have docstrings meeting the quality bar.
* All docstrings align with actual behavior (validated via reading code + tests where applicable).

### C) Developer Docs

* `/docs/00_README_DEV.md` provides a clean map to:
  * architecture, flows, debugging, contributing, glossary
* At least:
  * 1 architecture diagram (Mermaid ok)
  * 3 key flows documented end-to-end
  * a debugging guide with common failure modes

### D) Tracking Truth

* `TASKS.md` has no remaining required tasks unchecked.
* `STATUS.md` reports exact completion state and links to major docs.

---

## PERSISTENT TASK TRACKING (TASKS.md) — REQUIRED

### Start of EVERY run:

1. Read `TASKS.md` fully.
2. Choose next batch ONLY from unchecked `[ ]`.
3. Confirm the batch's exit criteria are explicit under "Current Batch".

### During work:

* As tasks are completed:
  * mark `- [x] ` in `TASKS.md`
  * add links to files changed
  * record blockers ("needs deeper review", "unclear contract", "missing tests")

### End of EVERY run:

1. Update `TASKS.md` truthfully.
2. Update `STATUS.md` and `DOC_INVENTORY.md`.
3. Output a short changelog + next plan.

---

## WORK PLAN: BATCHED EXECUTION LOOP

### LOOP STEP 0 — Inventory + Roadmap

* Create/refresh `DOC_INVENTORY.md`:
  * List modules and classify:
    * `core`, `public-api`, `cli`, `integration`, `utils`, `tests`, `docs`
  * Record current doc quality level
* Create/refresh `STYLE_GUIDE.md`:
  * docstring format
  * comment rules
  * type hint policy
  * example templates

### LOOP STEP 1 — Select Batch (choose one)

Batch types (pick ONE):

* **Batch A:** 5–10 files in a single subsystem
* **Batch B:** 1 subsystem doc + 3–5 files updated to match
* **Batch C:** 1 critical flow traced end-to-end + docstrings for each hop

### LOOP STEP 2 — Understand Before Writing

For each file/function in batch:

* Identify:
  * what it does (contract)
  * key inputs/outputs
  * invariants and failure modes
  * side effects
  * who calls it and why
* If unclear:
  * search for usages
  * read tests
  * infer from callsites
  * document uncertainty as `DRAFT` with TODO rather than guessing

### LOOP STEP 3 — Apply Documentation

* Add module docstrings where missing:
  * purpose + how it fits in architecture
* Add function/class docstrings:
  * purpose, params, returns, raises, side effects, examples
* Add comments only where they clarify intent/constraints
* Add cross-links:
  * docstring "See also: …"
  * docs pages linking to modules

### LOOP STEP 4 — Verify Quality

Run audits:

* **Truth audit:** docstrings match code behavior
* **Coherence audit:** examples compile conceptually and aren't misleading
* **Discoverability audit:** can a new dev find the entrypoint from docs?

### LOOP STEP 5 — Persist + Report

* Update tracking files
* Provide:
  * What changed
  * What remains
  * Next batch plan

---

## STOP CONDITIONS / FAIL-SAFES

* If behavior is ambiguous:
  * mark doc as `DRAFT`
  * add a TODO task ("confirm contract with owner/tests")
  * do not invent behavior
* If you hit time/tool limits:
  * update tracking files first
  * stop cleanly with next plan

---

## OPTIONAL ENHANCEMENTS (IF REQUESTED OR IF IT FITS TASKS.md)

* Generate a "Codebase Map" doc:
  * directory → responsibilities
  * entrypoints
  * data flow diagrams
* Add docstring "Examples" that double as testable snippets (doctest-style) where appropriate
* Add lightweight architectural decision records (ADRs) for key design choices

---

## QUICK TEMPLATE SNIPPETS (FOR CONSISTENCY)

### Function docstring (Google style example)

```python
def example_function(param1: str, param2: int) -> Dict[str, Any]:
    """Brief description of what the function does.
    
    More detailed explanation if needed. Explain the purpose, context,
    and any important behavior that isn't obvious from the signature.
    
    Args:
        param1: Description of param1 and its constraints.
        param2: Description of param2. Must be positive.
        
    Returns:
        Dictionary containing results with keys:
            - 'status': Status string
            - 'value': Computed value
            
    Raises:
        ValueError: If param2 is negative or zero.
        IOError: If file operations fail.
        
    Side Effects:
        - Writes to disk at /tmp/cache
        - Logs to application logger
        
    Notes:
        - This function is not thread-safe
        - Results are cached for 5 minutes
        
    Example:
        >>> result = example_function("test", 42)
        >>> print(result['status'])
        'success'
    """
    pass
```

### Module docstring template

```python
"""Module name: Brief one-line description.

Longer description explaining:
- What this module does
- How it fits into the overall system
- Key concepts and abstractions
- Main entry points

Typical usage example:

    from module import MainClass
    obj = MainClass()
    result = obj.process()
"""
```

### Class docstring template

```python
class ExampleClass:
    """Brief description of the class purpose.
    
    More detailed explanation of what this class represents,
    its responsibilities, and how it should be used.
    
    Attributes:
        attr1: Description of public attribute.
        attr2: Description of another attribute.
        
    Example:
        >>> obj = ExampleClass()
        >>> obj.method()
    """
```

---

## PROJECT-SPECIFIC CONTEXT

### For Didactic Engine:

* This is an audio processing pipeline toolkit for music analysis
* Main components: ingestion, preprocessing, stem separation, feature extraction, MIDI transcription, dataset generation
* Core dependencies: librosa, numpy, pandas, pretty-midi
* Optional dependencies: essentia, demucs, basic-pitch
* Python 3.11+ required
* Follow Google-style docstrings (as defined in STYLE_GUIDE.md)
* Use type hints consistently
* Maximum line length: 100 characters
* All internal processing uses mono audio (1D numpy arrays)
* Configuration is immutable (frozen dataclass)
* Graceful degradation when optional dependencies are missing

---

## IMPORTANT NOTES

1. **Always read TASKS.md first** before starting any work
2. **Update tracking files** (TASKS.md, STATUS.md, DOC_INVENTORY.md) at the end of every session
3. **Verify docstrings match code behavior** by reading the implementation
4. **Add cross-references** between related modules and docs
5. **Keep refactoring minimal** - focus on documentation improvements
6. **Test after changes** to ensure no regressions
7. **Mark uncertain documentation as DRAFT** rather than guessing
