# DOC_INVENTORY.md - Module Documentation Status

Last Updated: 2026-01-02

---

## Legend

| Status | Meaning |
|--------|---------|
| `DONE` | Complete documentation meeting quality bar |
| `DRAFT` | Has docstrings but needs enhancement |
| `TODO` | Missing or minimal documentation |

---

## Package: src/didactic_engine/

### Public API Surfaces

| Module | Status | Category | Notes |
|--------|--------|----------|-------|
| `__init__.py` | DONE | public-api | Has module docstring, exports well-defined |
| `config.py` | DONE | public-api | PipelineConfig fully documented |
| `pipeline.py` | DONE | public-api | AudioPipeline class well documented |
| `cli.py` | DONE | cli | argparse provides help strings |

### Core Domain Logic

| Module | Status | Category | Notes |
|--------|--------|----------|-------|
| `ingestion.py` | DRAFT | core | Has basic docstrings, needs examples |
| `analysis.py` | DRAFT | core | Has docstrings, needs feature details |
| `preprocessing.py` | TODO | core | Needs comprehensive docstrings |
| `separation.py` | TODO | core | Needs Demucs integration docs |
| `transcription.py` | TODO | core | Needs Basic Pitch integration docs |

### Processing Modules

| Module | Status | Category | Notes |
|--------|--------|----------|-------|
| `features.py` | TODO | processing | Needs feature extraction docs |
| `segmentation.py` | TODO | processing | Has functions, needs class docs |
| `bar_chunker.py` | TODO | processing | Needs comprehensive docs |
| `align.py` | DONE | processing | Well documented alignment functions |
| `midi_parser.py` | TODO | processing | Needs MIDI parsing details |

### Export Modules

| Module | Status | Category | Notes |
|--------|--------|----------|-------|
| `export_md.py` | TODO | export | Needs Markdown export docs |
| `export_abc.py` | TODO | export | Needs ABC notation docs |

### Utility Modules

| Module | Status | Category | Notes |
|--------|--------|----------|-------|
| `essentia_features.py` | TODO | utils | Needs Essentia feature docs |
| `utils_flatten.py` | DRAFT | utils | Has basic docstrings |

---

## Tests

| File | Status | Notes |
|------|--------|-------|
| `tests/test_pipeline.py` | DRAFT | Test docstrings minimal |

---

## Documentation Files

| File | Status | Notes |
|------|--------|-------|
| `README.md` | DONE | User-facing documentation |
| `AGENT_INSTRUCTIONS.md` | DONE | Maintenance guide |
| `docs/00_README_DEV.md` | DONE | Developer navigation |
| `docs/01_ARCHITECTURE.md` | DONE | System architecture |
| `docs/02_KEY_FLOWS.md` | DONE | Pipeline flows |
| `docs/03_DEBUGGING.md` | DONE | Troubleshooting |
| `docs/04_CONTRIBUTING.md` | DONE | Contribution guide |
| `docs/05_GLOSSARY.md` | DONE | Domain terms |

---

## Summary Statistics

- **Total Modules:** 18
- **DONE:** 5 (28%)
- **DRAFT:** 4 (22%)
- **TODO:** 9 (50%)

---

## Priority Queue

1. **High:** preprocessing.py, separation.py, transcription.py
2. **Medium:** features.py, segmentation.py, midi_parser.py
3. **Low:** export_md.py, export_abc.py, essentia_features.py
