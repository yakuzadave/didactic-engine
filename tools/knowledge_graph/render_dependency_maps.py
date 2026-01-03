"""Render visual dependency maps and coupling analysis from the knowledge graph.

This script consumes the JSON produced by `tools/knowledge_graph/build_knowledge_graph.py`
and renders a Markdown report with Mermaid diagrams and coupling hotspot tables.

Output:
- docs/09_DEPENDENCY_MAPS.md

Run:
    python tools/knowledge_graph/render_dependency_maps.py

Notes:
- The report intentionally focuses on *internal module dependencies*.
- Call graph edges are not used here; only module IMPORTS edges.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
KG_JSON = REPO_ROOT / "docs" / "knowledge_graph" / "output" / "knowledge_graph.json"
OUT_DOC = REPO_ROOT / "docs" / "09_DEPENDENCY_MAPS.md"


@dataclass(frozen=True)
class ModuleCoupling:
    module: str
    imports_out: int
    imported_by: int

    @property
    def total(self) -> int:
        return self.imports_out + self.imported_by


def _load_graph() -> dict:
    if not KG_JSON.exists():
        raise FileNotFoundError(
            f"Knowledge graph JSON not found at {KG_JSON}. Run: "
            "python tools/knowledge_graph/build_knowledge_graph.py"
        )
    return json.loads(KG_JSON.read_text(encoding="utf-8"))


def _is_internal_module(name: str) -> bool:
    return name.startswith("didactic_engine") or name.startswith("music_etl")


def _split_edges(graph: dict) -> Tuple[List[dict], List[dict]]:
    edges = graph.get("edges", [])
    imports = [e for e in edges if e.get("type") == "IMPORTS"]
    return edges, imports


def _parse_import_edges(import_edges: List[dict]) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """Return (modules, internal_import_pairs).

    Edges are emitted as (source_module, target_module) for internal targets.
    """
    modules: Set[str] = set()
    pairs: List[Tuple[str, str]] = []

    for e in import_edges:
        src = e.get("source", "")
        tgt = e.get("target", "")

        if not src.startswith("module:"):
            continue

        src_mod = src.split(":", 1)[1]
        modules.add(src_mod)

        if tgt.startswith("module:"):
            tgt_mod = tgt.split(":", 1)[1]
            modules.add(tgt_mod)
            if _is_internal_module(src_mod) and _is_internal_module(tgt_mod):
                pairs.append((src_mod, tgt_mod))

    return modules, pairs


def _filter_package(modules: Iterable[str], pairs: List[Tuple[str, str]], prefix: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    mods = sorted([m for m in modules if m.startswith(prefix)])
    modset = set(mods)
    pkg_pairs = [(a, b) for (a, b) in pairs if a in modset and b in modset and a != b]
    return mods, pkg_pairs


def _coupling(mods: List[str], pairs: List[Tuple[str, str]]) -> List[ModuleCoupling]:
    out_counts = Counter(a for a, _ in pairs)
    in_counts = Counter(b for _, b in pairs)

    metrics = [
        ModuleCoupling(module=m, imports_out=int(out_counts.get(m, 0)), imported_by=int(in_counts.get(m, 0)))
        for m in mods
    ]

    metrics.sort(key=lambda x: (x.total, x.imported_by, x.imports_out, x.module), reverse=True)
    return metrics


def _find_cycles(mods: List[str], pairs: List[Tuple[str, str]]) -> List[List[str]]:
    """Find directed cycles using a DFS-based algorithm.

    This is not optimized for huge graphs; our module graphs are small.
    Returns a list of cycles as lists of module names.
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    for a, b in pairs:
        adj[a].append(b)

    cycles: Set[Tuple[str, ...]] = set()

    def dfs(start: str, node: str, path: List[str], on_path: Set[str]) -> None:
        for nxt in adj.get(node, []):
            if nxt == start:
                cyc = path + [start]
                # canonicalize rotation
                min_idx = min(range(len(cyc) - 1), key=lambda i: cyc[i])
                rotated = tuple(cyc[min_idx:-1] + cyc[:min_idx] + [cyc[min_idx]])
                cycles.add(rotated)
            elif nxt not in on_path and len(path) < 20:
                on_path.add(nxt)
                dfs(start, nxt, path + [nxt], on_path)
                on_path.remove(nxt)

    for m in mods:
        dfs(m, m, [m], {m})

    # de-dup and sort by length then name
    out = [list(c) for c in cycles]
    out.sort(key=lambda c: (len(c), c))
    return out


def _mermaid_graph(mods: List[str], pairs: List[Tuple[str, str]], title: str) -> str:
    lines: List[str] = []
    lines.append("```mermaid")
    lines.append("graph TD")
    lines.append(f"%% {title}")

    # Use short ids for readability
    ids = {m: f"M{idx}" for idx, m in enumerate(mods, start=1)}

    for m in mods:
        safe_label = m.replace("`", "")
        lines.append(f"    {ids[m]}[{safe_label}]")

    for a, b in sorted(set(pairs)):
        if a in ids and b in ids:
            lines.append(f"    {ids[a]} --> {ids[b]}")

    lines.append("```")
    return "\n".join(lines)


def _render_table(metrics: List[ModuleCoupling], limit: int = 12) -> str:
    header = "| Module | Imports (out) | Imported-by (in) | Total coupling |\n|---|---:|---:|---:|"
    rows = [
        f"| `{m.module}` | {m.imports_out} | {m.imported_by} | {m.total} |"
        for m in metrics[:limit]
    ]
    return "\n".join([header] + rows)


def main() -> int:
    graph = _load_graph()
    _all_edges, import_edges = _split_edges(graph)
    modules, pairs = _parse_import_edges(import_edges)

    did_mods, did_pairs = _filter_package(modules, pairs, "didactic_engine")
    etl_mods, etl_pairs = _filter_package(modules, pairs, "music_etl")

    did_metrics = _coupling(did_mods, did_pairs)
    etl_metrics = _coupling(etl_mods, etl_pairs)

    did_cycles = _find_cycles(did_mods, did_pairs)
    etl_cycles = _find_cycles(etl_mods, etl_pairs)

    # External package usage by internal modules
    pkg_imports = []
    for e in import_edges:
        src = e.get("source", "")
        tgt = e.get("target", "")
        if not src.startswith("module:"):
            continue
        src_mod = src.split(":", 1)[1]
        if not _is_internal_module(src_mod):
            continue
        if tgt.startswith("package:"):
            pkg = tgt.split(":", 1)[1]
            pkg_imports.append(pkg)

    pkg_counts = Counter(pkg_imports)

    md: List[str] = []
    md.append("# Dependency Maps & Coupling Report")
    md.append("")
    md.append("This report is generated from the repository knowledge graph.")
    md.append("")
    md.append("Artifacts used:")
    md.append("- `docs/knowledge_graph/output/knowledge_graph.json`")
    md.append("")
    md.append("Regenerate:")
    md.append("- `python tools/knowledge_graph/build_knowledge_graph.py`")
    md.append("- `python tools/knowledge_graph/render_dependency_maps.py`")
    md.append("")

    md.append("## didactic_engine module dependency graph")
    md.append("")
    md.append(_mermaid_graph(did_mods, did_pairs, "didactic_engine IMPORTS graph"))
    md.append("")

    md.append("## didactic_engine coupling hotspots")
    md.append("")
    md.append(_render_table(did_metrics, limit=12))
    md.append("")

    md.append("### didactic_engine cycles")
    md.append("")
    if did_cycles:
        for cyc in did_cycles:
            md.append("- " + " -> ".join(f"`{m}`" for m in cyc))
    else:
        md.append("- No cycles detected.")
    md.append("")

    md.append("## music_etl module dependency graph")
    md.append("")
    md.append(_mermaid_graph(etl_mods, etl_pairs, "music_etl IMPORTS graph"))
    md.append("")

    md.append("## music_etl coupling hotspots")
    md.append("")
    md.append(_render_table(etl_metrics, limit=12))
    md.append("")

    md.append("### music_etl cycles")
    md.append("")
    if etl_cycles:
        for cyc in etl_cycles:
            md.append("- " + " -> ".join(f"`{m}`" for m in cyc))
    else:
        md.append("- No cycles detected.")
    md.append("")

    md.append("## External dependency fan-in (imported by internal modules)")
    md.append("")
    md.append("| Package | Import references |\n|---|---:|")
    for pkg, count in pkg_counts.most_common(15):
        md.append(f"| `{pkg}` | {count} |")
    md.append("")

    md.append("## Interpretation guide")
    md.append("")
    md.append("- **High imports (out)**: a module depends on many others (broad dependency surface).")
    md.append("- **High imported-by (in)**: a module is a shared dependency (potential hotspot).")
    md.append("- **Cycles**: increased refactor risk; consider extracting interfaces or moving shared types.")
    md.append("")

    OUT_DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote dependency report to {OUT_DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
