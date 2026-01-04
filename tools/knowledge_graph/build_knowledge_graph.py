"""Build a code knowledge graph for this repository.

This script performs lightweight static analysis over the Python source tree to
produce a *knowledge graph* of code entities and relationships.

Outputs
-------
- JSON graph: docs/knowledge_graph/output/knowledge_graph.json
- GraphML:    docs/knowledge_graph/output/knowledge_graph.graphml

Graph schema (high level)
-------------------------
Nodes:
- module, class, function, method
- package (external dependency)
- project (root)

Edges:
- CONTAINS: module->(class|function), class->method
- IMPORTS: module->(module|package)
- CALLS: function/method -> function/method (best-effort)
- INHERITS: class->class (best-effort, name-only)
- DEPENDS_ON: project->package (from pyproject.toml)

Notes
-----
- CALLS edges are best-effort and primarily capture intra-module calls.
- IMPORTS edges include unresolved targets as package nodes when they appear
  external.

Run
---
From repo root:
    python tools/knowledge_graph/build_knowledge_graph.py

This file intentionally uses only the Python standard library.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "knowledge_graph" / "output"


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    name: str
    file: str = ""
    lineno: int = 0
    doc: str = ""


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    type: str
    detail: str = ""


@dataclass
class DefIndex:
    """Symbol table for a module (what it defines)."""

    functions: Dict[str, str]
    classes: Dict[str, str]
    methods: Dict[Tuple[str, str], str]  # (class, method) -> node_id


class ModuleAnalyzer(ast.NodeVisitor):
    """Extract entities, imports, and best-effort call relationships."""

    def __init__(self, module_name: str, file_path: Path):
        self.module_name = module_name
        self.file_path = file_path

        self.nodes: List[Node] = []
        self.edges: List[Edge] = []

        self.imports: List[str] = []
        self.calls_raw: List[Tuple[str, str]] = []  # (caller_id, callee_expr)
        self.inherits_raw: List[Tuple[str, str]] = []  # (class_id, base_expr)

        self._current_class: Optional[str] = None
        self._current_callable_id: Optional[str] = None

        self._def_index = DefIndex(functions={}, classes={}, methods={})

    @property
    def def_index(self) -> DefIndex:
        return self._def_index

    def _node_id(self, kind: str, qualname: str) -> str:
        return f"{kind}:{qualname}"

    def _add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def _add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        class_qual = f"{self.module_name}.{node.name}"
        class_id = self._node_id("class", class_qual)

        self._add_node(
            Node(
                id=class_id,
                type="class",
                name=class_qual,
                file=str(self.file_path.relative_to(REPO_ROOT)),
                lineno=node.lineno,
                doc=ast.get_docstring(node) or "",
            )
        )
        self._def_index.classes[node.name] = class_id
        self._add_edge(Edge(source=self._node_id("module", self.module_name), target=class_id, type="CONTAINS"))

        for base in node.bases:
            base_expr = _safe_unparse(base)
            if base_expr:
                self.inherits_raw.append((class_id, base_expr))

        prev_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_callable(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_callable(node, is_async=True)

    def _visit_callable(self, node: ast.AST, is_async: bool) -> None:
        name = getattr(node, "name")

        if self._current_class:
            qual = f"{self.module_name}.{self._current_class}.{name}"
            kind = "method"
        else:
            qual = f"{self.module_name}.{name}"
            kind = "function"

        callable_id = self._node_id(kind, qual)

        doc = ast.get_docstring(node) or ""
        lineno = getattr(node, "lineno", 0)

        self._add_node(
            Node(
                id=callable_id,
                type=kind,
                name=qual,
                file=str(self.file_path.relative_to(REPO_ROOT)),
                lineno=lineno,
                doc=doc,
            )
        )

        if kind == "function":
            self._def_index.functions[name] = callable_id
            self._add_edge(Edge(source=self._node_id("module", self.module_name), target=callable_id, type="CONTAINS"))
        else:
            self._def_index.methods[(self._current_class, name)] = callable_id
            class_id = self._def_index.classes.get(self._current_class)
            if class_id:
                self._add_edge(Edge(source=class_id, target=callable_id, type="CONTAINS"))

        prev_callable = self._current_callable_id
        self._current_callable_id = callable_id
        self.generic_visit(node)
        self._current_callable_id = prev_callable

    def visit_Call(self, node: ast.Call) -> Any:
        if self._current_callable_id:
            callee_expr = _safe_unparse(node.func)
            if callee_expr:
                self.calls_raw.append((self._current_callable_id, callee_expr))
        self.generic_visit(node)


def _safe_unparse(node: ast.AST) -> str:
    """Return a best-effort string representation for an AST node."""
    try:
        return ast.unparse(node)
    except Exception:
        # ast.unparse may fail on some edge cases.
        return ""


def _iter_python_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip virtualenvs and caches
        parts = set(Path(dirpath).parts)
        if {".venv", "venv", "__pycache__", ".git"} & parts:
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def _module_name_from_path(file_path: Path) -> Optional[str]:
    """Map file path to importable module name for src-layout packages."""
    rel = file_path.relative_to(REPO_ROOT)

    # didactic_engine package (repo root)
    if rel.parts[:2] == ("src", "didactic_engine"):
        parts = list(rel.parts[1:])  # drop leading 'src'
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].replace(".py", "")
        return ".".join(parts)

    # music_etl package (nested project)
    if rel.parts[:3] == ("music_etl", "src", "music_etl"):
        parts = list(rel.parts[2:])  # drop leading 'music_etl/src'
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].replace(".py", "")
        return ".".join(parts)

    return None


def _read_pyproject_dependencies(pyproject_path: Path) -> List[str]:
    if not pyproject_path.exists():
        return []

    # Python 3.11+ stdlib
    import tomllib

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])

    packages: List[str] = []
    for dep in deps:
        # dep examples: "numpy>=1.24.0,<2.0.0" or "basic-pitch>=0.2.0; python_version < '3.12'"
        name = dep.split(";")[0].strip()
        # Strip version constraints
        for sep in ["==", ">=", "<=", "~=", "!=", ">", "<"]:
            if sep in name:
                name = name.split(sep)[0].strip()
                break
        if name:
            packages.append(name)

    return sorted(set(packages))


def build_graph(
    source_roots: List[Path],
    report_syntax_errors: bool = False,
) -> Tuple[List[Node], List[Edge]]:
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []
    syntax_errors: List[Tuple[Path, SyntaxError]] = []

    # Root node
    project_id = "project:didactic-engine"
    nodes[project_id] = Node(id=project_id, type="project", name="didactic-engine")

    # External dependencies (from root pyproject)
    for pkg in _read_pyproject_dependencies(REPO_ROOT / "pyproject.toml"):
        pkg_id = f"package:{pkg}"
        nodes.setdefault(pkg_id, Node(id=pkg_id, type="package", name=pkg))
        edges.append(Edge(source=project_id, target=pkg_id, type="DEPENDS_ON"))

    # Analyze modules
    module_indexes: Dict[str, DefIndex] = {}
    module_files: Dict[str, Path] = {}
    per_module_calls: Dict[str, List[Tuple[str, str]]] = {}

    for root in source_roots:
        for py_file in _iter_python_files(root):
            mod = _module_name_from_path(py_file)
            if not mod:
                continue

            module_id = f"module:{mod}"
            module_files[mod] = py_file
            nodes.setdefault(
                module_id,
                Node(
                    id=module_id,
                    type="module",
                    name=mod,
                    file=str(py_file.relative_to(REPO_ROOT)),
                    lineno=1,
                    doc="",
                ),
            )

            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except SyntaxError as exc:
                # Skip broken files rather than failing the entire graph build.
                syntax_errors.append((py_file, exc))
                continue

            analyzer = ModuleAnalyzer(mod, py_file)
            analyzer.visit(tree)

            # Collect nodes/edges
            for n in analyzer.nodes:
                nodes.setdefault(n.id, n)
            edges.extend(analyzer.edges)

            module_indexes[mod] = analyzer.def_index
            per_module_calls[mod] = analyzer.calls_raw

            # IMPORTS edges
            for imp in sorted(set(analyzer.imports)):
                target_id: str
                if imp.startswith("didactic_engine") or imp.startswith("music_etl"):
                    target_id = f"module:{imp}"
                    nodes.setdefault(target_id, Node(id=target_id, type="module", name=imp))
                else:
                    # record as external package-ish target
                    root_name = imp.split(".")[0]
                    target_id = f"package:{root_name}"
                    nodes.setdefault(target_id, Node(id=target_id, type="package", name=root_name))

                edges.append(Edge(source=module_id, target=target_id, type="IMPORTS", detail=imp))

            # INHERITS edges (best-effort)
            for class_id, base_expr in analyzer.inherits_raw:
                # If base is defined in the same module by simple name, link it.
                base_id = module_indexes[mod].classes.get(base_expr)
                if base_id:
                    edges.append(Edge(source=class_id, target=base_id, type="INHERITS"))
                else:
                    # fallback: represent the base as a reference node
                    ref_id = f"ref:class:{base_expr}"
                    nodes.setdefault(ref_id, Node(id=ref_id, type="ref", name=base_expr))
                    edges.append(Edge(source=class_id, target=ref_id, type="INHERITS", detail=base_expr))

    # Resolve CALLS edges (best-effort intra-module)
    for mod, call_pairs in per_module_calls.items():
        defs = module_indexes.get(mod)
        if not defs:
            continue

        for caller_id, callee_expr in call_pairs:
            # Very conservative resolution:
            # - If callee is a simple name and matches a function in this module, link it.
            # - If callee is "self.method" and method exists on the current class, link it.
            # - Otherwise create a ref node.
            target_id: Optional[str] = None

            if callee_expr in defs.functions:
                target_id = defs.functions[callee_expr]
            elif callee_expr.startswith("self."):
                meth_name = callee_expr.split(".", 1)[1]
                # Attempt to infer class from caller ID if it is a method
                if caller_id.startswith("method:"):
                    # caller name: module.Class.method
                    caller_name = caller_id.split(":", 1)[1]
                    parts = caller_name.split(".")
                    if len(parts) >= 3:
                        cls = parts[-2]
                        target_id = defs.methods.get((cls, meth_name))

            if target_id is None:
                ref_id = f"ref:call:{callee_expr}"
                nodes.setdefault(ref_id, Node(id=ref_id, type="ref", name=callee_expr))
                target_id = ref_id

            edges.append(Edge(source=caller_id, target=target_id, type="CALLS", detail=callee_expr))

    if report_syntax_errors and syntax_errors:
        print(
            f"Skipped {len(syntax_errors)} file(s) due to SyntaxError.",
            file=sys.stderr,
        )
        for path, exc in syntax_errors[:5]:
            rel_path = path.relative_to(REPO_ROOT)
            print(f"  - {rel_path}: {exc}", file=sys.stderr)
        if len(syntax_errors) > 5:
            print(f"  ... {len(syntax_errors) - 5} more", file=sys.stderr)

    return list(nodes.values()), edges


def write_json(nodes: List[Node], edges: List[Edge], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repo_root": str(REPO_ROOT),
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        "nodes": [asdict(n) for n in sorted(nodes, key=lambda x: (x.type, x.name))],
        "edges": [asdict(e) for e in edges],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_graphml(nodes: List[Node], edges: List[Edge], out_path: Path) -> None:
    """Write GraphML for use in tools like Gephi."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    graphml = ET.Element("graphml", attrib={
        "xmlns": "http://graphml.graphdrawing.org/xmlns",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
    })

    # Define keys
    def key(id_: str, for_: str, attr_name: str, attr_type: str) -> None:
        ET.SubElement(graphml, "key", attrib={
            "id": id_,
            "for": for_,
            "attr.name": attr_name,
            "attr.type": attr_type,
        })

    key("d0", "node", "type", "string")
    key("d1", "node", "name", "string")
    key("d2", "node", "file", "string")
    key("d3", "node", "lineno", "int")
    key("d4", "edge", "type", "string")
    key("d5", "edge", "detail", "string")

    graph = ET.SubElement(graphml, "graph", attrib={"id": "G", "edgedefault": "directed"})

    for n in nodes:
        node_el = ET.SubElement(graph, "node", attrib={"id": n.id})
        ET.SubElement(node_el, "data", attrib={"key": "d0"}).text = n.type
        ET.SubElement(node_el, "data", attrib={"key": "d1"}).text = n.name
        ET.SubElement(node_el, "data", attrib={"key": "d2"}).text = n.file
        ET.SubElement(node_el, "data", attrib={"key": "d3"}).text = str(int(n.lineno or 0))

    for i, e in enumerate(edges):
        edge_el = ET.SubElement(graph, "edge", attrib={"id": f"e{i}", "source": e.source, "target": e.target})
        ET.SubElement(edge_el, "data", attrib={"key": "d4"}).text = e.type
        ET.SubElement(edge_el, "data", attrib={"key": "d5"}).text = e.detail

    tree = ET.ElementTree(graphml)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main(argv: List[str]) -> int:
    out_dir = DEFAULT_OUT_DIR

    source_roots = [REPO_ROOT / "src", REPO_ROOT / "music_etl" / "src"]
    nodes, edges = build_graph(source_roots, report_syntax_errors=True)

    write_json(nodes, edges, out_dir / "knowledge_graph.json")
    write_graphml(nodes, edges, out_dir / "knowledge_graph.graphml")

    print(f"Wrote {len(nodes)} nodes and {len(edges)} edges to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
