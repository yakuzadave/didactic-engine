# Knowledge Graph (didactic-engine)

This folder contains a generated **knowledge graph** of code entities and relationships.

## What it includes

- Code entities: modules, classes, functions, methods
- Relationships: contains, imports, calls (best-effort), inherits (best-effort)
- External dependencies: parsed from `pyproject.toml`

## How to (re)generate

From repo root:

- `python tools/knowledge_graph/build_knowledge_graph.py`

Outputs are written to:
- `docs/knowledge_graph/output/knowledge_graph.json`
- `docs/knowledge_graph/output/knowledge_graph.graphml`

## JSON schema (high level)

The JSON output has:

- `meta`: generation timestamp and counts
- `nodes`: list of nodes with `id`, `type`, `name`, `file`, `lineno`, `doc`
- `edges`: list of edges with `source`, `target`, `type`, `detail`

## Notes and limitations

- `CALLS` edges are conservative and mostly resolve **intra-module** calls.
  Unresolved calls are represented as `ref:*` nodes.
- `INHERITS` edges are name-only best-effort.
- This graph is intended for navigation, impact analysis, and documentation,
  not as a perfect call graph.

## Visualization

- The GraphML file can be opened with tools such as Gephi.
