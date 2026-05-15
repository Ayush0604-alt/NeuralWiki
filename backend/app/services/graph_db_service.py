"""
Graph Database Service
----------------------
Persistent in-process knowledge graph using NetworkX.
Replaces the flat list store in knowledge_graph_service.py with a
real directed multigraph that supports:

  • Multi-hop traversal  (BFS / DFS up to N hops)
  • Shortest-path queries
  • Neighbour expansion
  • Entity-centric sub-graph extraction
  • Cross-document relationship discovery
  • Degree / centrality ranking

All data survives for the lifetime of the process (same guarantee as
ChromaDB's in-memory layer between restarts).  Persistence to disk can
be added trivially via networkx.readwrite.gpickle.
"""

from __future__ import annotations

import logging
import json
import os
from collections import deque
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

# ── Global directed multigraph ─────────────────────────────────────────────────
_G: nx.MultiDiGraph = nx.MultiDiGraph()

# Path for optional disk persistence
_PERSIST_PATH = os.getenv("GRAPH_PERSIST_PATH", "chroma_db/knowledge_graph.json")


# ── Persistence helpers ────────────────────────────────────────────────────────

def _save_graph() -> None:
    """Serialize graph to JSON on disk."""
    try:
        os.makedirs(os.path.dirname(_PERSIST_PATH), exist_ok=True)
        data = nx.node_link_data(_G)
        with open(_PERSIST_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning("Graph persistence write failed: %s", e)


def load_graph() -> None:
    """Load graph from disk if it exists (call at startup)."""
    global _G
    if not os.path.exists(_PERSIST_PATH):
        return
    try:
        with open(_PERSIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _G = nx.node_link_graph(data, directed=True, multigraph=True)
        logger.info("Loaded persisted graph: %d nodes, %d edges", _G.number_of_nodes(), _G.number_of_edges())
    except Exception as e:
        logger.warning("Graph persistence load failed (starting fresh): %s", e)
        _G = nx.MultiDiGraph()


# ── Write operations ───────────────────────────────────────────────────────────

def add_document_graph(
    doc_id: str,
    center: dict,
    clusters: dict[str, list[dict]],
    relationships: list[dict],
    doc_type: str = "general",
) -> None:
    """
    Ingest one document's extracted graph into the global graph.

    Parameters
    ----------
    doc_id      : filename used as provenance tag on every node/edge
    center      : {"id": str, "label": str, "cluster": str}
    clusters    : {"TECH": [{"id": str, "cluster": str}, ...], ...}
    relationships : [{"source": str, "target": str, "label": str}, ...]
    doc_type    : document classification string
    """
    # Add / update center node
    cid = center["id"]
    _upsert_node(cid, cluster=center.get("cluster", "CENTER"),
                 label=center.get("label", ""), doc_ids=[doc_id],
                 doc_type=doc_type, is_center=True)

    # Add cluster nodes
    for cluster_name, nodes in clusters.items():
        for node in nodes:
            nid = node["id"]
            _upsert_node(nid, cluster=cluster_name, label=node.get("label", ""),
                         doc_ids=[doc_id], doc_type=doc_type)

    # Add edges
    existing_pairs: set[tuple[str, str, str]] = set()
    for rel in relationships:
        src, tgt, lbl = rel["source"], rel["target"], rel.get("label", "related_to")
        key = (src, tgt, lbl)
        if key in existing_pairs:
            continue
        if not _G.has_node(src) or not _G.has_node(tgt):
            continue
        existing_pairs.add(key)
        # Check if edge already exists with same label
        already = False
        if _G.has_edge(src, tgt):
            for _, edata in _G[src][tgt].items():
                if edata.get("label") == lbl and doc_id in edata.get("doc_ids", []):
                    already = True
                    break
        if not already:
            _G.add_edge(src, tgt, label=lbl, doc_ids=[doc_id], weight=1.0)

    _save_graph()
    logger.info("Graph updated [%s]: nodes=%d, edges=%d",
                doc_id, _G.number_of_nodes(), _G.number_of_edges())


def _upsert_node(node_id: str, **attrs: Any) -> None:
    """Add or merge a node. doc_ids list is accumulated across calls."""
    if _G.has_node(node_id):
        existing = _G.nodes[node_id]
        # Merge doc_ids
        existing_docs = existing.get("doc_ids", [])
        new_docs = attrs.pop("doc_ids", [])
        merged = list(set(existing_docs + new_docs))
        _G.nodes[node_id].update(attrs)
        _G.nodes[node_id]["doc_ids"] = merged
    else:
        _G.add_node(node_id, **attrs)


# ── Read / Query operations ────────────────────────────────────────────────────

def get_neighbors(entity: str, max_hops: int = 2) -> list[dict]:
    """
    BFS expansion from *entity* up to *max_hops*.
    Returns list of {node, cluster, distance, path, relationships}.
    """
    if not _G.has_node(entity):
        return []

    visited: dict[str, int] = {entity: 0}
    queue: deque[tuple[str, int, list[str]]] = deque([(entity, 0, [entity])])
    results: list[dict] = []

    while queue:
        current, dist, path = queue.popleft()
        if dist >= max_hops:
            continue
        # Outgoing neighbours
        for nbr in _G.successors(current):
            if nbr not in visited:
                visited[nbr] = dist + 1
                new_path = path + [nbr]
                edge_labels = [
                    edata.get("label", "→")
                    for _, edata in _G[current][nbr].items()
                ]
                results.append({
                    "node": nbr,
                    "cluster": _G.nodes[nbr].get("cluster", "MISC"),
                    "distance": dist + 1,
                    "path": new_path,
                    "via": edge_labels,
                })
                queue.append((nbr, dist + 1, new_path))
        # Incoming neighbours (reverse edges)
        for nbr in _G.predecessors(current):
            if nbr not in visited:
                visited[nbr] = dist + 1
                new_path = path + [nbr]
                edge_labels = [
                    edata.get("label", "←")
                    for _, edata in _G[nbr][current].items()
                ]
                results.append({
                    "node": nbr,
                    "cluster": _G.nodes[nbr].get("cluster", "MISC"),
                    "distance": dist + 1,
                    "path": new_path,
                    "via": edge_labels,
                })
                queue.append((nbr, dist + 1, new_path))

    return sorted(results, key=lambda x: x["distance"])


def find_paths(source: str, target: str, max_hops: int = 4) -> list[list[str]]:
    """Find all simple paths between two nodes (up to max_hops length)."""
    if not _G.has_node(source) or not _G.has_node(target):
        return []
    try:
        paths = list(nx.all_simple_paths(
            _G.to_undirected(), source, target, cutoff=max_hops
        ))
        return sorted(paths, key=len)[:5]  # top-5 shortest
    except nx.NetworkXError:
        return []


def multi_hop_query(seed_entities: list[str], max_hops: int = 2) -> dict:
    """
    Given a list of seed entities extracted from a query, expand their
    neighbourhoods and return a structured context dict for the LLM.

    Returns
    -------
    {
      "entities_found": [...],
      "graph_context": "human-readable relationship summary",
      "paths": [...],      # cross-entity paths if multiple seeds
      "subgraph_nodes": [...],
    }
    """
    found_seeds = [e for e in seed_entities if _G.has_node(e)]
    if not found_seeds:
        return {"entities_found": [], "graph_context": "", "paths": [], "subgraph_nodes": []}

    all_neighbours: dict[str, list[dict]] = {}
    for seed in found_seeds:
        all_neighbours[seed] = get_neighbors(seed, max_hops=max_hops)

    # Cross-entity paths
    paths = []
    if len(found_seeds) >= 2:
        for i, s1 in enumerate(found_seeds):
            for s2 in found_seeds[i + 1:]:
                ps = find_paths(s1, s2, max_hops=max_hops + 1)
                paths.extend(ps)

    # Build human-readable graph context
    lines: list[str] = ["=== Knowledge Graph Context ==="]
    for seed in found_seeds:
        nbrs = all_neighbours[seed]
        if not nbrs:
            continue
        lines.append(f"\n[{seed}] connects to:")
        for n in nbrs[:12]:  # cap per seed
            via = ", ".join(n["via"][:2])
            lines.append(f"  {'→' * n['distance']} {n['node']} (via: {via})")

    if paths:
        lines.append("\n[Cross-entity paths]")
        for p in paths[:4]:
            lines.append("  " + " → ".join(p))

    # Collect all subgraph node ids
    subgraph_nodes = list(found_seeds)
    for nbr_list in all_neighbours.values():
        subgraph_nodes.extend(n["node"] for n in nbr_list)
    subgraph_nodes = list(dict.fromkeys(subgraph_nodes))  # deduplicate, preserve order

    return {
        "entities_found": found_seeds,
        "graph_context": "\n".join(lines),
        "paths": paths,
        "subgraph_nodes": subgraph_nodes,
    }


def get_entity_docs(entity: str) -> list[str]:
    """Return list of doc_ids that mention this entity."""
    if not _G.has_node(entity):
        return []
    return _G.nodes[entity].get("doc_ids", [])


def get_top_entities(n: int = 20) -> list[dict]:
    """Return top-n entities by degree (most connected)."""
    if _G.number_of_nodes() == 0:
        return []
    ranked = sorted(
        _G.nodes(data=True),
        key=lambda x: _G.degree(x[0]),
        reverse=True,
    )
    return [
        {
            "id": nid,
            "cluster": data.get("cluster", "MISC"),
            "degree": _G.degree(nid),
            "doc_ids": data.get("doc_ids", []),
        }
        for nid, data in ranked[:n]
    ]


def search_nodes_by_text(query: str, top_k: int = 10) -> list[dict]:
    """
    Fuzzy text search over node IDs (case-insensitive substring match).
    Returns list of matching node dicts sorted by degree desc.
    """
    q = query.lower()
    matches = []
    for nid, data in _G.nodes(data=True):
        if q in nid.lower():
            matches.append({
                "id": nid,
                "cluster": data.get("cluster", "MISC"),
                "degree": _G.degree(nid),
                "doc_ids": data.get("doc_ids", []),
            })
    return sorted(matches, key=lambda x: x["degree"], reverse=True)[:top_k]


def extract_query_entities(query: str) -> list[str]:
    """
    Lightweight entity matching: find graph nodes mentioned in the query.
    Tries exact match first, then partial match.
    """
    q_lower = query.lower()
    exact: list[str] = []
    partial: list[str] = []

    for nid in _G.nodes():
        nid_lower = nid.lower()
        if nid_lower in q_lower:
            exact.append(nid)
        elif len(nid) > 3 and any(word in q_lower for word in nid_lower.split()):
            partial.append(nid)

    # Prefer exact, fall back to partial, rank by degree
    combined = list(dict.fromkeys(exact + partial))
    return sorted(combined, key=lambda x: _G.degree(x), reverse=True)[:8]


def get_graph_stats() -> dict:
    """Summary statistics for the graph."""
    if _G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "components": 0, "density": 0.0}
    undirected = _G.to_undirected()
    components = nx.number_connected_components(undirected)
    density = nx.density(_G)
    return {
        "nodes": _G.number_of_nodes(),
        "edges": _G.number_of_edges(),
        "components": components,
        "density": round(density, 6),
        "top_entities": get_top_entities(5),
    }


def get_full_graph_for_frontend() -> list[dict]:
    """
    Export graph in the format expected by the existing frontend
    KnowledgeGraph component (list of segment dicts).
    Groups nodes back into per-document segments.
    """
    if _G.number_of_nodes() == 0:
        return []

    # Group nodes by their first doc_id
    doc_segments: dict[str, dict] = {}

    for nid, data in _G.nodes(data=True):
        cluster = data.get("cluster", "MISC")
        doc_ids = data.get("doc_ids", ["unknown"])
        primary_doc = doc_ids[0] if doc_ids else "unknown"

        if primary_doc not in doc_segments:
            doc_segments[primary_doc] = {
                "center": None,
                "clusters": {},
                "relationships": [],
                "entities": [],
            }

        seg = doc_segments[primary_doc]

        if cluster == "CENTER" or data.get("is_center"):
            seg["center"] = {"id": nid, "label": data.get("label", ""), "cluster": "CENTER"}
        else:
            seg["clusters"].setdefault(cluster, [])
            if not any(n["id"] == nid for n in seg["clusters"][cluster]):
                seg["clusters"][cluster].append({"id": nid, "cluster": cluster})

        seg["entities"].append({"text": nid, "label": data.get("label", "")})

    # Add edges to their primary doc segment
    for src, tgt, edata in _G.edges(data=True):
        doc_ids = edata.get("doc_ids", ["unknown"])
        primary_doc = doc_ids[0] if doc_ids else "unknown"
        if primary_doc in doc_segments:
            doc_segments[primary_doc]["relationships"].append({
                "source": src,
                "target": tgt,
                "label": edata.get("label", "related_to"),
            })

    # Ensure every segment has a center node
    results = []
    for doc_id, seg in doc_segments.items():
        if seg["center"] is None:
            # Pick highest-degree node as center
            doc_nodes = [
                nid for nid, data in _G.nodes(data=True)
                if doc_id in data.get("doc_ids", [])
            ]
            if doc_nodes:
                best = max(doc_nodes, key=lambda x: _G.degree(x))
                seg["center"] = {"id": best, "label": "", "cluster": "CENTER"}
        if seg["center"]:
            results.append(seg)

    return results


def remove_document_graph(doc_id: str) -> int:
    """Remove all nodes and edges that belong exclusively to doc_id."""
    nodes_to_remove = []
    for nid, data in list(_G.nodes(data=True)):
        doc_ids = data.get("doc_ids", [])
        if doc_id in doc_ids:
            doc_ids.remove(doc_id)
            if not doc_ids:
                nodes_to_remove.append(nid)
            else:
                _G.nodes[nid]["doc_ids"] = doc_ids

    # Also clean edges
    edges_to_remove = []
    for src, tgt, key, edata in list(_G.edges(data=True, keys=True)):
        doc_ids = edata.get("doc_ids", [])
        if doc_id in doc_ids:
            doc_ids.remove(doc_id)
            if not doc_ids:
                edges_to_remove.append((src, tgt, key))
            else:
                _G[src][tgt][key]["doc_ids"] = doc_ids

    for src, tgt, key in edges_to_remove:
        _G.remove_edge(src, tgt, key=key)
    for nid in nodes_to_remove:
        _G.remove_node(nid)

    _save_graph()
    logger.info("Removed doc '%s' from graph: %d nodes, %d edges deleted",
                doc_id, len(nodes_to_remove), len(edges_to_remove))
    return len(nodes_to_remove)


def clear_graph() -> None:
    """Wipe the entire graph."""
    _G.clear()
    if os.path.exists(_PERSIST_PATH):
        try:
            os.remove(_PERSIST_PATH)
        except OSError:
            pass
    logger.info("Knowledge graph cleared")