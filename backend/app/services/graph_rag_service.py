"""
GraphRAG Service (unified)
--------------------------
Single source of truth for all knowledge-graph operations.

Changes vs original:
  • get_full_graph_for_frontend() now returns BOTH the legacy document-segmented
    format (for backward compatibility) AND a new flat node-link format under
    the "flat" key. KnowledgeGraph.jsx should prefer the flat format.
  • Community detection via NetworkX greedy_modularity_communities runs after
    ingestion and is cached. Each node gains a community_id attribute.
  • get_graph_stats() now includes betweenness centrality top-5 and community count.
  • The "center" concept is preserved for the legacy format but the flat format
    exposes degree + betweenness so the frontend can size nodes properly.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

_G: nx.MultiDiGraph = nx.MultiDiGraph()
_entity_meta: dict[str, dict] = {}
_doc_entities: dict[str, set[str]] = {}
_community_map: dict[str, int] = {}   # node_id → community_id

_PERSIST_PATH = os.getenv("GRAPH_PERSIST_PATH", "chroma_db/knowledge_graph.json")


# ── Persistence ────────────────────────────────────────────────────────────────

def _save() -> None:
    try:
        os.makedirs(os.path.dirname(_PERSIST_PATH), exist_ok=True)
        payload = {
            "graph":       nx.node_link_data(_G),
            "meta":        _entity_meta,
            "doc_entities": {k: list(v) for k, v in _doc_entities.items()},
            "communities": _community_map,
        }
        with open(_PERSIST_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as exc:
        logger.warning("Graph persist failed: %s", exc)


def load_graph() -> None:
    global _G, _entity_meta, _doc_entities, _community_map
    if not os.path.exists(_PERSIST_PATH):
        return
    try:
        with open(_PERSIST_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        _G = nx.node_link_graph(payload["graph"], directed=True, multigraph=True)
        _entity_meta = payload.get("meta", {})
        _doc_entities = {k: set(v) for k, v in payload.get("doc_entities", {}).items()}
        _community_map = payload.get("communities", {})
        logger.info(
            "Graph loaded: %d nodes, %d edges, %d docs",
            _G.number_of_nodes(), _G.number_of_edges(), len(_doc_entities),
        )
    except Exception as exc:
        logger.warning("Graph load failed (starting fresh): %s", exc)
        _G = nx.MultiDiGraph()
        _entity_meta = {}
        _doc_entities = {}
        _community_map = {}


# ── Community detection ────────────────────────────────────────────────────────

def _recompute_communities() -> None:
    """Run greedy modularity community detection on the undirected projection."""
    global _community_map
    if _G.number_of_nodes() < 3:
        _community_map = {n: 0 for n in _G.nodes()}
        return
    try:
        undirected = _G.to_undirected()
        communities = list(nx.community.greedy_modularity_communities(undirected))
        _community_map = {}
        for cid, members in enumerate(communities):
            for node in members:
                _community_map[node] = cid
        logger.info("Community detection: %d communities for %d nodes", len(communities), _G.number_of_nodes())
    except Exception as exc:
        logger.warning("Community detection failed: %s", exc)
        _community_map = {n: 0 for n in _G.nodes()}


# ── Write operations ───────────────────────────────────────────────────────────

def ingest_document_graph(
    doc_id: str,
    center: dict,
    clusters: dict[str, list[dict]],
    relationships: list[dict],
    doc_type: str = "general",
) -> None:
    _doc_entities.setdefault(doc_id, set())

    cid = center.get("id", "")
    if cid:
        _upsert_node(cid, cluster=center.get("cluster", "CENTER"),
                     label=center.get("label", ""), doc_id=doc_id,
                     doc_type=doc_type, is_center=True)
        _doc_entities[doc_id].add(cid)

    for cluster_name, nodes in clusters.items():
        for node in (nodes or []):
            nid = node.get("id", "")
            if not nid:
                continue
            _upsert_node(nid, cluster=cluster_name, label=node.get("label", ""),
                         doc_id=doc_id, doc_type=doc_type)
            _doc_entities[doc_id].add(nid)

    added_keys: set[tuple[str, str, str]] = set()
    for rel in relationships:
        src = rel.get("source", "").strip()
        tgt = rel.get("target", "").strip()
        lbl = (rel.get("label", "related_to") or "related_to").strip()

        if not src or not tgt:
            continue
        if not _G.has_node(src) or not _G.has_node(tgt):
            continue

        key = (src, tgt, lbl)
        if key in added_keys:
            continue

        already = False
        if _G.has_edge(src, tgt):
            for _, edata in _G[src][tgt].items():
                if edata.get("label") == lbl and doc_id in edata.get("doc_ids", []):
                    already = True
                    break
        if not already:
            added_keys.add(key)
            weight = rel.get("weight", 1.0)
            _G.add_edge(src, tgt, label=lbl, doc_ids=[doc_id], weight=weight)

    # Recompute communities after each ingestion
    _recompute_communities()

    _save()
    logger.info(
        "Graph ingested [%s | %s]: +%d nodes, +%d edges → total %d nodes, %d edges",
        doc_id, doc_type,
        len(_doc_entities[doc_id]),
        len(added_keys),
        _G.number_of_nodes(),
        _G.number_of_edges(),
    )


def _upsert_node(node_id: str, doc_id: str, **attrs: Any) -> None:
    if _G.has_node(node_id):
        existing = _G.nodes[node_id]
        doc_ids = list(set(existing.get("doc_ids", []) + [doc_id]))
        existing.update(attrs)
        existing["doc_ids"] = doc_ids
        meta = _entity_meta.get(node_id, {})
        meta["freq"] = meta.get("freq", 0) + 1
        meta["doc_ids"] = doc_ids
        _entity_meta[node_id] = meta
    else:
        _G.add_node(node_id, doc_ids=[doc_id], **attrs)
        _entity_meta[node_id] = {
            "cluster": attrs.get("cluster", "MISC"),
            "freq": 1,
            "doc_ids": [doc_id],
            "label": attrs.get("label", ""),
        }


# ── Query operations ───────────────────────────────────────────────────────────

def find_query_entities(query: str) -> list[str]:
    if _G.number_of_nodes() == 0:
        return []

    q_lower = query.lower()
    exact: list[str] = []
    partial: list[str] = []

    for nid in _G.nodes():
        nid_lower = nid.lower()
        if len(nid) < 2:
            continue
        if nid_lower in q_lower:
            exact.append(nid)
        elif len(nid) > 3:
            words = nid_lower.split()
            if any(w in q_lower for w in words if len(w) > 3):
                partial.append(nid)

    combined = list(dict.fromkeys(exact + partial))
    return sorted(combined, key=lambda x: _G.degree(x), reverse=True)[:10]


def get_neighbors(entity: str, max_hops: int = 2, max_nodes: int = 60) -> list[dict]:
    if not _G.has_node(entity):
        return []

    visited: dict[str, int] = {entity: 0}
    queue: deque[tuple[str, int, list[str]]] = deque([(entity, 0, [entity])])
    results: list[dict] = []

    while queue and len(visited) < max_nodes:
        current, dist, path = queue.popleft()
        if dist >= max_hops:
            continue

        for nbr in _G.successors(current):
            if nbr not in visited:
                visited[nbr] = dist + 1
                edge_labels = [edata.get("label", "→") for _, edata in _G[current][nbr].items()]
                results.append({
                    "node":      nbr,
                    "cluster":   _G.nodes[nbr].get("cluster", "MISC"),
                    "distance":  dist + 1,
                    "path":      path + [nbr],
                    "via":       edge_labels,
                    "direction": "out",
                })
                queue.append((nbr, dist + 1, path + [nbr]))

        for nbr in _G.predecessors(current):
            if nbr not in visited:
                visited[nbr] = dist + 1
                edge_labels = [edata.get("label", "←") for _, edata in _G[nbr][current].items()]
                results.append({
                    "node":      nbr,
                    "cluster":   _G.nodes[nbr].get("cluster", "MISC"),
                    "distance":  dist + 1,
                    "path":      [nbr] + path,
                    "via":       edge_labels,
                    "direction": "in",
                })
                queue.append((nbr, dist + 1, path + [nbr]))

    return sorted(results, key=lambda x: x["distance"])


def find_paths(source: str, target: str, max_hops: int = 4) -> list[list[str]]:
    if not (_G.has_node(source) and _G.has_node(target)):
        return []
    try:
        paths = list(nx.all_simple_paths(_G.to_undirected(), source, target, cutoff=max_hops))
        return sorted(paths, key=len)[:5]
    except nx.NetworkXError:
        return []


# ── Context builders ───────────────────────────────────────────────────────────

def build_graph_context(query: str, max_triples: int = 40) -> str:
    if _G.number_of_nodes() == 0:
        return ""

    seed_entities = find_query_entities(query)
    if not seed_entities:
        seed_entities = [nid for nid, _ in sorted(
            _G.degree(), key=lambda x: x[1], reverse=True
        )[:5]]

    if not seed_entities:
        return ""

    lines: list[str] = ["## Knowledge Graph Context\n"]
    lines.append(f"**Query entities:** {', '.join(seed_entities[:5])}\n")

    all_edges_seen: set[str] = set()
    all_paths: list[str] = []
    all_nodes: list[dict] = []
    nodes_seen: set[str] = set()
    triple_count = 0

    for seed in seed_entities[:4]:
        neighbours = get_neighbors(seed, max_hops=2, max_nodes=40)
        for nbr_info in neighbours:
            nid = nbr_info["node"]
            if nid not in nodes_seen:
                nodes_seen.add(nid)
                meta = _entity_meta.get(nid, {})
                all_nodes.append({
                    "id":       nid,
                    "cluster":  nbr_info["cluster"],
                    "freq":     meta.get("freq", 1),
                    "distance": nbr_info["distance"],
                })

            path = nbr_info["path"]
            vias = nbr_info["via"]
            if len(path) >= 2 and vias and triple_count < max_triples:
                triple = f"{path[-2]} → [{vias[0]}] → {path[-1]}"
                if triple not in all_edges_seen:
                    all_edges_seen.add(triple)
                    triple_count += 1

            if nbr_info["distance"] >= 2:
                all_paths.append(" → ".join(nbr_info["path"]))

    if len(seed_entities) >= 2:
        for i, s1 in enumerate(seed_entities[:3]):
            for s2 in seed_entities[i + 1: 4]:
                for p in find_paths(s1, s2, max_hops=3)[:2]:
                    all_paths.append(" → ".join(p))

    if all_edges_seen:
        lines.append("**Entity Relationships (triples):**")
        for triple in sorted(all_edges_seen)[:max_triples]:
            lines.append(f"  - {triple}")

    if all_paths:
        lines.append("\n**Multi-hop Reasoning Paths:**")
        for p in list(dict.fromkeys(all_paths))[:6]:
            lines.append(f"  • {p}")

    important = sorted(all_nodes, key=lambda n: n.get("freq", 1), reverse=True)[:8]
    if important:
        lines.append("\n**Key Entities in Context:**")
        for node in important:
            lines.append(
                f"  - **{node['id']}** ({node.get('cluster', '')}, freq={node.get('freq', 1)})"
            )

    return "\n".join(lines)


def enrich_retrieval_context(query: str, vector_chunks: list[dict]) -> str:
    vector_parts = [
        f"[Chunk {i + 1} | Source: {c['source']}]\n{c['content']}"
        for i, c in enumerate(vector_chunks)
    ]
    vector_context = "\n\n".join(vector_parts)
    graph_context = build_graph_context(query)
    if not graph_context:
        return vector_context
    return f"{vector_context}\n\n{graph_context}"


# ── Analytics ──────────────────────────────────────────────────────────────────

def get_graph_stats() -> dict:
    if _G.number_of_nodes() == 0:
        return {
            "nodes": 0, "edges": 0, "components": 0,
            "density": 0.0, "documents": 0,
            "avg_degree": 0.0, "top_entities": [],
            "communities": 0,
        }

    undirected = _G.to_undirected()
    try:
        components = nx.number_connected_components(undirected)
    except Exception:
        components = 0

    try:
        centrality = nx.degree_centrality(_G)
        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    except Exception:
        top_central = []

    num_communities = len(set(_community_map.values())) if _community_map else 0

    return {
        "nodes":        _G.number_of_nodes(),
        "edges":        _G.number_of_edges(),
        "components":   components,
        "density":      round(nx.density(_G), 6),
        "documents":    len(_doc_entities),
        "avg_degree":   round(
            sum(d for _, d in _G.degree()) / max(_G.number_of_nodes(), 1), 2
        ),
        "top_entities": [
            {"id": nid, "centrality": round(c, 4)}
            for nid, c in top_central
        ],
        "communities":  num_communities,
    }


def get_top_entities(n: int = 20) -> list[dict]:
    if _G.number_of_nodes() == 0:
        return []
    ranked = sorted(_G.nodes(data=True), key=lambda x: _G.degree(x[0]), reverse=True)
    return [
        {
            "id":        nid,
            "cluster":   data.get("cluster", "MISC"),
            "degree":    _G.degree(nid),
            "freq":      _entity_meta.get(nid, {}).get("freq", 1),
            "doc_ids":   data.get("doc_ids", []),
            "community": _community_map.get(nid, 0),
        }
        for nid, data in ranked[:n]
    ]


def query_related_entities(query: str, top_k: int = 10) -> list[dict]:
    seed = find_query_entities(query)
    all_nodes: list[dict] = []
    nodes_seen: set[str] = set()

    for s in seed[:4]:
        for nbr in get_neighbors(s, max_hops=2, max_nodes=80):
            nid = nbr["node"]
            if nid not in nodes_seen:
                nodes_seen.add(nid)
                meta = _entity_meta.get(nid, {})
                all_nodes.append({
                    "id":       nid,
                    "cluster":  nbr["cluster"],
                    "freq":     meta.get("freq", 1),
                    "distance": nbr["distance"],
                })

    try:
        centrality = nx.degree_centrality(_G)
    except Exception:
        centrality = {}

    scored = []
    for node in all_nodes:
        nid = node["id"]
        score = node.get("freq", 1) * (1 + centrality.get(nid, 0) * 10)
        scored.append({**node, "relevance_score": round(score, 3)})

    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:top_k]


# ── Frontend export ────────────────────────────────────────────────────────────

def get_full_graph_for_frontend() -> list[dict]:
    """
    Export graph in two formats:

    1. Legacy document-segmented format (each entry = one source doc):
       [{center, clusters, relationships, entities}, ...]
       Kept for backward compatibility.

    2. Flat node-link format appended as the last list entry under key "_flat":
       {"_flat": True, "nodes": [...], "links": [...]}
       Nodes carry: id, cluster, community_id, degree, freq, doc_ids
       Links carry: source, target, label, weight

    KnowledgeGraph.jsx should check for the _flat entry and prefer it.
    Cross-document relationships are only visible in the flat format.
    """
    if _G.number_of_nodes() == 0:
        return []

    # ── Compute degree for all nodes ──────────────────────────────────────────
    degree_map = dict(_G.degree())

    # ── Build flat node list ───────────────────────────────────────────────────
    flat_nodes = []
    for nid, data in _G.nodes(data=True):
        meta = _entity_meta.get(nid, {})
        flat_nodes.append({
            "id":          nid,
            "cluster":     data.get("cluster", "MISC"),
            "community_id": _community_map.get(nid, 0),
            "degree":      degree_map.get(nid, 0),
            "freq":        meta.get("freq", 1),
            "doc_ids":     data.get("doc_ids", []),
            "is_center":   bool(data.get("is_center", False)),
        })

    # ── Build flat link list ───────────────────────────────────────────────────
    flat_links = []
    seen_pairs: set[tuple] = set()
    for src, tgt, edata in _G.edges(data=True):
        key = (src, tgt, edata.get("label", ""))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        flat_links.append({
            "source": src,
            "target": tgt,
            "label":  edata.get("label", "related_to"),
            "weight": edata.get("weight", 1.0),
        })

    # ── Legacy document-segmented format ──────────────────────────────────────
    doc_segments: dict[str, dict] = {}

    for nid, data in _G.nodes(data=True):
        cluster = data.get("cluster", "MISC")
        doc_ids_list = data.get("doc_ids", ["unknown"])
        primary_doc = doc_ids_list[0] if doc_ids_list else "unknown"

        seg = doc_segments.setdefault(primary_doc, {
            "center": None, "clusters": {}, "relationships": [], "entities": [],
        })

        if cluster == "CENTER" or data.get("is_center"):
            seg["center"] = {"id": nid, "label": data.get("label", ""), "cluster": "CENTER"}
        else:
            seg["clusters"].setdefault(cluster, [])
            if not any(n["id"] == nid for n in seg["clusters"][cluster]):
                seg["clusters"][cluster].append({"id": nid, "cluster": cluster})

        seg["entities"].append({"text": nid, "label": data.get("label", "")})

    for src, tgt, edata in _G.edges(data=True):
        doc_ids_list = edata.get("doc_ids", ["unknown"])
        primary_doc = doc_ids_list[0] if doc_ids_list else "unknown"
        if primary_doc in doc_segments:
            doc_segments[primary_doc]["relationships"].append({
                "source": src, "target": tgt,
                "label":  edata.get("label", "related_to"),
                "weight": edata.get("weight", 1.0),
            })

    results = []
    for doc_id, seg in doc_segments.items():
        if seg["center"] is None:
            doc_nodes = [
                nid for nid, data in _G.nodes(data=True)
                if doc_id in data.get("doc_ids", [])
            ]
            if doc_nodes:
                best = max(doc_nodes, key=lambda x: _G.degree(x))
                seg["center"] = {"id": best, "label": "", "cluster": "CENTER"}
        if seg["center"]:
            results.append(seg)

    # Append the flat format as a sentinel entry
    results.append({
        "_flat":  True,
        "nodes":  flat_nodes,
        "links":  flat_links,
        "num_communities": len(set(_community_map.values())) if _community_map else 0,
    })

    return results


# ── Document removal ───────────────────────────────────────────────────────────

def remove_document(doc_id: str) -> int:
    nodes_removed = 0
    edges_to_remove: list[tuple] = []

    for nid, data in list(_G.nodes(data=True)):
        doc_ids = list(data.get("doc_ids", []))
        if doc_id in doc_ids:
            doc_ids.remove(doc_id)
            if not doc_ids:
                nodes_removed += 1
            else:
                _G.nodes[nid]["doc_ids"] = doc_ids
                meta = _entity_meta.get(nid, {})
                meta["doc_ids"] = doc_ids
                meta["freq"] = max(1, meta.get("freq", 1) - 1)
                _entity_meta[nid] = meta

    for src, tgt, key, edata in list(_G.edges(data=True, keys=True)):
        doc_ids = list(edata.get("doc_ids", []))
        if doc_id in doc_ids:
            doc_ids.remove(doc_id)
            if not doc_ids:
                edges_to_remove.append((src, tgt, key))
            else:
                _G[src][tgt][key]["doc_ids"] = doc_ids

    for src, tgt, key in edges_to_remove:
        _G.remove_edge(src, tgt, key=key)

    for nid in [n for n, d in list(_G.nodes(data=True)) if not d.get("doc_ids")]:
        _G.remove_node(nid)
        _entity_meta.pop(nid, None)

    _doc_entities.pop(doc_id, None)
    _recompute_communities()
    _save()

    logger.info(
        "Graph: removed doc '%s' → %d nodes deleted, %d edges deleted",
        doc_id, nodes_removed, len(edges_to_remove),
    )
    return nodes_removed


def clear_graph() -> None:
    global _G, _entity_meta, _doc_entities, _community_map
    _G.clear()
    _entity_meta.clear()
    _doc_entities.clear()
    _community_map.clear()
    if os.path.exists(_PERSIST_PATH):
        try:
            os.remove(_PERSIST_PATH)
        except OSError:
            pass
    logger.info("Knowledge graph cleared")