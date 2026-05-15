"""
GraphRAG Service
----------------
Replaces the simple in-memory list with a proper NetworkX knowledge graph.

Features:
  - Persistent NetworkX DiGraph with entity/relationship storage
  - Multi-hop graph traversal (BFS/DFS up to configurable depth)
  - Subgraph extraction around query entities
  - Entity-centric context building
  - Hybrid retrieval: vector results + graph-expanded context
  - Community detection for topic clustering
  - Entity frequency / importance scoring
"""

import logging
import re
from collections import defaultdict, deque
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

# ── Global graph instance ─────────────────────────────────────────────────────
_graph: nx.DiGraph = nx.DiGraph()

# Entity metadata: entity_id → {cluster, label, freq, docs}
_entity_meta: dict[str, dict] = {}

# Source tracking: which docs mention which entities
_doc_entities: dict[str, set] = defaultdict(set)  # filename → {entity_ids}


# ── Graph population ──────────────────────────────────────────────────────────

def build_graph_from_extraction(graph_data: dict, source_filename: str) -> None:
    """
    Ingest entity+relationship data from knowledge_graph_service into NetworkX.
    Called after each document upload.
    """
    global _graph

    center = graph_data.get("center", {})
    center_id = center.get("id", "")
    clusters = graph_data.get("clusters", {})
    relationships = graph_data.get("relationships", [])

    # Add center node
    if center_id:
        _add_or_update_node(center_id, "CENTER", source_filename)

    # Add cluster nodes
    for cluster_name, nodes in clusters.items():
        for node in (nodes or []):
            nid = node.get("id", "")
            if nid:
                _add_or_update_node(nid, cluster_name, source_filename)

    # Add edges
    for rel in relationships:
        src = rel.get("source", "").strip()
        tgt = rel.get("target", "").strip()
        label = rel.get("label", "related_to").strip()
        if src and tgt and _graph.has_node(src) and _graph.has_node(tgt):
            if _graph.has_edge(src, tgt):
                # Accumulate edge labels and weight
                existing = _graph[src][tgt]
                if label not in existing.get("labels", []):
                    existing.setdefault("labels", [existing.get("label", "")]).append(label)
                existing["weight"] = existing.get("weight", 1) + 1
            else:
                _graph.add_edge(src, tgt, label=label, labels=[label], weight=1, source=source_filename)

    # Track doc→entity mapping
    all_node_ids = {center_id} if center_id else set()
    for nodes in clusters.values():
        all_node_ids.update(n.get("id", "") for n in (nodes or []))
    _doc_entities[source_filename].update(all_node_ids)

    logger.info(
        "GraphRAG: ingested '%s' → nodes=%d, edges=%d (total graph: %d nodes, %d edges)",
        source_filename, len(all_node_ids), len(relationships),
        _graph.number_of_nodes(), _graph.number_of_edges(),
    )


def _add_or_update_node(node_id: str, cluster: str, source: str) -> None:
    if _graph.has_node(node_id):
        meta = _entity_meta.get(node_id, {})
        meta["freq"] = meta.get("freq", 0) + 1
        meta["docs"] = list(set(meta.get("docs", []) + [source]))
        _entity_meta[node_id] = meta
        # Update cluster if it was MISC before
        if _graph.nodes[node_id].get("cluster") == "MISC" and cluster != "MISC":
            _graph.nodes[node_id]["cluster"] = cluster
    else:
        _graph.add_node(node_id, cluster=cluster, source=source)
        _entity_meta[node_id] = {"cluster": cluster, "freq": 1, "docs": [source]}


# ── Entity detection in query ─────────────────────────────────────────────────

def find_query_entities(query: str) -> list[str]:
    """
    Find graph nodes that appear in the query text.
    Uses case-insensitive substring matching.
    Returns list of matched node IDs, sorted by length (longer = more specific).
    """
    q_lower = query.lower()
    matched = []
    for node_id in _graph.nodes():
        if len(node_id) >= 3 and node_id.lower() in q_lower:
            matched.append(node_id)
    # Longer matches are more specific
    matched.sort(key=len, reverse=True)
    return matched[:10]  # cap at 10 seed entities


# ── Multi-hop traversal ───────────────────────────────────────────────────────

def get_neighborhood(
    entity_ids: list[str],
    hops: int = 2,
    max_nodes: int = 50,
) -> dict:
    """
    BFS from seed entities up to `hops` hops.
    Returns {nodes: [...], edges: [...], paths: [...]}
    """
    if not entity_ids or _graph.number_of_nodes() == 0:
        return {"nodes": [], "edges": [], "paths": []}

    visited_nodes: set[str] = set()
    visited_edges: list[dict] = []
    paths: list[str] = []
    queue: deque = deque()

    # Seed
    for eid in entity_ids:
        if _graph.has_node(eid):
            queue.append((eid, 0, [eid]))
            visited_nodes.add(eid)

    while queue and len(visited_nodes) < max_nodes:
        current, depth, path = queue.popleft()

        if depth >= hops:
            continue

        # Outgoing edges
        for neighbor in _graph.successors(current):
            edge_data = _graph[current][neighbor]
            label = edge_data.get("label", "related_to")
            edge_entry = {"source": current, "target": neighbor, "label": label}
            if edge_entry not in visited_edges:
                visited_edges.append(edge_entry)

            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                new_path = path + [f"→[{label}]→", neighbor]
                queue.append((neighbor, depth + 1, new_path))
                if depth == hops - 1:
                    paths.append(" ".join(new_path))

        # Incoming edges (reverse traversal, 1 hop only)
        if depth == 0:
            for predecessor in _graph.predecessors(current):
                edge_data = _graph[predecessor][current]
                label = edge_data.get("label", "related_to")
                edge_entry = {"source": predecessor, "target": current, "label": label}
                if edge_entry not in visited_edges:
                    visited_edges.append(edge_entry)
                if predecessor not in visited_nodes:
                    visited_nodes.add(predecessor)
                    queue.append((predecessor, depth + 1, [predecessor, f"→[{label}]→", current]))

    node_details = []
    for nid in visited_nodes:
        meta = _entity_meta.get(nid, {})
        node_details.append({
            "id": nid,
            "cluster": _graph.nodes[nid].get("cluster", "MISC"),
            "freq": meta.get("freq", 1),
            "docs": meta.get("docs", []),
        })

    return {
        "nodes": node_details,
        "edges": visited_edges,
        "paths": paths,
        "seed_entities": entity_ids,
    }


# ── Relationship path finding ─────────────────────────────────────────────────

def find_paths_between(entity_a: str, entity_b: str, max_hops: int = 3) -> list[list[str]]:
    """Find all simple paths between two entities up to max_hops."""
    if not (_graph.has_node(entity_a) and _graph.has_node(entity_b)):
        return []
    try:
        paths = list(nx.all_simple_paths(_graph, entity_a, entity_b, cutoff=max_hops))
        return paths[:5]  # top 5 paths
    except nx.NetworkXError:
        return []


# ── Context building ──────────────────────────────────────────────────────────

def build_graph_context(query: str, max_triples: int = 40) -> str:
    """
    Given a query, extract relevant graph context as structured triples.
    Returns a formatted string ready for LLM injection.
    """
    if _graph.number_of_nodes() == 0:
        return ""

    seed_entities = find_query_entities(query)

    if not seed_entities:
        # Fall back to high-frequency entities
        seed_entities = _get_top_entities(5)

    neighborhood = get_neighborhood(seed_entities, hops=2, max_nodes=60)

    if not neighborhood["edges"] and not neighborhood["nodes"]:
        return ""

    lines = ["## Knowledge Graph Context\n"]
    lines.append(f"**Query entities found:** {', '.join(seed_entities[:5])}\n")

    # Triples
    if neighborhood["edges"]:
        lines.append("**Relationships:**")
        seen = set()
        count = 0
        for edge in neighborhood["edges"]:
            triple = f"{edge['source']} → [{edge['label']}] → {edge['target']}"
            if triple not in seen and count < max_triples:
                seen.add(triple)
                lines.append(f"  - {triple}")
                count += 1

    # Reasoning paths (multi-hop)
    if neighborhood["paths"]:
        lines.append("\n**Multi-hop reasoning paths:**")
        for path in neighborhood["paths"][:5]:
            lines.append(f"  • {path}")

    # Entity summaries
    important = sorted(
        neighborhood["nodes"],
        key=lambda n: n.get("freq", 1),
        reverse=True,
    )[:8]
    if important:
        lines.append("\n**Key entities in context:**")
        for node in important:
            cluster = node.get("cluster", "")
            freq = node.get("freq", 1)
            lines.append(f"  - **{node['id']}** ({cluster}, mentioned {freq}x)")

    return "\n".join(lines)


def _get_top_entities(n: int = 5) -> list[str]:
    """Return top-N entities by frequency."""
    scored = [(nid, meta.get("freq", 1)) for nid, meta in _entity_meta.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:n]]


# ── Hybrid retrieval context ──────────────────────────────────────────────────

def enrich_retrieval_context(query: str, vector_chunks: list[dict]) -> str:
    """
    Takes vector search results and enriches them with graph context.
    Returns a combined context string for the LLM.
    """
    # Build vector context
    vector_context_parts = []
    for i, chunk in enumerate(vector_chunks, 1):
        vector_context_parts.append(
            f"[Chunk {i} | Source: {chunk['source']}]\n{chunk['content']}"
        )
    vector_context = "\n\n".join(vector_context_parts)

    # Build graph context
    graph_context = build_graph_context(query)

    if not graph_context:
        return vector_context

    return f"{vector_context}\n\n{graph_context}"


# ── Graph analytics ───────────────────────────────────────────────────────────

def get_graph_stats() -> dict:
    """Return graph statistics for the API."""
    if _graph.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "entities": 0, "documents": 0}

    # Degree centrality for top entities
    try:
        centrality = nx.degree_centrality(_graph)
        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    except Exception:
        top_central = []

    # Connected components (undirected view)
    try:
        undirected = _graph.to_undirected()
        components = nx.number_connected_components(undirected)
    except Exception:
        components = 0

    return {
        "nodes": _graph.number_of_nodes(),
        "edges": _graph.number_of_edges(),
        "entities": len(_entity_meta),
        "documents": len(_doc_entities),
        "connected_components": components,
        "top_entities": [{"id": nid, "centrality": round(c, 4)} for nid, c in top_central],
        "avg_degree": round(
            sum(d for _, d in _graph.degree()) / max(_graph.number_of_nodes(), 1), 2
        ),
    }


def get_subgraph_for_entity(entity_id: str, hops: int = 2) -> dict:
    """Return a serializable subgraph centered on entity_id."""
    neighborhood = get_neighborhood([entity_id], hops=hops, max_nodes=40)
    return {
        "center": entity_id,
        "nodes": neighborhood["nodes"],
        "edges": neighborhood["edges"],
        "paths": neighborhood["paths"],
    }


def query_related_entities(query: str, top_k: int = 10) -> list[dict]:
    """
    Return entities most relevant to a query, ranked by graph centrality + freq.
    """
    seed = find_query_entities(query)
    neighborhood = get_neighborhood(seed, hops=2, max_nodes=80)

    # Score: frequency × centrality
    try:
        centrality = nx.degree_centrality(_graph)
    except Exception:
        centrality = {}

    scored = []
    for node in neighborhood["nodes"]:
        nid = node["id"]
        freq = node.get("freq", 1)
        central = centrality.get(nid, 0)
        score = freq * (1 + central * 10)
        scored.append({**node, "relevance_score": round(score, 3)})

    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:top_k]


# ── Persistence helpers ───────────────────────────────────────────────────────

def delete_document_from_graph(filename: str) -> int:
    """Remove all nodes/edges that came exclusively from this document."""
    global _graph
    nodes_to_remove = []
    for nid in list(_graph.nodes()):
        meta = _entity_meta.get(nid, {})
        docs = meta.get("docs", [])
        if docs == [filename] or docs == filename:
            nodes_to_remove.append(nid)
        elif filename in docs:
            # Entity appears in other docs too — just update doc list
            meta["docs"] = [d for d in docs if d != filename]
            meta["freq"] = max(1, meta["freq"] - 1)
            _entity_meta[nid] = meta

    removed = 0
    for nid in nodes_to_remove:
        _graph.remove_node(nid)
        _entity_meta.pop(nid, None)
        removed += 1

    _doc_entities.pop(filename, None)
    logger.info("GraphRAG: removed %d nodes for '%s'", removed, filename)
    return removed


def clear_graph() -> None:
    """Full graph reset."""
    global _graph
    _graph.clear()
    _entity_meta.clear()
    _doc_entities.clear()
    logger.info("GraphRAG: graph cleared")


def get_full_graph_data() -> dict:
    """Serialise the full graph for the frontend knowledge graph view."""
    nodes = []
    for nid, data in _graph.nodes(data=True):
        meta = _entity_meta.get(nid, {})
        nodes.append({
            "id": nid,
            "cluster": data.get("cluster", "MISC"),
            "freq": meta.get("freq", 1),
            "docs": meta.get("docs", []),
        })

    edges = []
    for src, tgt, data in _graph.edges(data=True):
        edges.append({
            "source": src,
            "target": tgt,
            "label": data.get("label", "related_to"),
            "weight": data.get("weight", 1),
        })

    return {"nodes": nodes, "edges": edges, "stats": get_graph_stats()}