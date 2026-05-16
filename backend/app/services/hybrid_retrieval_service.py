"""
Hybrid Retrieval Service
------------------------
Combines entity-filtered vector search + graph BFS traversal.

Key change: uses entity_filtered_search() so the graph actually
drives which chunks surface, not just adds context strings.
Falls back to plain semantic_search() when graph is empty.
"""

import logging

from app.services.embedding_service import generate_embedding
from app.services.vector_service import (
    semantic_search,
    entity_filtered_search,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from app.services.graph_rag_service import (
    find_query_entities,
    build_graph_context,
    enrich_retrieval_context,
)

logger = logging.getLogger(__name__)


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict:
    """
    Full GraphRAG retrieval pipeline.

    When seed entities are found in the graph, uses entity_filtered_search()
    which boosts chunks containing those entities — the graph now changes
    WHAT gets retrieved, not just what context string gets appended.
    """
    # 1. Graph entity detection first
    seed_entities = find_query_entities(query)

    # 2. Vector retrieval — entity-boosted when graph has matches
    query_embedding = generate_embedding(query)

    if seed_entities:
        vector_chunks = entity_filtered_search(
            query_embedding,
            seed_entities=seed_entities,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        # Supplement if too sparse
        if len(vector_chunks) < 2:
            plain = semantic_search(query_embedding, top_k=top_k,
                                    similarity_threshold=similarity_threshold)
            existing = {(c["source"], c["chunk_id"]) for c in vector_chunks}
            for c in plain:
                if (c["source"], c["chunk_id"]) not in existing:
                    vector_chunks.append(c)
            vector_chunks = vector_chunks[:top_k]
    else:
        vector_chunks = semantic_search(query_embedding, top_k=top_k,
                                        similarity_threshold=similarity_threshold)

    # 3. Graph context string
    graph_context = build_graph_context(query, max_triples=40) if seed_entities else ""

    # 4. Retrieval mode
    has_vector = bool(vector_chunks)
    has_graph  = bool(graph_context)

    if has_vector and has_graph:
        retrieval_mode   = "hybrid"
        enriched_context = enrich_retrieval_context(query, vector_chunks)
    elif has_vector:
        retrieval_mode   = "vector_only"
        enriched_context = "\n\n".join(
            f"[Chunk {i+1} | Source: {c['source']}]\n{c['content']}"
            for i, c in enumerate(vector_chunks)
        )
    elif has_graph:
        retrieval_mode   = "graph_only"
        enriched_context = graph_context
    else:
        retrieval_mode   = "none"
        enriched_context = ""

    logger.info(
        "HybridRetrieve: mode=%s  chunks=%d  seeds=%d  entity_filter=%s",
        retrieval_mode, len(vector_chunks), len(seed_entities),
        "ON" if seed_entities else "OFF",
    )

    return {
        "vector_chunks":    vector_chunks,
        "seed_entities":    seed_entities,
        "graph_context":    graph_context,
        "enriched_context": enriched_context,
        "retrieval_mode":   retrieval_mode,
    }


def format_sources(vector_chunks: list[dict]) -> list[dict]:
    return [
        {
            "source":           c["source"],
            "chunk_id":         c["chunk_id"],
            "similarity_score": c["similarity_score"],
        }
        for c in vector_chunks
    ]