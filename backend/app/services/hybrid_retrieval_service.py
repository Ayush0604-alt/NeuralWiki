"""
Hybrid Retrieval Service
------------------------
Combines:
  1. Entity-filtered vector search  (ChromaDB — NEW: uses entity tags)
  2. Graph neighbourhood BFS        (NetworkX GraphRAG)
  3. Entity-centric re-ranking      (seed entity overlap scoring)

Changes vs original:
  • Uses entity_filtered_search() instead of plain semantic_search()
    so chunks containing seed entities rank higher regardless of pure cosine sim
  • Tighter default similarity_threshold (0.7 vs 1.5) — less noise
  • Passes seed_entities into vector search so the boost applies
  • Falls back to plain semantic_search() when no seed entities exist
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

    Returns
    -------
    {
        "vector_chunks"    : list[dict],   ChromaDB results (with entity tags)
        "seed_entities"    : list[str],    query entities found in graph
        "graph_context"    : str,          formatted graph triples/paths
        "enriched_context" : str,          combined context for LLM
        "retrieval_mode"   : str,          "hybrid" | "vector_only" | "graph_only" | "none"
    }

    Key improvement over original:
    When seed_entities exist, we call entity_filtered_search() which boosts
    chunks containing those entities, implementing genuine chunk-graph linking.
    Without seed entities we fall back to plain cosine search.
    """
    # 1. Graph entity detection first — drives vector search too
    seed_entities = find_query_entities(query)

    # 2. Entity-aware vector retrieval
    query_embedding = generate_embedding(query)

    if seed_entities:
        # NEW: entity-boosted retrieval — this is the real hybrid step
        vector_chunks = entity_filtered_search(
            query_embedding,
            seed_entities=seed_entities,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        # If entity-filtered returns too few results, supplement with plain search
        if len(vector_chunks) < 2:
            logger.info(
                "Entity-filtered search returned only %d results; "
                "supplementing with plain semantic search",
                len(vector_chunks),
            )
            plain = semantic_search(
                query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            # Merge without duplicates
            existing_ids = {(c["source"], c["chunk_id"]) for c in vector_chunks}
            for c in plain:
                if (c["source"], c["chunk_id"]) not in existing_ids:
                    vector_chunks.append(c)
            vector_chunks = vector_chunks[:top_k]
    else:
        # No graph entities found — plain vector search
        vector_chunks = semantic_search(
            query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    # 3. Graph context
    graph_context = build_graph_context(query, max_triples=40) if seed_entities else ""

    # 4. Determine retrieval mode
    has_vector = bool(vector_chunks)
    has_graph  = bool(graph_context)

    if has_vector and has_graph:
        retrieval_mode = "hybrid"
        enriched_context = enrich_retrieval_context(query, vector_chunks)
    elif has_vector:
        retrieval_mode = "vector_only"
        enriched_context = "\n\n".join(
            f"[Chunk {i + 1} | Source: {c['source']}]\n{c['content']}"
            for i, c in enumerate(vector_chunks)
        )
    elif has_graph:
        retrieval_mode = "graph_only"
        enriched_context = graph_context
    else:
        retrieval_mode = "none"
        enriched_context = ""

    logger.info(
        "HybridRetrieve: mode=%s  vector_chunks=%d  seed_entities=%d  "
        "entity_filtering=%s",
        retrieval_mode,
        len(vector_chunks),
        len(seed_entities),
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
    """Format source citations for the API response."""
    return [
        {
            "source":           c["source"],
            "chunk_id":         c["chunk_id"],
            "similarity_score": c["similarity_score"],
        }
        for c in vector_chunks
    ]