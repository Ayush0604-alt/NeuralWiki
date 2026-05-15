"""
Hybrid Retrieval Service
------------------------
Combines:
  1. Vector similarity search  (ChromaDB)
  2. Graph neighbourhood BFS   (NetworkX GraphRAG)
  3. Entity-centric re-ranking

This is the single entry-point called by the /chat endpoint.
"""

import logging

from app.services.embedding_service import generate_embedding
from app.services.vector_service import semantic_search
from app.services.graph_rag_service import (
    find_query_entities,
    build_graph_context,
    enrich_retrieval_context,
)

logger = logging.getLogger(__name__)


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 1.5,
) -> dict:
    """
    Full GraphRAG retrieval pipeline.

    Returns
    -------
    {
        "vector_chunks"    : list[dict],   raw ChromaDB results
        "seed_entities"    : list[str],    query entities found in graph
        "graph_context"    : str,          formatted graph triples/paths
        "enriched_context" : str,          combined context for LLM
        "retrieval_mode"   : str,          "hybrid" | "vector_only" | "graph_only" | "none"
    }
    """
    # 1. Vector retrieval
    query_embedding = generate_embedding(query)
    vector_chunks = semantic_search(
        query_embedding,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    # 2. Graph entity detection & context building
    seed_entities = find_query_entities(query)
    graph_context = build_graph_context(query, max_triples=40) if seed_entities else ""

    # 3. Determine retrieval mode
    has_vector = bool(vector_chunks)
    has_graph = bool(graph_context)

    if has_vector and has_graph:
        retrieval_mode = "hybrid"
        enriched_context = enrich_retrieval_context(query, vector_chunks)
    elif has_vector:
        retrieval_mode = "vector_only"
        enriched_context = "\n\n".join(
            f"[Chunk {i+1} | Source: {c['source']}]\n{c['content']}"
            for i, c in enumerate(vector_chunks)
        )
    elif has_graph:
        retrieval_mode = "graph_only"
        enriched_context = graph_context
    else:
        retrieval_mode = "none"
        enriched_context = ""

    logger.info(
        "HybridRetrieve: mode=%s vector_chunks=%d seed_entities=%d",
        retrieval_mode, len(vector_chunks), len(seed_entities),
    )

    return {
        "vector_chunks": vector_chunks,
        "seed_entities": seed_entities,
        "graph_context": graph_context,
        "enriched_context": enriched_context,
        "retrieval_mode": retrieval_mode,
    }


def format_sources(vector_chunks: list[dict]) -> list[dict]:
    """Format source citations for the API response."""
    return [
        {
            "source": c["source"],
            "chunk_id": c["chunk_id"],
            "similarity_score": c["similarity_score"],
        }
        for c in vector_chunks
    ]