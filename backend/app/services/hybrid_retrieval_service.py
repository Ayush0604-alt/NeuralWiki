"""
Hybrid Retrieval Service
------------------------
Combines:
  1. Vector similarity search (ChromaDB)
  2. Graph neighbourhood traversal (NetworkX GraphRAG)
  3. Entity-centric re-ranking

Returns an enriched context object ready for the LLM.
"""

import logging
from app.services.embedding_service import generate_embedding
from app.services.vector_service import semantic_search
from app.services.graph_rag_service import (
    find_query_entities,
    get_neighborhood,
    build_graph_context,
    query_related_entities,
    enrich_retrieval_context,
)

logger = logging.getLogger(__name__)


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 1.5,
    graph_hops: int = 2,
) -> dict:
    """
    Main GraphRAG retrieval pipeline.

    Returns:
    {
        "vector_chunks": [...],      # raw vector results
        "graph_entities": [...],     # relevant graph entities
        "graph_context": "...",      # formatted graph triples/paths
        "enriched_context": "...",   # combined context for LLM
        "seed_entities": [...],      # query entities found in graph
        "retrieval_mode": "hybrid"|"vector_only"|"graph_only",
    }
    """
    # 1. Vector retrieval
    query_embedding = generate_embedding(query)
    vector_chunks = semantic_search(
        query_embedding,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    # 2. Graph entity detection
    seed_entities = find_query_entities(query)

    # 3. Graph traversal
    graph_entities = []
    graph_context = ""

    if seed_entities:
        graph_entities = query_related_entities(query, top_k=15)
        graph_context = build_graph_context(query, max_triples=40)

    # 4. Determine retrieval mode
    has_vector = bool(vector_chunks)
    has_graph = bool(graph_context)

    if has_vector and has_graph:
        retrieval_mode = "hybrid"
    elif has_graph:
        retrieval_mode = "graph_only"
    elif has_vector:
        retrieval_mode = "vector_only"
    else:
        retrieval_mode = "none"

    # 5. Build enriched context
    if has_vector and has_graph:
        enriched_context = enrich_retrieval_context(query, vector_chunks)
    elif has_vector:
        enriched_context = "\n\n".join(
            f"[Chunk {i+1} | Source: {c['source']}]\n{c['content']}"
            for i, c in enumerate(vector_chunks)
        )
    elif has_graph:
        enriched_context = graph_context
    else:
        enriched_context = ""

    logger.info(
        "HybridRetrieval: query='%s...' mode=%s vector_chunks=%d seed_entities=%d",
        query[:40], retrieval_mode, len(vector_chunks), len(seed_entities),
    )

    return {
        "vector_chunks": vector_chunks,
        "graph_entities": graph_entities,
        "graph_context": graph_context,
        "enriched_context": enriched_context,
        "seed_entities": seed_entities,
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