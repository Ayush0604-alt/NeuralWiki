"""
Vector Service
--------------
ChromaDB wrapper — updated to store entity tags per chunk.

Changes vs original:
  • store_embeddings() now accepts optional entity list per chunk
    and persists it as a JSON-encoded metadata field ("entities")
  • semantic_search() returns entity tags with each result
  • entity_filtered_search() — NEW: filter by seed entities before
    doing similarity ranking; this is the real hybrid retrieval win
  • Default similarity_threshold tightened to 0.7 (was 1.5)
    to reduce noise in the LLM context window
"""

import json
import logging

import chromadb

logger = logging.getLogger(__name__)

_client = chromadb.PersistentClient(path="chroma_db")
_COLLECTION_NAME = "neuralwiki"

# ── Tightened default — change here affects all callers that don't override ───
DEFAULT_SIMILARITY_THRESHOLD = 0.7


def _get_collection():
    return _client.get_or_create_collection(name=_COLLECTION_NAME)


# ──────────────────────────────────────────────────────────────────────────────
# Write
# ──────────────────────────────────────────────────────────────────────────────

def store_embeddings(filename: str, embedded_chunks: list[dict]) -> None:
    """
    Store embeddings in ChromaDB.

    Each chunk may optionally carry an "entities" key (list[str]) from the
    chunk-to-entity linking step in knowledge_graph_service.py.
    Those entity tags are serialised to JSON and stored as metadata so
    entity_filtered_search() can retrieve them cheaply later.
    """
    collection = _get_collection()
    for chunk in embedded_chunks:
        # Serialise entity list (may be absent for old-style chunks)
        entities_json = json.dumps(chunk.get("entities", []))

        collection.add(
            documents=[chunk["content"]],
            embeddings=[chunk["embedding"]],
            metadatas=[{
                "source":   filename,
                "chunk_id": chunk["chunk_id"],
                "entities": entities_json,          # NEW
            }],
            ids=[f"{filename}_{chunk['chunk_id']}"],
        )
    logger.info("Stored %d chunks for '%s'", len(embedded_chunks), filename)


# ──────────────────────────────────────────────────────────────────────────────
# Read
# ──────────────────────────────────────────────────────────────────────────────

def semantic_search(
    query_embedding,
    top_k: int = 5,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Standard semantic search.  Tighter default threshold (0.7) reduces noise.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    formatted = []
    seen_content: set[str] = set()

    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if distance > similarity_threshold:
            continue
        if doc in seen_content:
            continue
        seen_content.add(doc)

        # Deserialise entity tags stored at index time
        entities: list[str] = []
        raw = metadata.get("entities", "[]")
        try:
            entities = json.loads(raw) if isinstance(raw, str) else []
        except (json.JSONDecodeError, TypeError):
            entities = []

        formatted.append({
            "content":          doc,
            "source":           metadata["source"],
            "chunk_id":         metadata["chunk_id"],
            "similarity_score": round(distance, 4),
            "entities":         entities,            # NEW — passed through
        })

    return formatted


def entity_filtered_search(
    query_embedding,
    seed_entities: list[str],
    top_k: int = 5,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    entity_boost: float = 0.15,
) -> list[dict]:
    """
    Hybrid entity-aware retrieval — NEW.

    Algorithm:
    1. Run a wider semantic search (top_k * 3) to get candidates.
    2. For each candidate, count how many seed_entities it contains.
    3. Apply a score boost proportional to entity overlap.
    4. Re-rank and return top_k.

    This means a chunk that mentions "AWS Lambda" and "API Gateway" will rank
    higher than a generic chunk with slightly better cosine similarity.

    Parameters
    ----------
    query_embedding    : embedding vector
    seed_entities      : list of canonical entity strings from graph traversal
    top_k              : final number of results to return
    similarity_threshold : max cosine distance to consider (lower = stricter)
    entity_boost       : distance reduction per matched seed entity
                         (0.15 means matching 2 entities → -0.30 on distance)
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    # Wider initial fetch
    wide_k = min(top_k * 4, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=wide_k,
    )

    seed_lower = {e.lower() for e in seed_entities}
    candidates = []
    seen_content: set[str] = set()

    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if distance > similarity_threshold + 0.3:   # wider initial window
            continue
        if doc in seen_content:
            continue
        seen_content.add(doc)

        # Deserialise stored entity tags
        chunk_entities: list[str] = []
        raw = metadata.get("entities", "[]")
        try:
            chunk_entities = json.loads(raw) if isinstance(raw, str) else []
        except (json.JSONDecodeError, TypeError):
            chunk_entities = []

        # Count entity overlap with seed entities
        chunk_lower = {e.lower() for e in chunk_entities}
        overlap = len(seed_lower & chunk_lower)

        # Also do a quick substring scan for seeds not tagged at index time
        content_lower = doc.lower()
        for seed in seed_lower:
            if seed not in chunk_lower and seed in content_lower:
                overlap += 0.5   # partial credit for untagged mentions

        # Adjusted score: lower is better (cosine distance)
        adjusted_distance = distance - (overlap * entity_boost)

        candidates.append({
            "content":          doc,
            "source":           metadata["source"],
            "chunk_id":         metadata["chunk_id"],
            "similarity_score": round(distance, 4),
            "adjusted_score":   round(adjusted_distance, 4),
            "entities":         chunk_entities,
            "entity_overlap":   overlap,
        })

    # Filter by adjusted score, then rank
    filtered = [c for c in candidates if c["adjusted_score"] <= similarity_threshold]
    filtered.sort(key=lambda x: x["adjusted_score"])

    logger.debug(
        "EntityFilteredSearch: %d candidates → %d after threshold → returning %d",
        len(candidates), len(filtered), min(top_k, len(filtered)),
    )

    return filtered[:top_k]


def list_documents() -> list[dict]:
    collection = _get_collection()
    if collection.count() == 0:
        return []
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for meta in all_meta:
        src = meta.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return [
        {"filename": name, "chunks": count}
        for name, count in sorted(counts.items())
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Delete
# ──────────────────────────────────────────────────────────────────────────────

def delete_document(filename: str) -> int:
    collection = _get_collection()
    if collection.count() == 0:
        return 0
    results = collection.get(
        where={"source": filename},
        include=["metadatas"],
    )
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks for '%s'", len(ids_to_delete), filename)
    return len(ids_to_delete)


def clear_all_documents() -> int:
    collection = _get_collection()
    total = collection.count()
    _client.delete_collection(_COLLECTION_NAME)
    _client.get_or_create_collection(_COLLECTION_NAME)
    logger.info("Vector store cleared (%d chunks removed)", total)
    return total