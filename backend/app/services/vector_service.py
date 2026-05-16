"""
Vector Service
--------------
ChromaDB wrapper — updated for true hybrid retrieval.

Changes vs original:
  • entity_boost raised from 0.15 → 0.25 per exact match (was too small to
    meaningfully reorder results in ChromaDB L2 distance space).
  • Rare entity bonus: entities appearing in fewer chunks get a larger boost;
    common entities (high chunk_count) are discounted.
  • Adaptive threshold: if fewer than 2 results pass the threshold, it relaxes
    by 0.1 per retry up to two times so short/ambiguous queries still surface
    something.
  • store_embeddings() and semantic_search() unchanged (already correct).
"""

import json
import logging

import chromadb

logger = logging.getLogger(__name__)

_client = chromadb.PersistentClient(path="chroma_db")
_COLLECTION_NAME = "neuralwiki"

DEFAULT_SIMILARITY_THRESHOLD = 0.7


def _get_collection():
    return _client.get_or_create_collection(name=_COLLECTION_NAME)


# ──────────────────────────────────────────────────────────────────────────────
# Write
# ──────────────────────────────────────────────────────────────────────────────

def store_embeddings(filename: str, embedded_chunks: list[dict]) -> None:
    """
    Store embeddings. Each chunk carries an "entities" key (list[str])
    from link_entities_to_chunks(). Stored as JSON in ChromaDB metadata.
    """
    collection = _get_collection()
    for chunk in embedded_chunks:
        entities_json = json.dumps(chunk.get("entities", []))
        collection.add(
            documents=[chunk["content"]],
            embeddings=[chunk["embedding"]],
            metadatas=[{
                "source":   filename,
                "chunk_id": chunk["chunk_id"],
                "entities": entities_json,
            }],
            ids=[f"{filename}_{chunk['chunk_id']}"],
        )
    logger.info("Stored %d chunks for '%s'", len(embedded_chunks), filename)


# ──────────────────────────────────────────────────────────────────────────────
# Read
# ──────────────────────────────────────────────────────────────────────────────

def _parse_entities(raw) -> list[str]:
    try:
        return json.loads(raw) if isinstance(raw, str) else []
    except (json.JSONDecodeError, TypeError):
        return []


def semantic_search(
    query_embedding,
    top_k: int = 5,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict]:
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    formatted = []
    seen: set[str] = set()

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist > similarity_threshold or doc in seen:
            continue
        seen.add(doc)
        formatted.append({
            "content":          doc,
            "source":           meta["source"],
            "chunk_id":         meta["chunk_id"],
            "similarity_score": round(dist, 4),
            "entities":         _parse_entities(meta.get("entities", "[]")),
        })

    # Adaptive threshold: relax up to twice if results are sparse
    if len(formatted) < 2:
        relaxed = similarity_threshold
        for _ in range(2):
            relaxed += 0.1
            results2 = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, collection.count()),
            )
            for doc, meta, dist in zip(
                results2["documents"][0],
                results2["metadatas"][0],
                results2["distances"][0],
            ):
                if dist > relaxed or doc in seen:
                    continue
                seen.add(doc)
                formatted.append({
                    "content":          doc,
                    "source":           meta["source"],
                    "chunk_id":         meta["chunk_id"],
                    "similarity_score": round(dist, 4),
                    "entities":         _parse_entities(meta.get("entities", "[]")),
                })
            if len(formatted) >= 2:
                break

    return formatted[:top_k]


def entity_filtered_search(
    query_embedding,
    seed_entities: list[str],
    top_k: int = 5,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    entity_boost: float = 0.25,   # raised from 0.15 — meaningful in L2 space
) -> list[dict]:
    """
    Hybrid entity-aware retrieval.

    1. Fetch wider candidate pool (4× top_k).
    2. Score each chunk by entity overlap with seed_entities.
       • Exact tagged match:  entity_boost * rarity_factor per entity
       • Partial content hit: 0.5 * entity_boost per entity (untagged at index time)
    3. Re-rank by adjusted score, return top_k.

    Rarity factor: entities that appear in very few chunks are more
    discriminating; common entities get a smaller boost.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    wide_k = min(top_k * 4, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=wide_k,
        include=["documents", "metadatas", "distances"],
    )

    # Build entity → chunk_count frequency table from the candidate pool
    # for rarity weighting (avoids a full collection scan)
    entity_chunk_count: dict[str, int] = {}
    for meta in results["metadatas"][0]:
        for e in _parse_entities(meta.get("entities", "[]")):
            entity_chunk_count[e.lower()] = entity_chunk_count.get(e.lower(), 0) + 1

    total_candidates = len(results["metadatas"][0]) or 1
    seed_lower = {e.lower() for e in seed_entities}
    candidates = []
    seen: set[str] = set()

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist > similarity_threshold + 0.3 or doc in seen:
            continue
        seen.add(doc)

        chunk_entities = _parse_entities(meta.get("entities", "[]"))
        chunk_lower = {e.lower() for e in chunk_entities}
        content_lower = doc.lower()

        boost_total = 0.0
        for seed in seed_lower:
            count = entity_chunk_count.get(seed, 1)
            # rarity factor: 1.0 for unique entity, ~0.3 for entity in 70%+ of chunks
            rarity = max(0.3, 1.0 - (count / total_candidates))

            if seed in chunk_lower:
                # Exact tagged match
                boost_total += entity_boost * rarity
            elif seed in content_lower:
                # Partial content hit (not tagged at index time)
                boost_total += entity_boost * 0.5 * rarity

        adjusted = dist - boost_total

        candidates.append({
            "content":          doc,
            "source":           meta["source"],
            "chunk_id":         meta["chunk_id"],
            "similarity_score": round(dist, 4),
            "adjusted_score":   round(adjusted, 4),
            "entities":         chunk_entities,
            "entity_overlap":   boost_total,
        })

    filtered = [c for c in candidates if c["adjusted_score"] <= similarity_threshold]
    filtered.sort(key=lambda x: x["adjusted_score"])

    # Adaptive fallback: relax threshold if too sparse
    if len(filtered) < 2:
        relaxed_pool = [c for c in candidates if c["adjusted_score"] <= similarity_threshold + 0.15]
        relaxed_pool.sort(key=lambda x: x["adjusted_score"])
        filtered = relaxed_pool

    logger.debug(
        "EntityFilteredSearch: %d candidates → %d filtered → %d returned",
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
    return [{"filename": name, "chunks": count} for name, count in sorted(counts.items())]


# ──────────────────────────────────────────────────────────────────────────────
# Delete
# ──────────────────────────────────────────────────────────────────────────────

def delete_document(filename: str) -> int:
    collection = _get_collection()
    if collection.count() == 0:
        return 0
    results = collection.get(where={"source": filename}, include=["metadatas"])
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
        logger.info("Deleted %d chunks for '%s'", len(ids), filename)
    return len(ids)


def clear_all_documents() -> int:
    collection = _get_collection()
    total = collection.count()
    _client.delete_collection(_COLLECTION_NAME)
    _client.get_or_create_collection(_COLLECTION_NAME)
    logger.info("Vector store cleared (%d chunks)", total)
    return total