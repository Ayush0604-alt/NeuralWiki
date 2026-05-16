"""
Vector Service — rich metadata + page provenance
-------------------------------------------------
Changes vs original:
  • store_embeddings() now persists the full metadata set:
      source, chunk_id, entities, page_number, section_heading,
      document_title, author, created_at, sentence_start, sentence_end
  • semantic_search() and entity_filtered_search() return all stored metadata
    so the UI can show "Page 4 · Introduction" instead of "chunk 4".
  • _parse_entities(), adaptive threshold, rarity boost logic — all unchanged.

Backward-compatible: callers that only looked at `content`, `source`,
`chunk_id`, and `similarity_score` will continue to work.
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

def store_embeddings(
    filename: str,
    embedded_chunks: list[dict],
    doc_meta: dict | None = None,
) -> None:
    """
    Store embeddings with rich metadata.

    Parameters
    ----------
    filename : str
        Document filename (used as `source`).
    embedded_chunks : list[dict]
        Each chunk must have: content, embedding, chunk_id, length, entities.
        May optionally carry: page_number, section_heading, sentence_start,
        sentence_end (populated by chunking_service + upload.py).
    doc_meta : dict | None
        Top-level document metadata from parser_service.parse_document():
        title, author, created_at, total_pages.
        If None, all optional fields default to empty strings.
    """
    if doc_meta is None:
        doc_meta = {}

    collection = _get_collection()

    doc_title   = str(doc_meta.get("title",      filename))
    doc_author  = str(doc_meta.get("author",      ""))
    created_at  = str(doc_meta.get("created_at",  ""))
    total_pages = int(doc_meta.get("total_pages",  1))

    for chunk in embedded_chunks:
        entities_json = json.dumps(chunk.get("entities", []))

        metadata = {
            # ── Original fields ───────────────────────────────────────────────
            "source":          filename,
            "chunk_id":        chunk["chunk_id"],
            "entities":        entities_json,
            # ── New provenance fields ─────────────────────────────────────────
            "page_number":     int(chunk.get("page_number",     0)),
            "section_heading": str(chunk.get("section_heading", "")),
            "document_title":  doc_title,
            "author":          doc_author,
            "created_at":      created_at,
            "total_pages":     total_pages,
            "sentence_start":  int(chunk.get("sentence_start", 0)),
            "sentence_end":    int(chunk.get("sentence_end",   0)),
        }

        collection.add(
            documents=[chunk["content"]],
            embeddings=[chunk["embedding"]],
            metadatas=[metadata],
            ids=[f"{filename}_{chunk['chunk_id']}"],
        )

    logger.info(
        "Stored %d chunks for '%s' (title='%s', pages=%d)",
        len(embedded_chunks), filename, doc_title[:40], total_pages,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Read helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_entities(raw) -> list[str]:
    try:
        return json.loads(raw) if isinstance(raw, str) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _format_result(doc: str, meta: dict, dist: float) -> dict:
    """Unify metadata into a single result dict with all provenance fields."""
    return {
        # Core
        "content":          doc,
        "source":           meta.get("source", ""),
        "chunk_id":         meta.get("chunk_id", 0),
        "similarity_score": round(dist, 4),
        "entities":         _parse_entities(meta.get("entities", "[]")),
        # Provenance — new
        "page_number":      meta.get("page_number",     0),
        "section_heading":  meta.get("section_heading", ""),
        "document_title":   meta.get("document_title",  ""),
        "author":           meta.get("author",          ""),
        "created_at":       meta.get("created_at",      ""),
        "total_pages":      meta.get("total_pages",     1),
        "sentence_start":   meta.get("sentence_start",  0),
        "sentence_end":     meta.get("sentence_end",    0),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Semantic search
# ──────────────────────────────────────────────────────────────────────────────

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
        formatted.append(_format_result(doc, meta, dist))

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
                formatted.append(_format_result(doc, meta, dist))
            if len(formatted) >= 2:
                break

    return formatted[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
# Entity-filtered search
# ──────────────────────────────────────────────────────────────────────────────

def entity_filtered_search(
    query_embedding,
    seed_entities: list[str],
    top_k: int = 5,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    entity_boost: float = 0.25,
) -> list[dict]:
    """
    Hybrid entity-aware retrieval — unchanged logic, richer output.

    Rarity factor + adjusted_score ranking are preserved from the original.
    Output dicts now include all provenance fields.
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

    # Build entity → chunk_count for rarity weighting
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
            rarity = max(0.3, 1.0 - (count / total_candidates))
            if seed in chunk_lower:
                boost_total += entity_boost * rarity
            elif seed in content_lower:
                boost_total += entity_boost * 0.5 * rarity

        adjusted = dist - boost_total

        result = _format_result(doc, meta, dist)
        result["adjusted_score"]  = round(adjusted, 4)
        result["entity_overlap"]  = boost_total
        candidates.append(result)

    filtered = [c for c in candidates if c["adjusted_score"] <= similarity_threshold]
    filtered.sort(key=lambda x: x["adjusted_score"])

    if len(filtered) < 2:
        relaxed_pool = [c for c in candidates if c["adjusted_score"] <= similarity_threshold + 0.15]
        relaxed_pool.sort(key=lambda x: x["adjusted_score"])
        filtered = relaxed_pool

    logger.debug(
        "EntityFilteredSearch: %d candidates → %d filtered → %d returned",
        len(candidates), len(filtered), min(top_k, len(filtered)),
    )
    return filtered[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
# List / Delete
# ──────────────────────────────────────────────────────────────────────────────

def list_documents() -> list[dict]:
    collection = _get_collection()
    if collection.count() == 0:
        return []
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    titles: dict[str, str] = {}
    for meta in all_meta:
        src = meta.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
        if src not in titles:
            titles[src] = meta.get("document_title", src)
    return [
        {"filename": name, "chunks": count, "title": titles.get(name, name)}
        for name, count in sorted(counts.items())
    ]


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