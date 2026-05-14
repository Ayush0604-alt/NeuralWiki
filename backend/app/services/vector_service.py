"""
Vector Service
--------------
ChromaDB wrapper.

New in this version:
  • list_documents()       — returns unique source filenames + chunk counts
  • delete_document(name)  — removes all chunks belonging to one file
  • clear_all_documents()  — nukes every chunk (full reset)
"""

import logging
import chromadb

logger = logging.getLogger(__name__)

# Persistent ChromaDB client
_client = chromadb.PersistentClient(path="chroma_db")
_COLLECTION_NAME = "neuralwiki"


def _get_collection():
    """Always fetch (or create) the collection fresh so deletions take effect."""
    return _client.get_or_create_collection(name=_COLLECTION_NAME)


# ──────────────────────────────────────────────────────────────────────────────
# Write
# ──────────────────────────────────────────────────────────────────────────────

def store_embeddings(filename: str, embedded_chunks: list[dict]) -> None:
    collection = _get_collection()
    for chunk in embedded_chunks:
        collection.add(
            documents=[chunk["content"]],
            embeddings=[chunk["embedding"]],
            metadatas=[{"source": filename, "chunk_id": chunk["chunk_id"]}],
            ids=[f"{filename}_{chunk['chunk_id']}"],
        )
    logger.info("Stored %d chunks for '%s'", len(embedded_chunks), filename)


# ──────────────────────────────────────────────────────────────────────────────
# Read
# ──────────────────────────────────────────────────────────────────────────────

def semantic_search(
    query_embedding,
    top_k: int = 5,
    similarity_threshold: float = 1.5,
) -> list[dict]:
    collection = _get_collection()

    # Guard: ChromaDB raises if the collection is empty
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
        formatted.append({
            "content":          doc,
            "source":           metadata["source"],
            "chunk_id":         metadata["chunk_id"],
            "similarity_score": round(distance, 4),
        })

    return formatted


def list_documents() -> list[dict]:
    """
    Returns one entry per unique source file:
        [{"filename": "report.pdf", "chunks": 14}, …]
    Sorted alphabetically by filename.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    # Fetch all metadata (no embeddings needed)
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
    """
    Delete every chunk whose metadata source == filename.
    Returns the number of chunks removed.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return 0

    # Find IDs that belong to this file
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
    """
    Remove every chunk from the vector store.
    Returns total chunks deleted.
    Recreates an empty collection so the system stays usable immediately.
    """
    collection = _get_collection()
    total = collection.count()

    # Delete and recreate — faster than fetching all IDs when count is large
    _client.delete_collection(_COLLECTION_NAME)
    _client.get_or_create_collection(_COLLECTION_NAME)

    logger.info("Vector store cleared (%d chunks removed)", total)
    return total