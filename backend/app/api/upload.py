"""
Upload API Router — GraphRAG edition
--------------------------------------
Endpoints:
  POST /upload              — upload + process (entity extraction → graph ingest → wiki)
  GET  /documents           — list indexed files + chunk counts
  POST /delete-document     — remove one file (vector + graph + wiki)
  POST /clear-documents     — full reset
  GET  /search              — semantic vector search
  POST /chat                — GraphRAG hybrid chat (vector + graph traversal)
  POST /clear-memory        — reset conversation memory
  GET  /knowledge-graph     — graph data for frontend visualisation
  GET  /graph/stats         — graph analytics
  POST /graph/query         — multi-hop entity query (dev/debug endpoint)

  GET  /wiki/pages
  GET  /wiki/page/{filename}
  POST /wiki/generate
  DELETE /wiki/page/{filename}
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status

import os
import re
import logging
import glob

from app.services.parser_service import parse_document
from app.services.chunking_service import semantic_chunking
from app.services.embedding_service import generate_embeddings, generate_embedding
from app.services.vector_service import (
    store_embeddings,
    semantic_search,
    list_documents,
    delete_document,
    clear_all_documents,
)
from app.services.llm.llm_manager import generate_ai_response, generate_graphrag_response
from app.services.memory_service import add_message, get_conversation_history, clear_memory

# ── GraphRAG imports ───────────────────────────────────────────────────────────
from app.services.knowledge_graph_service import (
    extract_entities_and_relationships,
    # legacy in-memory store (kept for backward compat with /knowledge-graph endpoint)
    store_knowledge_graph,
    clear_knowledge_graph,
)
from app.services.graph_rag_service import (
    ingest_document_graph,
    get_full_graph_for_frontend,
    get_graph_stats,
    remove_document as graph_remove_document,
    clear_graph as graph_clear,
    find_query_entities,
    get_neighbors,
    find_paths,
    load_graph,
)
from app.services.hybrid_retrieval_service import hybrid_retrieve, format_sources

from app.services.wiki_service import (
    generate_wiki_page,
    get_wiki_page,
    list_wiki_pages,
    delete_wiki_page,
    clear_wiki_pages,
)

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_MB = 50
_SAFE_FILENAME_RE = re.compile(r"[^\w.\-]")

# ── Load persisted graph on module import ─────────────────────────────────────
load_graph()


def _sanitize_filename(raw: str) -> str:
    name = os.path.basename(raw or "upload")
    name = _SAFE_FILENAME_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "upload"


def _delete_upload_file(filename: str) -> bool:
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def _make_wiki_llm():
    """Return a (query, context) → str callable using NVIDIA API for wiki gen."""
    import os as _os
    from openai import OpenAI

    client = OpenAI(
        api_key=_os.getenv("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1",
    )
    wiki_system = """You are NeuralWiki, an expert knowledge base curator.
Generate a structured wiki page in Markdown with these sections:
## Overview, ## Key Concepts, ## Notable Entities, ## Key Facts & Findings,
## Relationships & Connections, ## Quick Reference (table)
Be concise, accurate, and only use information from the document."""

    def _llm(query: str, context: str) -> str:
        completion = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                {"role": "system", "content": wiki_system},
                {"role": "user", "content": f"Document: {context}\n\nTask: {query}"},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        return completion.choices[0].message.content

    return _llm


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    raw_name = file.filename or "upload.bin"
    filename = _sanitize_filename(raw_name)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit ({size_mb:.1f} MB)",
        )

    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        extracted_text = parse_document(file_path)
        if not extracted_text or not extracted_text.strip():
            raise ValueError("Document appears to be empty or could not be parsed.")

        # ── 1. Vector store ──────────────────────────────────────────────────
        chunks = semantic_chunking(extracted_text)
        embedded_chunks = generate_embeddings(chunks)
        store_embeddings(filename, embedded_chunks)

        # ── 2. Entity + relationship extraction ─────────────────────────────
        graph_data = extract_entities_and_relationships(extracted_text)

        # ── 3a. Legacy in-memory store (keeps /knowledge-graph working) ─────
        store_knowledge_graph(graph_data)

        # ── 3b. GraphRAG: ingest into persistent NetworkX graph ─────────────
        ingest_document_graph(
            doc_id=filename,
            center=graph_data.get("center", {}),
            clusters=graph_data.get("clusters", {}),
            relationships=graph_data.get("relationships", []),
            doc_type=graph_data.get("doc_type", "general"),
        )

        # ── 4. Wiki generation ───────────────────────────────────────────────
        wiki_generated = False
        try:
            generate_wiki_page(filename, extracted_text, _make_wiki_llm())
            wiki_generated = True
        except Exception as wiki_err:
            logger.warning("Wiki generation skipped for '%s': %s", filename, wiki_err)

        logger.info(
            "Uploaded '%s': %d chunks, %d entities, %d relationships, wiki=%s",
            filename,
            len(embedded_chunks),
            len(graph_data["entities"]),
            len(graph_data["relationships"]),
            wiki_generated,
        )

        return {
            "success": True,
            "filename": filename,
            "file_size_kb": round(len(content) / 1024, 1),
            "chunks_stored": len(embedded_chunks),
            "entities_found": len(graph_data["entities"]),
            "relationships_found": len(graph_data["relationships"]),
            "wiki_generated": wiki_generated,
            "message": "Document processed successfully",
        }

    except Exception:
        logger.exception("Processing failed for '%s'", filename)
        _delete_upload_file(filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Processing failed. Check server logs for details.",
        )


# ── Document management ───────────────────────────────────────────────────────

@router.get("/documents")
async def list_uploaded_documents():
    docs = list_documents()
    return {
        "success": True,
        "total_documents": len(docs),
        "total_chunks": sum(d["chunks"] for d in docs),
        "documents": docs,
    }


@router.post("/delete-document")
async def delete_one_document(body: dict):
    filename = _sanitize_filename((body.get("filename") or "").strip())
    if not filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="filename is required")

    chunks_deleted = delete_document(filename)
    file_deleted = _delete_upload_file(filename)
    graph_nodes_removed = graph_remove_document(filename)
    delete_wiki_page(filename)

    logger.info(
        "Deleted '%s': %d chunks, %d graph nodes, file=%s",
        filename, chunks_deleted, graph_nodes_removed, file_deleted,
    )
    return {
        "success": True,
        "filename": filename,
        "chunks_deleted": chunks_deleted,
        "graph_nodes_removed": graph_nodes_removed,
        "file_deleted": file_deleted,
        "message": f"'{filename}' removed from knowledge base",
    }


@router.post("/clear-documents")
async def clear_all():
    """Full reset — vector store, files, graph, memory, wiki."""
    chunks_deleted = clear_all_documents()

    files_deleted = 0
    for path in glob.glob(os.path.join(UPLOAD_DIR, "*")):
        if os.path.isfile(path):
            try:
                os.remove(path)
                files_deleted += 1
            except OSError as exc:
                logger.warning("Could not delete %s: %s", path, exc)

    clear_knowledge_graph()   # legacy list
    graph_clear()             # NetworkX graph
    clear_memory()
    wiki_cleared = clear_wiki_pages()

    logger.info(
        "Full reset: %d chunks, %d files, %d wiki pages", chunks_deleted, files_deleted, wiki_cleared
    )
    return {
        "success": True,
        "chunks_deleted": chunks_deleted,
        "files_deleted": files_deleted,
        "wiki_pages_cleared": wiki_cleared,
        "message": "Knowledge base cleared. System is ready for new documents.",
    }


# ── Search ────────────────────────────────────────────────────────────────────

@router.get("/search")
async def search(query: str):
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query must not be empty")
    query_embedding = generate_embedding(q)
    results = semantic_search(query_embedding)
    return {"success": True, "query": q, "total_results": len(results), "results": results}


# ── Chat — GraphRAG hybrid ────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: dict):
    query = (request.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query is required")

    add_message("User", query)
    history = get_conversation_history()

    # GraphRAG hybrid retrieval
    retrieval = hybrid_retrieve(query, top_k=5, similarity_threshold=1.5)

    if retrieval["retrieval_mode"] == "none":
        response_text = (
            "I couldn't find relevant information in the uploaded documents. "
            "Try uploading more documents or rephrasing your question."
        )
        add_message("AI", response_text)
        return {
            "success": True,
            "query": query,
            "response": response_text,
            "sources": [],
            "retrieval_mode": "none",
            "seed_entities": [],
        }

    # Generate response with graph-enriched context
    ai_response = generate_graphrag_response(
        query=query,
        enriched_context=retrieval["enriched_context"],
        history=history,
        retrieval_mode=retrieval["retrieval_mode"],
        seed_entities=retrieval["seed_entities"],
    )
    add_message("AI", ai_response)

    return {
        "success": True,
        "query": query,
        "response": ai_response,
        "sources": format_sources(retrieval["vector_chunks"]),
        "retrieval_mode": retrieval["retrieval_mode"],
        "seed_entities": retrieval["seed_entities"],
        "conversation_history": history,
    }


# ── Memory ────────────────────────────────────────────────────────────────────

@router.post("/clear-memory")
async def reset_memory():
    clear_memory()
    return {"success": True, "message": "Conversation memory cleared"}


# ── Knowledge graph — legacy endpoint (for KnowledgeGraph.jsx) ───────────────

@router.get("/knowledge-graph")
async def knowledge_graph():
    """
    Returns graph data for the frontend.
    Prefers the persistent NetworkX graph; falls back to legacy in-memory store.
    """
    frontend_data = get_full_graph_for_frontend()
    if frontend_data:
        return {"success": True, "graph": frontend_data}

    # Fallback: legacy in-memory list from knowledge_graph_service
    from app.services.knowledge_graph_service import get_knowledge_graph as legacy_get
    return {"success": True, "graph": legacy_get()}


# ── Graph analytics ───────────────────────────────────────────────────────────

@router.get("/graph/stats")
async def graph_stats():
    """Return NetworkX graph statistics."""
    return {"success": True, "stats": get_graph_stats()}


@router.post("/graph/query")
async def graph_query(body: dict):
    """
    Multi-hop entity query.
    Body: {"entities": ["AWS Lambda", "API Gateway"], "hops": 2}
    """
    entities = body.get("entities") or []
    hops = min(int(body.get("hops", 2)), 4)

    if not entities:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="entities list is required")

    found = []
    all_neighbors = {}
    for ent in entities[:5]:
        nbrs = get_neighbors(ent, max_hops=hops)
        if nbrs:
            found.append(ent)
            all_neighbors[ent] = [
                {
                    "node": n["node"],
                    "cluster": n["cluster"],
                    "distance": n["distance"],
                    "via": n["via"],
                    "path": " → ".join(n["path"]),
                }
                for n in nbrs[:20]
            ]

    # Cross-entity paths
    paths = []
    if len(found) >= 2:
        for i, s1 in enumerate(found):
            for s2 in found[i + 1:]:
                for p in find_paths(s1, s2, max_hops=hops + 1)[:3]:
                    paths.append(" → ".join(p))

    return {
        "success": True,
        "entities_found": found,
        "neighbors": all_neighbors,
        "cross_paths": paths,
    }


# ── Wiki ──────────────────────────────────────────────────────────────────────

@router.get("/wiki/pages")
async def get_wiki_pages():
    pages = list_wiki_pages()
    return {"success": True, "total": len(pages), "pages": pages}


@router.get("/wiki/page/{filename:path}")
async def get_single_wiki_page(filename: str):
    page = get_wiki_page(filename)
    if not page:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No wiki page found for '{filename}'. Try regenerating it.",
        )
    return {"success": True, "page": page}


@router.post("/wiki/generate")
async def regenerate_wiki(body: dict):
    filename = _sanitize_filename((body.get("filename") or "").strip())
    if not filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="filename is required")

    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found on disk. Please re-upload it.",
        )

    try:
        extracted_text = parse_document(file_path)
        if not extracted_text or not extracted_text.strip():
            raise ValueError("Document is empty or unreadable.")
        page = generate_wiki_page(filename, extracted_text, _make_wiki_llm())
        return {"success": True, "page": page}
    except Exception as exc:
        logger.exception("Wiki regeneration failed for '%s'", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Wiki generation failed: {str(exc)}",
        )


@router.delete("/wiki/page/{filename:path}")
async def remove_wiki_page(filename: str):
    deleted = delete_wiki_page(filename)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wiki page not found")
    return {"success": True, "message": f"Wiki page for '{filename}' deleted"}