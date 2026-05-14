"""
Upload API Router — with document management
--------------------------------------------
New endpoints vs previous version:
  GET  /documents           — list indexed files + chunk counts
  POST /delete-document     — remove one file by name
  POST /clear-documents     — full reset (vector store + files + graph + memory)
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
from app.services.llm.llm_manager import generate_ai_response
from app.services.memory_service import add_message, get_conversation_history, clear_memory
from app.services.knowledge_graph_service import (
    extract_entities_and_relationships,
    store_knowledge_graph,
    get_knowledge_graph,
    clear_knowledge_graph,
)

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_MB = 50
_SAFE_FILENAME_RE = re.compile(r"[^\w.\-]")


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

        chunks = semantic_chunking(extracted_text)
        embedded_chunks = generate_embeddings(chunks)
        store_embeddings(filename, embedded_chunks)

        graph_data = extract_entities_and_relationships(extracted_text)
        store_knowledge_graph(graph_data)

        logger.info(
            "Uploaded '%s': %d chunks, %d entities, %d relationships",
            filename, len(embedded_chunks),
            len(graph_data["entities"]), len(graph_data["relationships"]),
        )

        return {
            "success": True,
            "filename": filename,
            "file_size_kb": round(len(content) / 1024, 1),
            "chunks_stored": len(embedded_chunks),
            "entities_found": len(graph_data["entities"]),
            "relationships_found": len(graph_data["relationships"]),
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

    logger.info("Deleted '%s': %d chunks, file on disk: %s", filename, chunks_deleted, file_deleted)
    return {
        "success": True,
        "filename": filename,
        "chunks_deleted": chunks_deleted,
        "file_deleted": file_deleted,
        "message": f"'{filename}' removed from knowledge base",
    }


@router.post("/clear-documents")
async def clear_all():
    """Full reset — vector store, physical files, graph, conversation memory."""
    chunks_deleted = clear_all_documents()

    files_deleted = 0
    for path in glob.glob(os.path.join(UPLOAD_DIR, "*")):
        if os.path.isfile(path):
            try:
                os.remove(path)
                files_deleted += 1
            except OSError as e:
                logger.warning("Could not delete %s: %s", path, e)

    clear_knowledge_graph()
    clear_memory()

    logger.info("Full reset: %d chunks, %d files deleted", chunks_deleted, files_deleted)
    return {
        "success": True,
        "chunks_deleted": chunks_deleted,
        "files_deleted": files_deleted,
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


# ── Chat ──────────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: dict):
    query = (request.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query is required")

    add_message("User", query)
    history = get_conversation_history()

    query_embedding = generate_embedding(query)
    results = semantic_search(query_embedding, top_k=5, similarity_threshold=1.5)

    if not results:
        response_text = (
            "I couldn't find relevant information in the uploaded documents. "
            "Try uploading more documents or rephrasing your question."
        )
        add_message("AI", response_text)
        return {"success": True, "query": query, "response": response_text, "sources": []}

    context = "\n\n".join(r["content"] for r in results)
    ai_response = generate_ai_response(query=query, context=context, history=history)
    add_message("AI", ai_response)

    return {
        "success": True,
        "query": query,
        "response": ai_response,
        "sources": [
            {"source": r["source"], "chunk_id": r["chunk_id"], "similarity_score": r["similarity_score"]}
            for r in results
        ],
        "conversation_history": history,
    }


# ── Clear memory only ─────────────────────────────────────────────────────────

@router.post("/clear-memory")
async def reset_memory():
    clear_memory()
    return {"success": True, "message": "Conversation memory cleared"}


# ── Knowledge graph ───────────────────────────────────────────────────────────

@router.get("/knowledge-graph")
async def knowledge_graph():
    return {"success": True, "graph": get_knowledge_graph()}