from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

import os
import shutil
import logging

from app.services.parser_service import parse_document
from app.services.chunking_service import semantic_chunking
from app.services.embedding_service import generate_embeddings, generate_embedding
from app.services.vector_service import store_embeddings, semantic_search
from app.services.llm.llm_manager import generate_ai_response
from app.services.memory_service import add_message, get_conversation_history, clear_memory
from app.services.knowledge_graph_service import (
    extract_entities_and_relationships,
    store_knowledge_graph,
    get_knowledge_graph,
)

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_MB = 50


# ──────────────────────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read content to check size before writing
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
            "Uploaded %s: %d chunks, %d entities, %d relationships",
            filename,
            len(embedded_chunks),
            len(graph_data["entities"]),
            len(graph_data["relationships"]),
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

    except Exception as exc:
        logger.exception("Processing failed for %s", filename)
        # Clean up the saved file if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {exc}",
        )


# ──────────────────────────────────────────────────────────────
# Semantic Search
# ──────────────────────────────────────────────────────────────

@router.get("/search")
async def search(query: str):
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty",
        )

    query_embedding = generate_embedding(query.strip())
    results = semantic_search(query_embedding)

    return {
        "success": True,
        "query": query,
        "total_results": len(results),
        "results": results,
    }


# ──────────────────────────────────────────────────────────────
# AI Chat
# ──────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: dict):
    query = (request.get("query") or "").strip()

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query is required",
        )

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
        return {
            "success": True,
            "query": query,
            "response": response_text,
            "sources": [],
        }

    context = "\n\n".join(r["content"] for r in results)
    ai_response = generate_ai_response(query=query, context=context, history=history)
    add_message("AI", ai_response)

    return {
        "success": True,
        "query": query,
        "response": ai_response,
        "sources": [
            {
                "source": r["source"],
                "chunk_id": r["chunk_id"],
                "similarity_score": r["similarity_score"],
            }
            for r in results
        ],
        "conversation_history": history,
    }


# ──────────────────────────────────────────────────────────────
# Clear Memory
# ──────────────────────────────────────────────────────────────

@router.post("/clear-memory")
async def reset_memory():
    clear_memory()
    return {"success": True, "message": "Conversation memory cleared"}


# ──────────────────────────────────────────────────────────────
# Knowledge Graph
# ──────────────────────────────────────────────────────────────

@router.get("/knowledge-graph")
async def knowledge_graph():
    graph = get_knowledge_graph()
    return {"success": True, "graph": graph}