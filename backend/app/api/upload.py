from fastapi import APIRouter, UploadFile, File

import os
import shutil

from app.services.parser_service import (
    parse_document
)

from app.services.chunking_service import (
    semantic_chunking
)

from app.services.embedding_service import (
    generate_embeddings,
    generate_embedding
)

from app.services.vector_service import (
    store_embeddings,
    semantic_search
)

from app.services.llm.llm_manager import (
    generate_ai_response
)

from app.services.memory_service import (
    add_message,
    get_conversation_history,
    clear_memory
)

from app.services.knowledge_graph_service import (
    extract_entities_and_relationships,
    store_knowledge_graph,
    get_knowledge_graph
)


router = APIRouter()

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

ALLOWED_EXTENSIONS = [
    ".pdf",
    ".txt",
    ".md"
]


# ==========================================
# Upload API
# ==========================================

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...)
):

    filename = file.filename

    extension = os.path.splitext(
        filename
    )[1].lower()

    # Validate file type
    if extension not in ALLOWED_EXTENSIONS:

        return {
            "success": False,
            "message": "Unsupported file type"
        }

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    # Save uploaded file
    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # Parse document
    extracted_text = parse_document(
        file_path
    )

    # Semantic chunking
    chunks = semantic_chunking(
        extracted_text
    )

    # Generate embeddings
    embedded_chunks = generate_embeddings(
        chunks
    )

    # Store embeddings in ChromaDB
    store_embeddings(
        filename,
        embedded_chunks
    )
    print("STARTING GRAPH EXTRACTION")
    # Extract knowledge graph
    graph_data = (
        extract_entities_and_relationships(
            extracted_text
        )
    )

    # Store graph data
    store_knowledge_graph(
        graph_data
    )
    print("GRAPH STORED SUCCESSFULLY")
    return {
        "success": True,
        "filename": filename,
        "chunks_stored": len(
            embedded_chunks
        ),
        "entities_found": len(
            graph_data["entities"]
        ),
        "relationships_found": len(
            graph_data["relationships"]
        ),
        "message":
            "Document processed successfully"
    }


# ==========================================
# Semantic Search API
# ==========================================

@router.get("/search")
async def search(query: str):

    # Generate query embedding
    query_embedding = generate_embedding(
        query
    )

    # Semantic retrieval
    results = semantic_search(
        query_embedding
    )

    return {
        "success": True,
        "query": query,
        "total_results": len(results),
        "results": results
    }


# ==========================================
# AI Chat API (Advanced RAG)
# ==========================================

@router.post("/chat")
async def chat(request: dict):

    query = request.get("query")

    if not query:

        return {
            "success": False,
            "message": "Query is required"
        }

    # Store user message
    add_message(
        "User",
        query
    )

    # Get conversation history
    history = (
        get_conversation_history()
    )

    # Generate query embedding
    query_embedding = generate_embedding(
        query
    )

    # Advanced semantic retrieval
    results = semantic_search(
        query_embedding,
        top_k=5,
        similarity_threshold=1.5
    )

    # Handle empty retrieval
    if len(results) == 0:

        return {
            "success": True,
            "query": query,
            "response":
                "No relevant information found "
                "in uploaded documents.",
            "sources": []
        }

    # Build retrieved context
    context = "\n\n".join([
        result["content"]
        for result in results
    ])

    # Generate AI response
    ai_response = generate_ai_response(
        query=query,
        context=context,
        history=history
    )

    # Store AI response
    add_message(
        "AI",
        ai_response
    )

    return {
        "success": True,
        "query": query,
        "response": ai_response,

        "sources": [
            {
                "source":
                    result["source"],

                "chunk_id":
                    result["chunk_id"]
            }

            for result in results
        ],

        "conversation_history":
            history
    }


# ==========================================
# Clear Memory API
# ==========================================

@router.post("/clear-memory")
async def reset_memory():

    clear_memory()

    return {
        "success": True,
        "message":
            "Conversation memory cleared"
    }


# ==========================================
# Knowledge Graph API
# ==========================================

@router.get("/knowledge-graph")
async def knowledge_graph():

    graph = get_knowledge_graph()

    return {
        "success": True,
        "graph": graph
    }