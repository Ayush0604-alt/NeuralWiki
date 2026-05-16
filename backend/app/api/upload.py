"""
Upload API Router — GraphRAG edition (true graph structure)
-----------------------------------------------------------
Pipeline order (critical):
  1. Parse text
  2. Chunk
  3. Extract entities + relationships  (3-layer: SVO + co-occurrence + LLM)
  4. link_entities_to_chunks()         (tag chunks BEFORE embedding)
  5. generate_embeddings()
  6. store_embeddings()                (entity tags stored in ChromaDB)
  7. ingest_document_graph()
  8. generate_wiki_page()
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse

import os, re, json, logging, glob

from app.services.parser_service import parse_document
from app.services.chunking_service import semantic_chunking
from app.services.embedding_service import generate_embeddings, generate_embedding
from app.services.vector_service import (
    store_embeddings, semantic_search, list_documents,
    delete_document, clear_all_documents,
)
from app.services.llm.llm_manager import generate_ai_response, generate_graphrag_response
from app.services.llm.nvidia_provider import NVIDIA_API_KEY, SYSTEM_PROMPT as NVIDIA_SYSTEM_PROMPT
from app.services.memory_service import add_message, get_conversation_history, clear_memory

from app.services.knowledge_graph_service import (
    extract_entities_and_relationships,
    store_knowledge_graph,
    clear_knowledge_graph,
    link_entities_to_chunks,
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
    generate_wiki_page, get_wiki_page, list_wiki_pages,
    delete_wiki_page, clear_wiki_pages,
)

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_MB = 50
_SAFE_RE = re.compile(r"[^\w.\-]")

load_graph()


def _sanitize_filename(raw: str) -> str:
    name = os.path.basename(raw or "upload")
    name = _SAFE_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    parts = name.rsplit(".", 1)
    if len(parts) == 2 and not parts[0]:
        name = "upload." + parts[1]
    return name or "upload"


def _delete_upload_file(filename: str) -> bool:
    path = os.path.join(UPLOAD_DIR, filename)
    resolved = os.path.realpath(path)
    if not resolved.startswith(os.path.realpath(UPLOAD_DIR)):
        logger.warning("Path traversal blocked: '%s'", filename)
        return False
    if os.path.exists(resolved):
        os.remove(resolved)
        return True
    return False


def _make_wiki_llm():
    from openai import OpenAI
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1")
    system = """You are NeuralWiki, an expert knowledge base curator.
Generate a structured wiki page in Markdown with sections:
## Overview, ## Key Concepts, ## Notable Entities, ## Key Facts & Findings,
## Relationships & Connections, ## Quick Reference (table)
Be concise, accurate, and only use information from the document."""

    def _llm(query: str, context: str) -> str:
        r = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Document: {context}\n\nTask: {query}"},
            ],
            temperature=0.2, max_tokens=1500,
        )
        return r.choices[0].message.content
    return _llm


_GRAPHRAG_ADDENDUM = """
You also have access to a **Knowledge Graph context** with:
- Entity relationship triples: `Entity A → [relationship] → Entity B`
- Multi-hop reasoning paths
- Key entity metadata (type, frequency)

Use triples to reason about HOW entities relate. Follow multi-hop paths for
chained facts. If graph triples and text chunks conflict, prefer text chunks.
"""


def _build_graphrag_context(query: str, retrieval: dict, history: str) -> str:
    seed_str = ""
    if retrieval["seed_entities"]:
        seed_str = f"\n**Query entities:** {', '.join(retrieval['seed_entities'][:5])}\n"
    mode_note = {
        "hybrid":      "Hybrid retrieval (vector + knowledge graph).",
        "graph_only":  "Knowledge graph only — no matching text chunks.",
        "vector_only": "Vector search only — no graph entities matched.",
        "none":        "No relevant information found.",
    }.get(retrieval["retrieval_mode"], "")
    return (
        f"Conversation History:\n{history}\n\n"
        f"[{retrieval['retrieval_mode'].upper()}] {mode_note}\n"
        f"{seed_str}{_GRAPHRAG_ADDENDUM}\n\n"
        f"Retrieved Knowledge:\n{retrieval['enriched_context']}"
    )


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    raw_name = file.filename or "upload.bin"
    filename = _sanitize_filename(raw_name)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB ({size_mb:.1f} MB)",
        )

    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.realpath(file_path).startswith(os.path.realpath(UPLOAD_DIR)):
        raise HTTPException(status_code=400, detail="Invalid filename")

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        extracted_text = parse_document(file_path)
        if not extracted_text or not extracted_text.strip():
            raise ValueError("Document is empty or could not be parsed.")

        # 1. Chunk
        chunks = semantic_chunking(extracted_text)

        # 2. Extract entities + relationships (3-layer)
        graph_data = extract_entities_and_relationships(extracted_text)
        dedup_map  = graph_data.pop("_dedup_map", {})

        # 3. Link entities to chunks BEFORE embedding
        chunks = link_entities_to_chunks(chunks, graph_data["entities"], dedup_map)

        # 4. Embed
        embedded_chunks = generate_embeddings(chunks)

        # 5. Store vectors (entity tags go into ChromaDB metadata)
        store_embeddings(filename, embedded_chunks)

        # 6. Legacy graph store
        store_knowledge_graph(graph_data)

        # 7. GraphRAG persistent graph
        ingest_document_graph(
            doc_id=filename,
            center=graph_data.get("center", {}),
            clusters=graph_data.get("clusters", {}),
            relationships=graph_data.get("relationships", []),
            doc_type=graph_data.get("doc_type", "general"),
        )

        # 8. Wiki
        wiki_generated = False
        try:
            generate_wiki_page(filename, extracted_text, _make_wiki_llm())
            wiki_generated = True
        except Exception as e:
            logger.warning("Wiki skipped for '%s': %s", filename, e)

        total_links = sum(len(c.get("entities", [])) for c in embedded_chunks)
        logger.info(
            "Uploaded '%s': %d chunks, %d entities, %d rels, %d links, %d merges, wiki=%s",
            filename, len(embedded_chunks), len(graph_data["entities"]),
            len(graph_data["relationships"]), total_links, len(dedup_map), wiki_generated,
        )

        return {
            "success":             True,
            "filename":            filename,
            "file_size_kb":        round(len(content) / 1024, 1),
            "chunks_stored":       len(embedded_chunks),
            "entities_found":      len(graph_data["entities"]),
            "relationships_found": len(graph_data["relationships"]),
            "entity_chunk_links":  total_links,
            "entity_dedup_merges": len(dedup_map),
            "wiki_generated":      wiki_generated,
            "message":             "Document processed successfully",
        }

    except Exception:
        logger.exception("Processing failed for '%s'", filename)
        _delete_upload_file(filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Processing failed. Check server logs.",
        )


# ── Documents ─────────────────────────────────────────────────────────────────

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
        raise HTTPException(status_code=400, detail="filename required")
    chunks_deleted     = delete_document(filename)
    file_deleted       = _delete_upload_file(filename)
    graph_nodes_removed = graph_remove_document(filename)
    delete_wiki_page(filename)
    return {
        "success": True, "filename": filename,
        "chunks_deleted": chunks_deleted,
        "graph_nodes_removed": graph_nodes_removed,
        "file_deleted": file_deleted,
        "message": f"'{filename}' removed",
    }


@router.post("/clear-documents")
async def clear_all():
    chunks_deleted = clear_all_documents()
    files_deleted = 0
    for path in glob.glob(os.path.join(UPLOAD_DIR, "*")):
        if os.path.isfile(path):
            try:
                os.remove(path); files_deleted += 1
            except OSError as e:
                logger.warning("Could not delete %s: %s", path, e)
    clear_knowledge_graph()
    graph_clear()
    clear_memory()
    wiki_cleared = clear_wiki_pages()
    return {
        "success": True,
        "chunks_deleted": chunks_deleted,
        "files_deleted": files_deleted,
        "wiki_pages_cleared": wiki_cleared,
        "memory_cleared": True,
        "message": "Knowledge base cleared.",
    }


# ── Search ────────────────────────────────────────────────────────────────────

@router.get("/search")
async def search(query: str):
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    qe = generate_embedding(q)
    results = semantic_search(qe)
    return {"success": True, "query": q, "total_results": len(results), "results": results}


# ── Chat (non-streaming) ──────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: dict):
    query = (request.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query required")

    add_message("User", query)
    history   = get_conversation_history()
    retrieval = hybrid_retrieve(query, top_k=5)

    if retrieval["retrieval_mode"] == "none":
        msg = ("I couldn't find relevant information in the uploaded documents. "
               "Try uploading more documents or rephrasing your question.")
        add_message("AI", msg)
        return {"success": True, "query": query, "response": msg,
                "sources": [], "retrieval_mode": "none", "seed_entities": []}

    ai_response = generate_graphrag_response(
        query=query,
        enriched_context=retrieval["enriched_context"],
        history=history,
        retrieval_mode=retrieval["retrieval_mode"],
        seed_entities=retrieval["seed_entities"],
    )
    add_message("AI", ai_response)
    return {
        "success": True, "query": query, "response": ai_response,
        "sources": format_sources(retrieval["vector_chunks"]),
        "retrieval_mode": retrieval["retrieval_mode"],
        "seed_entities": retrieval["seed_entities"],
        "conversation_history": history,
    }


# ── Chat (SSE streaming) ──────────────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(request: dict):
    query = (request.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query required")

    add_message("User", query)
    history   = get_conversation_history()
    retrieval = hybrid_retrieve(query, top_k=5)

    def _build_subgraph(seeds: list[str]) -> dict:
        nodes_map: dict[str, dict] = {}
        links: list[dict] = []
        seen_links: set[tuple] = set()
        for seed in seeds[:5]:
            neighbors = get_neighbors(seed, max_hops=2, max_nodes=30)
            nodes_map[seed] = {"id": seed, "isSeed": True}
            for nbr in neighbors:
                nid = nbr["node"]
                nodes_map.setdefault(nid, {"id": nid, "cluster": nbr["cluster"], "isSeed": False})
                path = nbr["path"]
                if len(path) >= 2:
                    key = (path[-2], path[-1])
                    if key not in seen_links:
                        seen_links.add(key)
                        links.append({
                            "source": path[-2], "target": path[-1],
                            "label": nbr["via"][0] if nbr["via"] else "→",
                        })
        return {"nodes": list(nodes_map.values()), "links": links}

    async def event_stream():
        if retrieval["retrieval_mode"] == "none":
            msg = ("I couldn't find relevant information in the uploaded documents. "
                   "Try uploading more documents or rephrasing your question.")
            yield f"data: {json.dumps({'type':'token','content':msg})}\n\n"
            add_message("AI", msg)
            yield f"data: {json.dumps({'type':'meta','retrieval_mode':'none','seed_entities':[],'sources':[],'subgraph':{'nodes':[],'links':[]}})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
            return

        key = os.getenv("NVIDIA_API_KEY")
        if not key:
            yield f"data: {json.dumps({'type':'error','content':'NVIDIA_API_KEY not configured.'})}\n\n"
            return

        ctx = _build_graphrag_context(query, retrieval, history)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1")
            full = ""
            stream = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[
                    {"role": "system", "content": NVIDIA_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion:\n{query}"},
                ],
                temperature=0.3, max_tokens=1024, stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full += delta
                    yield f"data: {json.dumps({'type':'token','content':delta})}\n\n"

            add_message("AI", full)
            subgraph = _build_subgraph(retrieval["seed_entities"]) if retrieval["seed_entities"] else {"nodes":[],"links":[]}
            yield f"data: {json.dumps({'type':'meta','retrieval_mode':retrieval['retrieval_mode'],'seed_entities':retrieval['seed_entities'],'sources':format_sources(retrieval['vector_chunks']),'subgraph':subgraph})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"

        except Exception as exc:
            logger.exception("Stream error for '%s'", query)
            yield f"data: {json.dumps({'type':'error','content':str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"X-Accel-Buffering":"no","Cache-Control":"no-cache","Connection":"keep-alive"},
    )


# ── Memory ────────────────────────────────────────────────────────────────────

@router.post("/clear-memory")
async def reset_memory():
    clear_memory()
    return {"success": True, "message": "Memory cleared"}


# ── Graph ─────────────────────────────────────────────────────────────────────

@router.get("/knowledge-graph")
async def knowledge_graph():
    data = get_full_graph_for_frontend()
    if data:
        return {"success": True, "graph": data}
    from app.services.knowledge_graph_service import get_knowledge_graph as lg
    return {"success": True, "graph": lg()}


@router.get("/graph/stats")
async def graph_stats():
    return {"success": True, "stats": get_graph_stats()}


@router.post("/graph/query")
async def graph_query(body: dict):
    entities = body.get("entities") or []
    hops = min(int(body.get("hops", 2)), 4)
    if not entities:
        raise HTTPException(status_code=400, detail="entities required")
    found, all_neighbors = [], {}
    for ent in entities[:5]:
        nbrs = get_neighbors(ent, max_hops=hops)
        if nbrs:
            found.append(ent)
            all_neighbors[ent] = [
                {"node": n["node"], "cluster": n["cluster"],
                 "distance": n["distance"], "via": n["via"],
                 "path": " → ".join(n["path"])}
                for n in nbrs[:20]
            ]
    paths = []
    if len(found) >= 2:
        for i, s1 in enumerate(found):
            for s2 in found[i+1:]:
                for p in find_paths(s1, s2, max_hops=hops+1)[:3]:
                    paths.append(" → ".join(p))
    return {"success": True, "entities_found": found,
            "neighbors": all_neighbors, "cross_paths": paths}


# ── Wiki ──────────────────────────────────────────────────────────────────────

@router.get("/wiki/pages")
async def get_wiki_pages():
    pages = list_wiki_pages()
    return {"success": True, "total": len(pages), "pages": pages}


@router.get("/wiki/page/{filename:path}")
async def get_single_wiki_page(filename: str):
    page = get_wiki_page(filename)
    if not page:
        raise HTTPException(status_code=404, detail=f"No wiki page for '{filename}'")
    return {"success": True, "page": page}


@router.post("/wiki/generate")
async def regenerate_wiki(body: dict):
    filename = _sanitize_filename((body.get("filename") or "").strip())
    if not filename:
        raise HTTPException(status_code=400, detail="filename required")
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.realpath(file_path).startswith(os.path.realpath(UPLOAD_DIR)):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
    try:
        text = parse_document(file_path)
        if not text or not text.strip():
            raise ValueError("Empty document")
        page = generate_wiki_page(filename, text, _make_wiki_llm())
        return {"success": True, "page": page}
    except Exception as exc:
        logger.exception("Wiki regen failed for '%s'", filename)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/wiki/page/{filename:path}")
async def remove_wiki_page(filename: str):
    if not delete_wiki_page(filename):
        raise HTTPException(status_code=404, detail="Wiki page not found")
    return {"success": True, "message": f"Deleted wiki for '{filename}'"}