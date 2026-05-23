# NeuralWiki

A GraphRAG-powered document knowledge system that combines vector search with knowledge graph traversal. Upload documents, ask questions, explore entity relationships, and generate AI wiki pages — all backed by a multi-layered extraction pipeline.

---

## What It Does

NeuralWiki ingests documents (PDF, TXT, Markdown) and builds two complementary knowledge structures:

1. **Vector store (ChromaDB)** — chunks are embedded with `sentence-transformers` for semantic similarity search.
2. **Knowledge graph (NetworkX)** — entities and relationships are extracted via a four-layer pipeline, then stored as a directed multigraph with community detection.

At query time, a hybrid retrieval system finds seed entities in the graph, boosts relevant chunks via entity-filtered vector search, and passes enriched context to an LLM for generation. Responses stream token-by-token to the frontend via SSE.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    React Frontend                    │
│  Chat · Wiki · Knowledge Graph · Upload              │
└────────────────────────┬────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────┐
│               FastAPI Backend                        │
│                                                      │
│  /upload  →  Parser → Chunker → Embedder             │
│              Entity Extractor → Graph Ingestor       │
│              Wiki Generator (background)             │
│                                                      │
│  /chat/stream  →  Hybrid Retrieval → LLM → SSE       │
│                                                      │
│  /knowledge-graph  →  Graph export (flat + legacy)   │
│  /wiki/*           →  Wiki CRUD                      │
└─────────┬──────────────────────┬────────────────────┘
          │                      │
    ┌─────▼──────┐        ┌──────▼──────┐
    │  ChromaDB  │        │  NetworkX   │
    │  (vectors) │        │  (graph)    │
    └────────────┘        └─────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite 8, D3.js v7, react-force-graph |
| Backend | FastAPI, Python 3.11+ |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent) |
| Graph | NetworkX (MultiDiGraph, persisted as JSON) |
| NER / dep-parse | spaCy `en_core_web_sm` |
| PDF parsing | PyMuPDF (fitz) |
| Primary LLM (chat) | NVIDIA API (llama-3.1-nemotron-nano-8b-v1 default) |
| Primary LLM (extraction) | Mistral API (`mistral-small-latest` default) |
| Wiki LLM | xAI / Grok (`grok-2-latest` default) |
| Fallbacks | Gemini (`gemini-2.5-flash`) |

---

## Project Structure

```
├── backend/
│   └── app/
│       ├── main.py                   # FastAPI app, CORS, router
│       └── api/
│           └── upload.py             # All API routes
│       └── services/
│           ├── parser_service.py     # PDF + text parsing w/ page metadata
│           ├── chunking_service.py   # Sentence-aware chunking w/ overlap
│           ├── embedding_service.py  # sentence-transformers wrapper
│           ├── vector_service.py     # ChromaDB CRUD + entity-filtered search
│           ├── knowledge_graph_service.py  # 4-layer entity/rel extraction
│           ├── graph_rag_service.py  # NetworkX graph, BFS, community detection
│           ├── hybrid_retrieval_service.py # GraphRAG retrieval pipeline
│           ├── memory_service.py     # Session-isolated conversation memory
│           ├── wiki_service.py       # Wiki generation + disk persistence
│           └── llm/
│               ├── llm_manager.py        # GraphRAG-aware response builder
│               ├── nvidia_provider.py    # NVIDIA / OpenAI-compat chat
│               └── gemini_provider.py    # Google Gemini fallback
├── frontend/
│   └── src/
│       ├── App.jsx                   # Shell, tab nav, event bus
│       ├── app.css                   # Design system (CSS variables, components)
│       ├── components/
│       │   ├── ChatBox.jsx           # Streaming chat w/ D3 subgraph panel
│       │   ├── KnowledgeGraph.jsx    # Full D3 force graph w/ community coloring
│       │   └── Wiki.jsx              # Wiki list + page viewer
│       └── pages/
│           └── Upload.jsx            # Upload, doc list, clear KB
└── .gitignore
```

---

## Document Processing Pipeline

When a file is uploaded, the following steps run in order:

1. **Parse** (`parser_service`) — extracts text plus page-level metadata (page number, heading via font-size heuristic, char offsets). Returns a rich dict instead of a plain string.

2. **Chunk** (`chunking_service`) — splits on spaCy sentence boundaries with configurable overlap (`overlap_sentences=3`). Target ~400 tokens per chunk. Falls back to regex sentence splitting if spaCy is unavailable.

3. **Annotate** (`upload.py`) — maps each chunk back to its source page number and section heading using keyword overlap with page headings.

4. **Extract entities & relationships** (`knowledge_graph_service`) — four-layer pipeline:
   - **Layer 0 (LLM primary)** — Mistral API extracts domain entities in windowed passes. Catches technical terms, legal terms, product names, etc. that spaCy misses.
   - **Layer 1 (spaCy NER + noun chunks)** — supplements with named entities and TECH/CONCEPT/ACTION noun chunks.
   - **Layer 2 (SVO)** — dependency-parse verb triples (`subject → verb → object`).
   - **Layer 3 (Co-occurrence)** — sentence-level co-occurrence with a minimum frequency threshold.
   - **Layer 4 (LLM relationships)** — Mistral extracts semantic relationships between the now-complete entity set (causality, hierarchy, dependency, etc.).
   - Falls back to NVIDIA then Gemini if Mistral fails or rate-limits.

5. **Link entities to chunks** — every chunk is tagged with the canonical entities it mentions. These tags travel through to ChromaDB metadata.

6. **Embed** (`embedding_service`) — `all-MiniLM-L6-v2` produces 384-dim vectors for each chunk.

7. **Store embeddings** (`vector_service`) — ChromaDB persists chunks with full metadata: source, chunk_id, entities, page_number, section_heading, document_title, author, created_at, sentence_start, sentence_end.

8. **Ingest graph** (`graph_rag_service`) — entities and relationships are added to the NetworkX MultiDiGraph. Community detection (greedy modularity) runs after each ingestion and is cached.

9. **Generate wiki** (background task) — an LLM produces a structured Markdown wiki page (Overview, Key Concepts, Notable Entities, Key Facts, Relationships, Quick Reference table). Persisted to `chroma_db/wiki_pages/`.

---

## Retrieval Pipeline

For each query the `/chat/stream` endpoint:

1. **Finds seed entities** in the graph (`find_query_entities`) via exact + partial node-id matching against the query string, ranked by degree.

2. **Entity-filtered vector search** (`entity_filtered_search`) — retrieves a wide candidate pool, then applies a rarity-weighted boost to chunks containing seed entities. Adjusted scores drive final ranking.

3. **Graph context** (`build_graph_context`) — BFS from seed entities up to 2 hops, collecting triples (`A → [label] → B`) and multi-hop reasoning paths.

4. **Retrieval mode** — `hybrid` (both), `vector_only`, `graph_only`, or `none`.

5. **LLM generation** — enriched context (chunks + graph triples) is sent to NVIDIA with a structured system prompt. Falls back to Gemini.

6. **SSE streaming** — tokens are streamed as `data: {"type":"token","content":"..."}` frames. A final `meta` frame carries sources, retrieval mode, seed entities, and a subgraph for visualization.

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- spaCy model: `python -m spacy download en_core_web_sm`

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create `backend/.env`:

```env
# Chat LLM (required for chat responses)
CHAT_API_KEY=nvapi-...
CHAT_API_BASE_URL=https://integrate.api.nvidia.com/v1
CHAT_MODEL=nvidia/llama-3.1-nemotron-nano-8b-v1

# Entity extraction LLM (required for knowledge graph)
EXTRACT_API_KEY=...
EXTRACT_API_BASE_URL=https://api.mistral.ai/v1
EXTRACT_MODEL=mistral-small-latest

# Wiki generation LLM (optional, falls back to NVIDIA)
WIKI_API_KEY=...
WIKI_API_BASE_URL=https://api.x.ai/v1
WIKI_MODEL=grok-2-latest

# Gemini fallback (optional)
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash

# Optional: secondary NVIDIA key for fallback
CHAT_API_KEY_FALLBACK=nvapi-...
EXTRACT_API_KEY_FALLBACK=nvapi-...
EXTRACT_API_BASE_URL_FALLBACK=https://integrate.api.nvidia.com/v1
EXTRACT_MODEL_FALLBACK=nvidia/llama-3.1-nemotron-nano-8b-v1
```

Start the backend:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload and process a document. Params: `background` (bool), `fast` (skip LLM extraction) |
| `GET` | `/documents` | List all indexed documents |
| `POST` | `/delete-document` | Remove one document `{"filename": "..."}` |
| `POST` | `/clear-documents` | Wipe entire knowledge base |
| `POST` | `/chat` | Non-streaming chat |
| `POST` | `/chat/stream` | SSE streaming chat |
| `POST` | `/clear-memory` | Clear conversation history |
| `GET` | `/knowledge-graph` | Export graph (flat + legacy format) |
| `GET` | `/graph/stats` | Node/edge counts, communities, top entities |
| `POST` | `/graph/query` | BFS neighbors + cross-entity paths |
| `GET` | `/wiki/pages` | List all wiki pages |
| `GET` | `/wiki/page/{filename}` | Fetch single wiki page |
| `POST` | `/wiki/generate` | Regenerate wiki for a file |
| `DELETE` | `/wiki/page/{filename}` | Delete wiki page |

Upload options:

```bash
# Background processing (returns 202 immediately)
curl -F "file=@doc.pdf" "http://localhost:8000/upload?background=true"

# Fast mode (skips LLM extraction, spaCy only)
curl -F "file=@doc.pdf" "http://localhost:8000/upload?fast=true"
```

---

## Frontend Features

### Chat

- Streams responses token-by-token via SSE.
- Retrieval mode badge shows `hybrid`, `vector_only`, or `graph_only`.
- "Show reasoning subgraph" button renders an interactive D3 mini-graph of nodes traversed during retrieval — draggable, zoomable.
- Full Markdown rendering: headings, bold/italic, code blocks with language tags and copy button, tables, ordered/unordered lists, blockquotes.

### Knowledge Graph

- Full-screen D3 force-directed graph.
- Node color = **community** (greedy modularity communities, up to 12 distinct colors).
- Node size = degree + betweenness centrality (hub connectors get a ★ badge).
- Edge style encodes extraction method: thick solid (LLM semantic), medium solid (SVO), thin (co-occurrence), dashed (bridge).
- Cross-community edges rendered dashed for visibility.
- Filter by cluster type (ENTITY, TECH, CONCEPT, etc.), edge strength (all / SVO+ / semantic), or node search.
- Click a node to highlight its neighborhood; "Focus selection" hides unrelated nodes.
- Hover tooltip shows cluster, community, degree, betweenness centrality, and up to 5 edge relationships.
- Community legend and edge-type legend always visible.
- Fit-to-canvas button, manual refresh.

### Wiki

- Lists all generated wiki pages with a preview of the Overview section.
- Shows which uploaded documents are missing wiki pages with a one-click generate button.
- Full page view with structured Markdown rendering.
- Regenerate button to refresh any page with updated content.
- Wiki pages persist across backend restarts (stored as JSON in `chroma_db/wiki_pages/`).

### Upload

- Drag-and-drop or click-to-browse.
- Real-time progress bar with animated status.
- Post-upload stats: chunks stored, entities found, relationships extracted.
- Scrollable knowledge base list showing all indexed documents with chunk counts.
- Per-document delete with confirmation modal.
- "Clear knowledge base" button with confirmation — wipes vectors, graph, files, conversation memory, and wiki pages.
- Activity log with timestamped entries.

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `CHAT_API_KEY` | — | NVIDIA (or OpenAI-compat) API key for chat |
| `CHAT_API_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Chat API base URL |
| `CHAT_MODEL` | `nvidia/llama-3.1-nemotron-nano-8b-v1` | Chat model ID |
| `EXTRACT_API_KEY` | — | Mistral API key for entity extraction |
| `EXTRACT_API_BASE_URL` | `https://api.mistral.ai/v1` | Extraction API base URL |
| `EXTRACT_MODEL` | `mistral-small-latest` | Extraction model ID |
| `WIKI_API_KEY` | — | xAI/Grok key for wiki generation |
| `WIKI_API_BASE_URL` | `https://api.x.ai/v1` | Wiki API base URL |
| `WIKI_MODEL` | `grok-2-latest` | Wiki model ID |
| `GEMINI_API_KEY` | — | Google Gemini key (fallback) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model ID |
| `GRAPH_PERSIST_PATH` | `chroma_db/knowledge_graph.json` | Graph persistence path |
| `WIKI_PERSIST_PATH` | `chroma_db/wiki_pages` | Wiki pages directory |

---

## Entity Extraction Details

Entities are deduplicated using substring containment and optional fuzzy matching (`rapidfuzz` / `thefuzz` if installed). A canonical form is chosen by frequency, and a `_dedup_map` is returned so chunks can reference the same canonical entity.

Relationship weights encode extraction method:
- `≥ 3.0` — LLM semantic (Mistral / NVIDIA / Gemini)
- `≥ 2.0` — SVO verb triple (spaCy dep-parse)
- `≥ 1.0` — Co-occurrence
- `< 1.0` — Bridge edge (added to connect disconnected components)

Disconnected graph components are automatically bridged by connecting the highest-degree node from each component to the main component.

---

## Data Persistence

| Data | Location |
|---|---|
| ChromaDB vectors | `backend/chroma_db/` |
| Knowledge graph | `backend/chroma_db/knowledge_graph.json` |
| Wiki pages | `backend/chroma_db/wiki_pages/` |
| Uploaded files | `backend/uploads/` |
| Conversation memory | In-memory only (cleared on restart) |

---

## Rate Limiting

The `/chat/stream` endpoint enforces a token-bucket rate limiter: 10 requests per 60 seconds per IP. Requests exceeding the limit receive HTTP 429.

---

## Security Notes

- Filenames are sanitized to remove path traversal characters (`[^\w.\-]`).
- All file writes are validated against the resolved upload directory path.
- Allowed file extensions: `.pdf`, `.txt`, `.md`.
- Max file size: 50 MB.

---

## Known Limitations

- Conversation memory is in-memory only and resets when the backend restarts.
- LLM extraction is rate-limited by upstream providers; large documents may fall back to spaCy-only extraction.
- The rate limiter is per-process and does not persist across restarts.
- The `fast=true` upload flag skips all LLM extraction, which significantly reduces graph quality on technical documents.
