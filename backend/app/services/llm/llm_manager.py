"""
LLM Manager — GraphRAG-aware
----------------------------
Generates responses with enriched context (vector chunks + graph triples
+ multi-hop reasoning paths). The system prompt addendum tells the model
how to use graph relationships for multi-hop inference.
"""

from app.services.llm.nvidia_provider import generate_nvidia_response


# Extra guidance injected when graph context is present
_GRAPHRAG_ADDENDUM = """
You also have access to a **Knowledge Graph context** section containing:
- Entity relationships as triples:  `Entity A → [relationship] → Entity B`
- Multi-hop reasoning paths connecting entities across documents
- Key entity metadata (type, frequency)

**How to use the Knowledge Graph:**
- Use triples to reason about HOW entities relate to each other
- Follow multi-hop paths to answer questions that require chaining facts
  (e.g. if A → [uses] → B and B → [integrates_with] → C, then A indirectly
   relates to C)
- Cite graph relationships when they directly support your answer
- If graph triples and text chunks conflict, prefer the text chunks
"""


def generate_ai_response(query: str, context: str, history: str) -> str:
    """
    Standard backward-compatible response.
    Context may already be GraphRAG-enriched by hybrid_retrieval_service.
    """
    final_context = f"""Conversation History:
{history}

Retrieved Knowledge:
{context}
"""
    return generate_nvidia_response(query, final_context)


def generate_graphrag_response(
    query: str,
    enriched_context: str,
    history: str,
    retrieval_mode: str = "hybrid",
    seed_entities: list | None = None,
) -> str:
    """
    GraphRAG-aware response generation with graph context guidance.
    Called by the /chat endpoint when graph data is available.
    """
    seed_str = ""
    if seed_entities:
        seed_str = f"\n**Detected query entities:** {', '.join(seed_entities[:5])}\n"

    mode_note = {
        "hybrid":      "Using hybrid retrieval (vector search + knowledge graph traversal).",
        "graph_only":  "Using knowledge graph only — no matching text chunks found.",
        "vector_only": "Using vector search only — no graph entities matched the query.",
        "none":        "No relevant information found in the knowledge base.",
    }.get(retrieval_mode, "")

    final_context = f"""Conversation History:
{history}

[Retrieval Mode: {retrieval_mode.upper()}] {mode_note}
{seed_str}
{_GRAPHRAG_ADDENDUM}

Retrieved Knowledge (Vector Chunks + Knowledge Graph Triples):
{enriched_context}
"""
    return generate_nvidia_response(query, final_context)