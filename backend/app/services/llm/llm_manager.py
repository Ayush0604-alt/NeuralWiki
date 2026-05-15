"""
LLM Manager — GraphRAG-aware
----------------------------
Passes enriched context (vector + graph) to the LLM with a system prompt
that instructs it to use relationship/path information for multi-hop reasoning.
"""

from app.services.llm.nvidia_provider import generate_nvidia_response


GRAPHRAG_SYSTEM_ADDENDUM = """
You also have access to a **Knowledge Graph context** section that contains:
- Entity relationships as triples: `Entity A → [relationship] → Entity B`
- Multi-hop reasoning paths connecting entities
- Key entity metadata

**How to use the Knowledge Graph:**
- Use relationships to reason about HOW entities are connected
- Follow multi-hop paths to answer questions that require chaining facts
- When the graph shows a path like `A → [uses] → B → [integrates_with] → C`, you can infer A indirectly relates to C
- Cite graph relationships when they support your answer
- If graph and text chunks conflict, prefer the text chunks (more detailed)
"""


def generate_ai_response(query: str, context: str, history: str) -> str:
    """
    Standard response (backward compatible).
    Context may already be enriched by hybrid_retrieval_service.
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
    seed_entities: list = None,
) -> str:
    """
    GraphRAG-aware response generation.
    Injects graph context and guides the model to use it.
    """
    seed_str = ""
    if seed_entities:
        seed_str = f"\n**Detected query entities:** {', '.join(seed_entities[:5])}\n"

    mode_note = {
        "hybrid": "Using hybrid retrieval (vector search + knowledge graph).",
        "graph_only": "Using knowledge graph retrieval only (no matching text chunks found).",
        "vector_only": "Using vector search only (no graph entities matched query).",
        "none": "No relevant information found in the knowledge base.",
    }.get(retrieval_mode, "")

    final_context = f"""Conversation History:
{history}

[Retrieval Mode: {retrieval_mode.upper()}] {mode_note}
{seed_str}
{GRAPHRAG_SYSTEM_ADDENDUM}

Retrieved Knowledge (Vector Chunks + Knowledge Graph):
{enriched_context}
"""

    return generate_nvidia_response(query, final_context)