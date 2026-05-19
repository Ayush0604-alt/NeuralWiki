"""
LLM Manager — GraphRAG-aware
----------------------------
Generates responses with enriched context (vector chunks + graph triples
+ multi-hop reasoning paths). The system prompt addendum tells the model
how to use graph relationships for multi-hop inference.
"""

from app.services.llm.nvidia_provider import generate_nvidia_response

_MISSING_INFO_SENTENCE = "The uploaded documents do not contain information about this topic."
_GRAPH_SECTION_PREFIXES = (
    "Supporting Details",
    "Key Details from the Context",
    "Key Details",
    "Entity Relationships",
    "Relationships",
    "Multi-hop Reasoning Paths",
    "Multi-hop",
    "Key Entities",
    "Key Entities Involved",
    "Key Entities in Context",
    "Entities:",
    "entities:",
    "Graph only",
    "Sources",
    "Source",
    "Detected query entities",
)


def _sanitize_response(text: str) -> str:
    if not text:
        return text
    stripped = text.strip()
    if stripped == _MISSING_INFO_SENTENCE:
        return stripped
    variants = [
        _MISSING_INFO_SENTENCE,
        f"> {_MISSING_INFO_SENTENCE}",
        f">{_MISSING_INFO_SENTENCE}",
    ]
    if any(v in text for v in variants):
        cleaned = text
        for v in variants:
            cleaned = cleaned.replace(v, "")
        cleaned = cleaned.strip()
        text = cleaned if cleaned else stripped

    lines = text.splitlines()
    out: list[str] = []
    skipping = False
    for line in lines:
        stripped_line = line.strip()
        if any(stripped_line.startswith(p) for p in _GRAPH_SECTION_PREFIXES):
            skipping = True
            continue
        if skipping:
            if not stripped_line:
                continue
            if stripped_line.startswith(("-", "•", "●", "+")):
                continue
            if stripped_line.endswith(":"):
                continue
            skipping = False
        if not skipping:
            out.append(line)

    cleaned = "\n".join(out).strip()
    return cleaned if cleaned else text


# Extra guidance injected when graph context is present
_GRAPHRAG_ADDENDUM = """
You have access to knowledge graph context. Use it to provide accurate, 
well-informed answers. Focus on answering the user's question clearly and concisely.
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
    response = generate_nvidia_response(query, final_context)
    return _sanitize_response(response)


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
    response = generate_nvidia_response(query, final_context)
    return _sanitize_response(response)