from app.services.llm.nvidia_provider import (
    generate_nvidia_response
)


def generate_ai_response(
    query,
    context,
    history
):

    final_context = f"""
Conversation History:
{history}

Retrieved Knowledge:
{context}
"""

    return generate_nvidia_response(
        query,
        final_context
    )