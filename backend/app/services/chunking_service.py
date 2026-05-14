import re


def semantic_chunking(text, chunk_size=500):

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If adding paragraph stays within limit
        if len(current_chunk) + len(paragraph) < chunk_size:

            current_chunk += paragraph + "\n\n"

        else:

            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"

    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    structured_chunks = []

    for index, chunk in enumerate(chunks):

        structured_chunks.append({
            "chunk_id": index + 1,
            "content": chunk,
            "length": len(chunk)
        })

    return structured_chunks