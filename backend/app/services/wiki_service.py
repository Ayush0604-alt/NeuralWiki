"""
Wiki Service
------------
Generates and stores LLM-powered wiki pages per uploaded document.
Each wiki page is a structured Markdown summary with sections:
  Overview, Key Concepts, Entities, Key Facts, Relationships
"""

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# In-memory wiki store: filename → wiki page dict
_wiki_store: dict[str, dict] = {}

WIKI_SYSTEM_PROMPT = """You are NeuralWiki, an expert knowledge base curator.
Given the full text of a document, generate a structured wiki page in Markdown.

Your wiki page MUST follow this exact structure:

## Overview
2-3 sentences summarizing what this document is about and its main purpose.

## Key Concepts
List the 4-8 most important concepts, ideas, or themes in this document.
- **Concept Name**: Brief explanation (1-2 sentences)

## Notable Entities
List important people, organizations, places, products, or technologies mentioned.
- **Entity Name** *(type)*: Role or significance in the document

## Key Facts & Findings
The most important factual claims, data points, or conclusions from the document.
1. First key fact or finding
2. Second key fact or finding
(list 4-8 items)

## Relationships & Connections
How the main entities and concepts relate to each other. Describe 3-5 meaningful relationships.

## Quick Reference
| Attribute | Value |
|-----------|-------|
| Document Type | (report/article/technical/research/etc) |
| Primary Topic | (main subject) |
| Complexity | (basic/intermediate/advanced) |
| Key Terms | term1, term2, term3 |

Keep the wiki page concise, accurate, and focused on what's in the document only.
Do NOT invent information not present in the text.
"""


def generate_wiki_page(filename: str, text: str, llm_fn) -> dict:
    """
    Generate a wiki page for a document using the provided LLM function.
    llm_fn(query, context) -> str
    """
    # Truncate to first 12000 chars to keep prompt manageable
    truncated = text[:12000]
    if len(text) > 12000:
        truncated += "\n\n[Document truncated for wiki generation]"

    try:
        content = llm_fn(
            query="Generate a comprehensive wiki page for this document following the exact structure specified.",
            context=f"Document filename: {filename}\n\nFull document text:\n{truncated}"
        )

        page = {
            "filename": filename,
            "title": _title_from_filename(filename),
            "content": content,
            "generated_at": datetime.now().isoformat(),
            "char_count": len(text),
            "word_count": len(text.split()),
        }

        _wiki_store[filename] = page
        logger.info("Generated wiki page for '%s' (%d words)", filename, page["word_count"])
        return page

    except Exception as e:
        logger.error("Wiki generation failed for '%s': %s", filename, e)
        raise


def get_wiki_page(filename: str) -> dict | None:
    return _wiki_store.get(filename)


def list_wiki_pages() -> list[dict]:
    """Return all wiki pages, sorted by filename."""
    return [
        {
            "filename": p["filename"],
            "title": p["title"],
            "generated_at": p["generated_at"],
            "word_count": p["word_count"],
            "preview": _extract_overview(p["content"]),
        }
        for p in sorted(_wiki_store.values(), key=lambda x: x["filename"])
    ]


def delete_wiki_page(filename: str) -> bool:
    if filename in _wiki_store:
        del _wiki_store[filename]
        return True
    return False


def clear_wiki_pages() -> int:
    count = len(_wiki_store)
    _wiki_store.clear()
    return count


def _title_from_filename(filename: str) -> str:
    """Convert filename to readable title."""
    name = os.path.splitext(filename)[0]
    name = name.replace("_", " ").replace("-", " ")
    return name.title()


def _extract_overview(content: str) -> str:
    """Extract the overview section as a preview."""
    lines = content.split("\n")
    in_overview = False
    overview_lines = []
    for line in lines:
        if "## Overview" in line:
            in_overview = True
            continue
        if in_overview:
            if line.startswith("## "):
                break
            if line.strip():
                overview_lines.append(line.strip())
    preview = " ".join(overview_lines)
    return preview[:200] + "…" if len(preview) > 200 else preview