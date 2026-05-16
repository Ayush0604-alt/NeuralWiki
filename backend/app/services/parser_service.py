"""
Parser Service — with page-level provenance
--------------------------------------------
Changes vs original:
  • parse_document() now returns a dict with `text` and `pages` keys instead
    of a plain string. Callers that expected a string can use result["text"].
  • PDF parsing uses PyMuPDF's page metadata: page number, section headings
    detected via font-size heuristic, and per-page char offsets so that
    chunking_service can map chunk positions back to page numbers later.
  • TXT/MD parsing returns the same structure with a single synthetic page.

Return schema
-------------
{
  "text": str,                   # full concatenated document text
  "pages": [                     # one entry per page / logical section
    {
      "page_number": int,        # 1-based
      "char_start": int,         # offset into `text` where this page begins
      "char_end": int,           # offset into `text` where this page ends
      "heading": str | None,     # largest-font line on this page (best-guess heading)
    }
  ],
  "title": str,                  # document title (from PDF metadata or filename)
  "author": str,                 # from PDF metadata, or ""
  "created_at": str,             # ISO timestamp from PDF metadata, or ""
  "total_pages": int,
}
"""

import os
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iso_date(raw) -> str:
    """Convert a PDF date string like 'D:20231015120000' to ISO format."""
    if not raw:
        return ""
    s = str(raw).strip().lstrip("D:").replace("'", "")
    try:
        # Try YYYYMMDDHHmmss
        return datetime.strptime(s[:14], "%Y%m%d%H%M%S").isoformat()
    except Exception:
        return s[:10]  # best effort


def _detect_heading(page_dict: dict) -> str | None:
    """
    Heuristic: the largest-font span on a page is probably its heading.
    page_dict comes from PyMuPDF page.get_text("dict").
    """
    best_size = 0.0
    best_text = None
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size", 0.0)
                text = span.get("text", "").strip()
                if size > best_size and len(text) > 3 and not re.fullmatch(r"[\d\s.]+", text):
                    best_size = size
                    best_text = text
    return best_text


# ──────────────────────────────────────────────────────────────────────────────
# PDF
# ──────────────────────────────────────────────────────────────────────────────

def _parse_pdf(file_path: str) -> dict:
    import fitz  # PyMuPDF

    pdf = fitz.open(file_path)
    meta = pdf.metadata or {}

    full_text_parts: list[str] = []
    pages_meta: list[dict] = []
    char_cursor = 0

    for page in pdf:
        page_text = page.get_text()  # plain text for this page

        # Detect heading via dict (font sizes)
        try:
            page_dict = page.get_text("dict")
            heading = _detect_heading(page_dict)
        except Exception:
            heading = None

        char_start = char_cursor
        char_end = char_cursor + len(page_text)
        char_cursor = char_end + 1  # +1 for the '\n' separator below

        pages_meta.append({
            "page_number": page.number + 1,  # 1-based
            "char_start":  char_start,
            "char_end":    char_end,
            "heading":     heading,
        })
        full_text_parts.append(page_text)

    pdf.close()

    full_text = "\n".join(full_text_parts)

    return {
        "text":        full_text,
        "pages":       pages_meta,
        "title":       meta.get("title", "") or os.path.splitext(os.path.basename(file_path))[0],
        "author":      meta.get("author", "") or "",
        "created_at":  _iso_date(meta.get("creationDate", "")),
        "total_pages": len(pages_meta),
    }


# ──────────────────────────────────────────────────────────────────────────────
# TXT / MD
# ──────────────────────────────────────────────────────────────────────────────

def _parse_text(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    # Try to infer a title from the first heading or first non-empty line
    title = os.path.splitext(os.path.basename(file_path))[0]
    for line in text.splitlines():
        stripped = line.lstrip("#").strip()
        if stripped:
            title = stripped[:80]
            break

    stat = os.stat(file_path)
    created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()

    return {
        "text":        text,
        "pages": [{
            "page_number": 1,
            "char_start":  0,
            "char_end":    len(text),
            "heading":     title,
        }],
        "title":       title,
        "author":      "",
        "created_at":  created_at,
        "total_pages": 1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def parse_document(file_path: str) -> dict | None:
    """
    Parse a document and return a rich metadata dict.

    Returns None if the extension is unsupported.

    The `text` key holds the full concatenated text (string), which is what
    chunking_service, knowledge_graph_service, etc. previously received.

    Callers that previously did:
        text = parse_document(path)
    should now do:
        result = parse_document(path)
        text = result["text"]
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        result = _parse_pdf(file_path)
    elif ext in (".txt", ".md"):
        result = _parse_text(file_path)
    else:
        logger.warning("Unsupported extension: %s", ext)
        return None

    if not result["text"].strip():
        logger.warning("Empty document after parsing: %s", file_path)
        return None

    logger.info(
        "Parsed '%s': %d chars, %d pages, title='%s'",
        os.path.basename(file_path),
        len(result["text"]),
        result["total_pages"],
        result["title"][:50],
    )
    return result