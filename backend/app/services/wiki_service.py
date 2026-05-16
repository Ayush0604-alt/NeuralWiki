"""
Wiki Service
------------
Generates and stores LLM-powered wiki pages per uploaded document.

Changes vs original:
  • Pages are persisted to disk (JSON files in WIKI_DIR) so they survive
    backend restarts. The in-memory dict is now a write-through cache.
  • generate_wiki_page() is still synchronous but upload.py should call it
    via FastAPI BackgroundTasks (see upload.py) to avoid blocking responses.
"""

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

WIKI_DIR = os.getenv("WIKI_PERSIST_PATH", "chroma_db/wiki_pages")
os.makedirs(WIKI_DIR, exist_ok=True)

# In-memory write-through cache: filename → wiki page dict
_wiki_store: dict[str, dict] = {}


def _page_path(filename: str) -> str:
    safe = filename.replace("/", "_").replace("\\", "_")
    return os.path.join(WIKI_DIR, safe + ".json")


def _load_all_from_disk() -> None:
    """Populate the in-memory cache from disk on first access."""
    if _wiki_store:
        return
    for fname in os.listdir(WIKI_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(WIKI_DIR, fname), "r", encoding="utf-8") as f:
                page = json.load(f)
            _wiki_store[page["filename"]] = page
        except Exception as exc:
            logger.warning("Could not load wiki page '%s': %s", fname, exc)


def _save_page(page: dict) -> None:
    try:
        with open(_page_path(page["filename"]), "w", encoding="utf-8") as f:
            json.dump(page, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Could not persist wiki page '%s': %s", page["filename"], exc)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_wiki_page(filename: str, text: str, llm_fn) -> dict:
    """
    Generate a wiki page for a document using the provided LLM function.
    llm_fn(query, context) -> str
    """
    truncated = text[:12000]
    if len(text) > 12000:
        truncated += "\n\n[Document truncated for wiki generation]"

    content = llm_fn(
        query="Generate a comprehensive wiki page for this document following the exact structure specified.",
        context=f"Document filename: {filename}\n\nFull document text:\n{truncated}",
    )

    page = {
        "filename":     filename,
        "title":        _title_from_filename(filename),
        "content":      content,
        "generated_at": datetime.now().isoformat(),
        "char_count":   len(text),
        "word_count":   len(text.split()),
    }

    _wiki_store[filename] = page
    _save_page(page)
    logger.info("Generated wiki page for '%s' (%d words)", filename, page["word_count"])
    return page


def get_wiki_page(filename: str) -> dict | None:
    _load_all_from_disk()
    return _wiki_store.get(filename)


def list_wiki_pages() -> list[dict]:
    _load_all_from_disk()
    return [
        {
            "filename":     p["filename"],
            "title":        p["title"],
            "generated_at": p["generated_at"],
            "word_count":   p["word_count"],
            "preview":      _extract_overview(p["content"]),
        }
        for p in sorted(_wiki_store.values(), key=lambda x: x["filename"])
    ]


def delete_wiki_page(filename: str) -> bool:
    _load_all_from_disk()
    if filename not in _wiki_store:
        return False
    del _wiki_store[filename]
    path = _page_path(filename)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning("Could not delete wiki file '%s': %s", path, exc)
    return True


def clear_wiki_pages() -> int:
    _load_all_from_disk()
    count = len(_wiki_store)
    _wiki_store.clear()
    for fname in list(os.listdir(WIKI_DIR)):
        if fname.endswith(".json"):
            try:
                os.remove(os.path.join(WIKI_DIR, fname))
            except OSError:
                pass
    return count


# ── Helpers ────────────────────────────────────────────────────────────────────

def _title_from_filename(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    return name.replace("_", " ").replace("-", " ").title()


def _extract_overview(content: str) -> str:
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
    return (preview[:200] + "…") if len(preview) > 200 else preview