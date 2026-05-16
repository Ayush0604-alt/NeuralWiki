"""
Chunking Service — sentence-aware with overlap
-----------------------------------------------
Replaces raw character-count splitting with spaCy sentence boundaries.

Key changes vs original:
  • Splits on spaCy sentence boundaries instead of double-newlines + character count.
  • Configurable overlap: the last `overlap_sentences` sentences of each chunk
    are carried into the start of the next chunk, so entities that span a
    sentence boundary are never split without context.
  • Falls back gracefully to paragraph splitting if spaCy isn't available.
  • Chunk metadata includes `sentence_start` and `sentence_end` indices for
    provenance (useful when you add page-number metadata in Fix 2).

Usage (unchanged from call-sites):
    chunks = semantic_chunking(text)

Each returned chunk dict has the same keys as before:
    {chunk_id, content, length, entities}
Plus two new keys (ignored safely by downstream if not needed):
    {sentence_start, sentence_end}
"""

import logging
import re

logger = logging.getLogger(__name__)

# ── Attempt to reuse the spaCy model already loaded in knowledge_graph_service ─
# Loading a second copy wastes ~500 MB RAM, so we share the module-level model.
try:
    from app.services.knowledge_graph_service import nlp as _nlp
    _SPACY_AVAILABLE = True
except Exception:
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        _SPACY_AVAILABLE = True
    except Exception:
        _nlp = None
        _SPACY_AVAILABLE = False
        logger.warning(
            "spaCy not available for chunking — falling back to paragraph splitting. "
            "Run: python -m spacy download en_core_web_sm"
        )


def _split_into_sentences(text: str) -> list[str]:
    """Return a list of sentence strings using spaCy sentence boundaries."""
    if not _SPACY_AVAILABLE or _nlp is None:
        # Regex fallback: split on sentence-ending punctuation
        raw = re.split(r"(?<=[.!?])\s+", text.replace("\n\n", " "))
        return [s.strip() for s in raw if s.strip()]

    # spaCy: disable heavy components we don't need here
    doc = _nlp(text[:200_000])  # cap to avoid OOM on huge docs
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def semantic_chunking(
    text: str,
    target_chunk_tokens: int = 400,   # ~400 words per chunk
    overlap_sentences: int = 3,        # sentences carried into the next chunk
    chars_per_token: float = 4.5,      # rough approximation
) -> list[dict]:
    """
    Sentence-aware chunking with configurable sentence overlap.

    Parameters
    ----------
    text : str
        Full document text.
    target_chunk_tokens : int
        Approximate target size in tokens (default 400 ≈ ~1800 chars).
    overlap_sentences : int
        Number of sentences from the end of chunk N to prepend to chunk N+1.
        This preserves cross-boundary entity context.
    chars_per_token : float
        Conversion factor used to estimate token count from character count.

    Returns
    -------
    list[dict]
        Each dict: {chunk_id, content, length, entities, sentence_start, sentence_end}
    """
    target_chars = int(target_chunk_tokens * chars_per_token)

    sentences = _split_into_sentences(text)
    if not sentences:
        logger.warning("No sentences extracted from document")
        return []

    logger.info("Chunking: %d sentences, target %d chars/chunk, overlap %d sentences",
                len(sentences), target_chars, overlap_sentences)

    chunks: list[dict] = []
    chunk_id = 1

    i = 0
    while i < len(sentences):
        current_sentences: list[str] = []
        current_chars = 0
        start_idx = i

        while i < len(sentences):
            s = sentences[i]
            s_len = len(s)

            # Always include at least one sentence per chunk
            if current_sentences and current_chars + s_len > target_chars:
                break

            current_sentences.append(s)
            current_chars += s_len + 1  # +1 for the space/newline
            i += 1

        if not current_sentences:
            # Safety: advance if somehow stuck
            i += 1
            continue

        content = " ".join(current_sentences)
        end_idx = i - 1

        chunks.append({
            "chunk_id":       chunk_id,
            "content":        content,
            "length":         len(content),
            "entities":       [],            # filled by link_entities_to_chunks()
            "sentence_start": start_idx,
            "sentence_end":   end_idx,
        })
        chunk_id += 1

        # ── Overlap: rewind by overlap_sentences so the next chunk starts
        # with the last few sentences of the current one.
        # We only rewind if there's more text left to process.
        if i < len(sentences) and overlap_sentences > 0:
            rewind = min(overlap_sentences, len(current_sentences))
            i = max(start_idx + 1, i - rewind)

    logger.info(
        "Chunking complete: %d chunks from %d sentences (avg %.0f chars/chunk)",
        len(chunks),
        len(sentences),
        sum(c["length"] for c in chunks) / max(len(chunks), 1),
    )
    return chunks