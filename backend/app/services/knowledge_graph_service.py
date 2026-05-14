"""
Knowledge Graph Service
-----------------------
Extracts named entities and grammatically meaningful subject-verb-object
triples from text using spaCy's dependency parser.

Previous approach: split every sentence into words and take words[0,1,2]
as subject/relation/object — extremely noisy ("The → is → a").

New approach:
  1. Walk every token that is a ROOT verb.
  2. Collect nominal subjects (nsubj / nsubjpass) from its children.
  3. Collect objects (dobj / attr / pobj) from its children (and prep phrases).
  4. Emit (subject, lemmatized_verb, object) triples.

Result: "Google released TensorFlow" → {"subject": "Google",
                                         "relation": "release",
                                         "object":  "TensorFlow"}
instead of the old: {"subject": "Google", "relation": "released", "object": "TensorFlow"} with
hundreds of junk triples like {"subject": "The", "relation": "is", "object": "a"}.
"""

import logging
import spacy

logger = logging.getLogger(__name__)

# Load spaCy model once at module level
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )

# In-memory graph storage (list of per-document dicts)
_knowledge_graph: list[dict] = []

# Minimum character length to keep an entity / relation node
_MIN_LEN = 3

# spaCy dependency labels that identify subjects and objects
_SUBJECT_DEPS = {"nsubj", "nsubjpass"}
_OBJECT_DEPS  = {"dobj", "attr", "pobj", "acomp"}

# Entity labels we care about (filter out noise like CARDINAL, ORDINAL …)
_KEEP_LABELS = {
    "ORG", "PERSON", "GPE", "LOC", "PRODUCT",
    "WORK_OF_ART", "EVENT", "FAC", "LANGUAGE", "LAW",
    "NORP",   # nationalities / political groups
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    return text.replace("\n", " ").replace("\t", " ").strip()


def _is_valid(text: str) -> bool:
    return len(text) >= _MIN_LEN and not text.isspace()


def _object_for_prep(token):
    """
    Follow preposition children of a verb to find prepositional objects.
    e.g. "works at Google" → token=at, returns "Google"
    """
    for child in token.children:
        if child.dep_ == "prep":
            for pobj in child.children:
                if pobj.dep_ == "pobj":
                    return pobj
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_entities_and_relationships(text: str) -> dict:
    """
    Returns:
        {
            "entities":      [{"text": str, "label": str}, …],
            "relationships": [{"subject": str, "relation": str, "object": str}, …],
        }
    """
    doc = nlp(text)

    # ── 1. Named Entities ────────────────────────────────────────────────────
    entities: list[dict] = []
    seen_entities: set[str] = set()

    for ent in doc.ents:
        if ent.label_ not in _KEEP_LABELS:
            continue
        cleaned = _clean(ent.text)
        if _is_valid(cleaned) and cleaned not in seen_entities:
            entities.append({"text": cleaned, "label": ent.label_})
            seen_entities.add(cleaned)

    # ── 2. Dependency-based SVO triples ─────────────────────────────────────
    relationships: list[dict] = []
    seen_triples: set[tuple] = set()

    for token in doc:
        # Only ROOT verbs drive meaningful triples
        if token.dep_ != "ROOT" or token.pos_ != "VERB":
            continue

        verb_lemma = token.lemma_.lower()
        if not _is_valid(verb_lemma):
            continue

        subjects = [c for c in token.children if c.dep_ in _SUBJECT_DEPS]
        objects  = [c for c in token.children if c.dep_ in _OBJECT_DEPS]

        # Also look for prepositional objects hanging off the verb
        prep_obj = _object_for_prep(token)
        if prep_obj:
            objects.append(prep_obj)

        for subj in subjects:
            for obj in objects:
                subj_text = _clean(subj.text)
                obj_text  = _clean(obj.text)

                if not _is_valid(subj_text) or not _is_valid(obj_text):
                    continue

                triple_key = (subj_text, verb_lemma, obj_text)
                if triple_key in seen_triples:
                    continue
                seen_triples.add(triple_key)

                relationships.append({
                    "subject":  subj_text,
                    "relation": verb_lemma,   # lemmatized → cleaner graph
                    "object":   obj_text,
                })

    graph_data = {"entities": entities, "relationships": relationships}

    logger.info(
        "Graph extracted: %d entities, %d relationships",
        len(entities),
        len(relationships),
    )
    return graph_data


# ──────────────────────────────────────────────────────────────────────────────
# Storage helpers
# ──────────────────────────────────────────────────────────────────────────────

def store_knowledge_graph(data: dict) -> None:
    _knowledge_graph.append(data)
    logger.debug("Knowledge graph updated (total segments: %d)", len(_knowledge_graph))


def get_knowledge_graph() -> list[dict]:
    return _knowledge_graph


def clear_knowledge_graph() -> None:
    """Remove all stored graph data (useful for testing or full reset)."""
    _knowledge_graph.clear()
    logger.info("Knowledge graph cleared")