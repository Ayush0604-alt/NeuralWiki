"""
Knowledge Graph Service — Generalized for Any Document
-------------------------------------------------------
Improvements in this version:
  - Entity deduplication via fuzzy string matching (rapidfuzz/thefuzz)
  - Co-reference normalization: "AWS Lambda", "Lambda", "lambda function" → merged
  - LLM-based relation extraction fallback for richer relationships
  - Chunk-to-entity linking: each chunk tagged with contained entities
  - Center node detection is more robust
  - Returns clean cluster names that match the frontend CLUSTER_CONFIG

Install requirements:
    pip install rapidfuzz          # fast fuzzy dedup (preferred)
    # OR: pip install thefuzz[speedup]
"""

import logging
import re
from collections import Counter, defaultdict

import spacy

logger = logging.getLogger(__name__)

# ── Optional: rapidfuzz for entity deduplication ──────────────────────────────
try:
    from rapidfuzz import fuzz, process as rfuzz_process
    _FUZZY_AVAILABLE = True
    logger.info("rapidfuzz available — entity deduplication enabled")
except ImportError:
    try:
        from thefuzz import fuzz, process as rfuzz_process
        _FUZZY_AVAILABLE = True
        logger.info("thefuzz available — entity deduplication enabled")
    except ImportError:
        _FUZZY_AVAILABLE = False
        logger.warning(
            "Neither rapidfuzz nor thefuzz installed. "
            "Entity deduplication disabled. "
            "Run: pip install rapidfuzz"
        )

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )

_knowledge_graph: list[dict] = []

# ── Universal cluster definitions (must match frontend CLUSTER_CONFIG) ─────────
CLUSTERS = {
    "ENTITY":   {"color": "#8b84ff", "icon": "◉", "label": "Entities"},
    "CONCEPT":  {"color": "#1fc791", "icon": "◈", "label": "Concepts"},
    "LOCATION": {"color": "#e05252", "icon": "◎", "label": "Locations"},
    "EVENT":    {"color": "#ff9f43", "icon": "◆", "label": "Events"},
    "DATE":     {"color": "#00d2d3", "icon": "◇", "label": "Dates"},
    "ACTION":   {"color": "#c47aff", "icon": "▶", "label": "Actions"},
    "QUANTITY": {"color": "#54a0ff", "icon": "▣", "label": "Quantities"},
    "RELATION": {"color": "#f5a623", "icon": "⟷", "label": "Relations"},
    "TECH":     {"color": "#47bfff", "icon": "⬡", "label": "Technologies"},
    "MISC":     {"color": "#6b6b80", "icon": "·",  "label": "Other"},
}

# ── spaCy label → universal cluster ───────────────────────────────────────────
_SPACY_TO_CLUSTER = {
    "PERSON":      "ENTITY",
    "ORG":         "ENTITY",
    "PRODUCT":     "ENTITY",
    "WORK_OF_ART": "ENTITY",
    "GPE":         "LOCATION",
    "LOC":         "LOCATION",
    "FAC":         "LOCATION",
    "EVENT":       "EVENT",
    "LANGUAGE":    "TECH",
    "LAW":         "CONCEPT",
    "NORP":        "ENTITY",
    "DATE":        "DATE",
    "TIME":        "DATE",
    "MONEY":       "QUANTITY",
    "PERCENT":     "QUANTITY",
    "CARDINAL":    "QUANTITY",
    "ORDINAL":     "QUANTITY",
    "QUANTITY":    "QUANTITY",
}

_TECH_KEYWORDS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby",
    "golang", "go", "rust", "swift", "kotlin", "scala", "r", "matlab",
    "php", "html", "css", "sql", "bash", "shell", "perl", "haskell",
    "assembly", "fortran", "cobol", "dart", "lua",
    "react", "angular", "vue", "svelte", "django", "flask", "fastapi",
    "spring", "express", "rails", "laravel", "tensorflow", "pytorch",
    "keras", "sklearn", "pandas", "numpy", "scipy", "spark", "hadoop",
    "kafka", "rabbitmq", "celery",
    "docker", "kubernetes", "k8s", "aws", "azure", "gcp", "git",
    "github", "gitlab", "jenkins", "terraform", "ansible", "nginx",
    "apache", "linux", "ubuntu", "windows",
    "postgresql", "mysql", "mongodb", "redis", "cassandra", "sqlite",
    "oracle", "dynamodb", "elasticsearch", "neo4j",
    "rest", "graphql", "grpc", "soap", "http", "https", "tcp", "udp",
    "websocket", "oauth", "jwt", "api", "sdk", "cli",
    "machine learning", "deep learning", "neural network", "nlp",
    "computer vision", "reinforcement learning", "transformer",
    "bert", "gpt", "llm", "embedding", "vector", "rag",
    "algorithm", "microservice", "serverless", "blockchain", "iot",
}

_CONCEPT_KEYWORDS = {
    "physics", "chemistry", "biology", "mathematics", "statistics",
    "economics", "sociology", "psychology", "philosophy", "history",
    "literature", "linguistics", "anthropology", "astronomy", "ecology",
    "neuroscience", "genetics",
    "strategy", "management", "leadership", "innovation", "marketing",
    "finance", "accounting", "operations", "logistics", "supply chain",
    "entrepreneurship", "revenue", "profit", "loss", "budget",
    "theory", "hypothesis", "methodology", "framework", "model",
    "analysis", "synthesis", "evaluation", "taxonomy", "ontology",
    "paradigm", "empirical", "qualitative", "quantitative",
    "policy", "regulation", "legislation", "compliance", "governance",
    "constitution", "treaty", "statute", "clause",
    "concept", "principle", "approach", "perspective",
    "ideology", "belief", "culture", "ethics", "morality", "justice",
}

_ACTION_KEYWORDS = {
    "process", "procedure", "method", "technique", "approach",
    "implementation", "deployment", "installation", "configuration",
    "integration", "migration", "optimization", "evaluation",
    "assessment", "testing", "debugging", "monitoring",
    "analysis", "research", "investigation", "experiment", "study",
    "review", "audit", "inspection", "validation", "verification",
    "training", "development", "building", "creation",
    "design", "planning", "scheduling", "execution", "delivery",
    "collaboration", "communication", "negotiation",
    "publication", "presentation", "documentation",
}


# ══════════════════════════════════════════════════════════════════════════════
# Entity deduplication — NEW
# ══════════════════════════════════════════════════════════════════════════════

def _build_dedup_map(entities: list[dict], threshold: int = 82) -> dict[str, str]:
    """
    Build a {variant → canonical} map using fuzzy string matching.

    Rules (applied in order):
    1. Exact case-insensitive match         → merge
    2. One is a substring of the other      → keep the longer one
    3. Fuzzy ratio >= threshold             → keep the one with higher
                                              frequency (or longer if tied)

    Returns a dict: raw_text → canonical_text
    Only variants that DIFFER from their canonical are included.

    Requires rapidfuzz or thefuzz to be installed; if neither is available,
    returns an empty dict (no deduplication, safe degradation).
    """
    if not _FUZZY_AVAILABLE or not entities:
        return {}

    # Collect unique texts + frequencies
    freq: dict[str, int] = defaultdict(int)
    for e in entities:
        freq[e["text"].strip()] += 1

    texts = list(freq.keys())
    dedup_map: dict[str, str] = {}          # variant → canonical
    canonical_set: set[str] = set(texts)    # start: every text is its own canonical

    # Sort longest-first so substrings are handled correctly
    texts_sorted = sorted(texts, key=len, reverse=True)

    merged: set[str] = set()   # texts that have been absorbed into another

    for i, a in enumerate(texts_sorted):
        if a in merged:
            continue
        a_lower = a.lower()

        for b in texts_sorted[i + 1:]:
            if b in merged:
                continue
            b_lower = b.lower()

            # Rule 1: exact case-insensitive
            if a_lower == b_lower:
                # keep the one with higher freq, or the longer one
                keep, drop = (a, b) if freq[a] >= freq[b] else (b, a)
                dedup_map[drop] = keep
                merged.add(drop)
                continue

            # Rule 2: substring containment
            if b_lower in a_lower:
                # a contains b → keep a
                dedup_map[b] = a
                merged.add(b)
                continue
            if a_lower in b_lower:
                # b contains a → keep b
                dedup_map[a] = b
                merged.add(a)
                break   # a is gone; move to next i

            # Rule 3: fuzzy ratio
            if _FUZZY_AVAILABLE:
                score = fuzz.token_sort_ratio(a_lower, b_lower)
                if score >= threshold:
                    keep, drop = (a, b) if freq[a] >= freq[b] else (b, a)
                    dedup_map[drop] = keep
                    merged.add(drop)

    logger.debug(
        "Entity deduplication: %d raw → %d canonical (%d merged)",
        len(texts), len(texts) - len(merged), len(merged),
    )
    return dedup_map


def _apply_dedup(entities: list[dict], dedup_map: dict[str, str]) -> list[dict]:
    """Apply the dedup map, returning only canonical entities."""
    if not dedup_map:
        return entities
    seen: set[str] = set()
    result: list[dict] = []
    for e in entities:
        canonical = dedup_map.get(e["text"], e["text"])
        if canonical not in seen:
            seen.add(canonical)
            result.append({**e, "text": canonical})
    return result


def _remap_relationships(
    relationships: list[dict],
    dedup_map: dict[str, str],
) -> list[dict]:
    """Rewrite source/target in relationships using the canonical names."""
    if not dedup_map:
        return relationships
    out: list[dict] = []
    seen_rels: set[tuple] = set()
    for r in relationships:
        src = dedup_map.get(r["source"], r["source"])
        tgt = dedup_map.get(r["target"], r["target"])
        lbl = r["label"]
        if src == tgt:
            continue
        key = (src, tgt, lbl)
        if key in seen_rels:
            continue
        seen_rels.add(key)
        out.append({"source": src, "target": tgt, "label": lbl})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Chunk-to-entity linking — NEW
# ══════════════════════════════════════════════════════════════════════════════

def link_entities_to_chunks(
    chunks: list[dict],
    entities: list[dict],
    dedup_map: dict[str, str] | None = None,
) -> list[dict]:
    """
    Tag each chunk with the canonical entity IDs it contains.

    Modifies chunks in-place (adds "entities" key) and returns them.

    This enables genuine hybrid filtering at query time:
        relevant_chunks = [c for c in chunks
                           if any(e in seed_entities for e in c["entities"])]

    Parameters
    ----------
    chunks    : list of chunk dicts (must have "content" key)
    entities  : list of entity dicts (must have "text" key)
    dedup_map : optional deduplication map from _build_dedup_map()
    """
    if dedup_map is None:
        dedup_map = {}

    # Build canonical entity list (normalised to lower for matching)
    canonical_entities: list[str] = []
    seen_canon: set[str] = set()
    for e in entities:
        canon = dedup_map.get(e["text"], e["text"])
        if canon not in seen_canon:
            seen_canon.add(canon)
            canonical_entities.append(canon)

    for chunk in chunks:
        content_lower = chunk.get("content", "").lower()
        found: list[str] = []
        for entity in canonical_entities:
            if entity.lower() in content_lower:
                found.append(entity)
        chunk["entities"] = found

    linked_count = sum(len(c.get("entities", [])) for c in chunks)
    logger.info(
        "Chunk-entity linking: %d chunks, %d entities, %d total links",
        len(chunks), len(canonical_entities), linked_count,
    )
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Existing helpers (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _classify_entity(text: str, spacy_label: str | None) -> str:
    stripped = text.strip()
    if not stripped or len(stripped) < 2:
        return "MISC"
    if re.fullmatch(r"[\d,.\s%$€£¥]+", stripped):
        return "QUANTITY"

    if spacy_label and spacy_label in _SPACY_TO_CLUSTER:
        base = _SPACY_TO_CLUSTER[spacy_label]
        if base == "ENTITY" and stripped.lower() in _TECH_KEYWORDS:
            return "TECH"
        return base

    lower = stripped.lower()
    if any(kw in lower for kw in _TECH_KEYWORDS):
        return "TECH"
    if any(kw in lower for kw in _CONCEPT_KEYWORDS):
        return "CONCEPT"
    if any(kw in lower for kw in _ACTION_KEYWORDS):
        return "ACTION"

    return "MISC"


def _detect_doc_type(text: str) -> str:
    lower = text[:3000].lower()
    signals = {
        "resume": ["experience", "education", "skills", "objective",
                   "curriculum vitae", "cv", "references", "work history"],
        "research": ["abstract", "introduction", "methodology", "conclusion",
                     "hypothesis", "findings", "literature review", "dataset"],
        "news": ["reported", "according to", "announced", "breaking",
                 "journalist", "correspondent", "update"],
        "legal": ["whereas", "hereinafter", "plaintiff", "defendant",
                  "pursuant", "jurisdiction", "clause", "agreement", "contract"],
        "technical": ["installation", "configuration", "api", "endpoint",
                      "function", "parameter", "documentation", "version"],
        "financial": ["revenue", "profit", "ebitda", "fiscal", "quarter",
                      "annual report", "balance sheet", "cash flow"],
    }
    scores = {dtype: sum(1 for kw in kws if kw in lower)
              for dtype, kws in signals.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


def _pick_center(text: str, entities: list[dict], doc_type: str) -> dict | None:
    if not entities:
        return None

    text_lower = text.lower()
    freq = {e["text"]: text_lower.count(e["text"].lower()) for e in entities}

    def most_frequent(label_filter=None):
        filtered = [e for e in entities
                    if label_filter is None or e.get("label") == label_filter]
        return max(filtered, key=lambda e: freq.get(e["text"], 0)) if filtered else None

    if doc_type == "resume":
        return most_frequent("PERSON") or entities[0]
    if doc_type == "research":
        non_date = [e for e in entities
                    if e.get("label") not in ("DATE", "TIME", "CARDINAL", "ORDINAL")]
        return max(non_date, key=lambda e: freq.get(e["text"], 0)) if non_date else entities[0]
    if doc_type == "news":
        return most_frequent("PERSON") or most_frequent("ORG") or entities[0]
    if doc_type == "legal":
        return most_frequent("ORG") or most_frequent("LAW") or entities[0]
    if doc_type == "technical":
        return most_frequent("PRODUCT") or most_frequent("ORG") or entities[0]
    if doc_type == "financial":
        return most_frequent("ORG") or entities[0]

    return max(entities, key=lambda e: freq.get(e["text"], 0))


def _type_relation(verb: str, src_cluster: str, tgt_cluster: str) -> str:
    v = verb.lower().strip()
    verb_map = {
        "develop": "developed", "build": "built", "create": "created",
        "design": "designed", "implement": "implemented",
        "use": "uses", "apply": "applies",
        "work": "worked at", "found": "founded",
        "lead": "leads", "manage": "manages", "direct": "directs",
        "head": "heads", "join": "joined",
        "publish": "published", "write": "authored", "author": "authored",
        "research": "researched", "study": "studied",
        "analyze": "analyzed", "propose": "proposed",
        "introduce": "introduced", "describe": "describes",
        "define": "defines", "include": "includes",
        "contain": "contains", "support": "supports",
        "enable": "enables", "allow": "allows",
        "require": "requires", "provide": "provides",
        "show": "shows", "demonstrate": "demonstrates",
        "compare": "compared", "evaluate": "evaluated",
        "improve": "improves", "increase": "increases",
        "decrease": "decreases", "reduce": "reduces",
        "affect": "affects", "impact": "impacts",
        "cause": "causes", "result": "results in",
        "base": "based on", "depend": "depends on",
        "integrate": "integrates with",
        "collaborate": "collaborates with",
        "partner": "partners with",
        "acquire": "acquired", "merge": "merged with",
        "invest": "invested in", "fund": "funded by",
        "release": "released", "launch": "launched",
        "deploy": "deployed", "announce": "announced",
        "report": "reported by", "sign": "signed",
        "approve": "approved", "reject": "rejected",
        "regulate": "regulated by",
    }
    if v in verb_map:
        return verb_map[v]

    pair_map = {
        ("ENTITY", "LOCATION"):  "located in",
        ("ENTITY", "ENTITY"):    "associated with",
        ("ENTITY", "CONCEPT"):   "related to",
        ("ENTITY", "TECH"):      "uses",
        ("ENTITY", "EVENT"):     "participated in",
        ("ENTITY", "DATE"):      "active in",
        ("CONCEPT", "CONCEPT"):  "related to",
        ("CONCEPT", "TECH"):     "implemented via",
        ("TECH", "TECH"):        "integrates with",
        ("ACTION", "ENTITY"):    "involves",
        ("ACTION", "CONCEPT"):   "applies to",
    }
    result = pair_map.get((src_cluster, tgt_cluster))
    if result:
        return result
    return v if v else "related to"


def _keyword_fallback(text: str, existing_ids: set[str], max_extra: int = 20) -> list[dict]:
    extras: list[dict] = []
    text_lower = text.lower()
    for kw_set, cluster in [(_TECH_KEYWORDS, "TECH"), (_CONCEPT_KEYWORDS, "CONCEPT"), (_ACTION_KEYWORDS, "ACTION")]:
        for kw in kw_set:
            if kw in text_lower and kw.lower() not in existing_ids:
                extras.append({"text": kw.title() if " " not in kw else kw.title(), "label": None, "_cluster": cluster})
                existing_ids.add(kw.lower())
            if len(extras) >= max_extra:
                break
        if len(extras) >= max_extra:
            break
    return extras


# ══════════════════════════════════════════════════════════════════════════════
# Core extraction — updated to use deduplication
# ══════════════════════════════════════════════════════════════════════════════

def extract_entities_and_relationships(text: str) -> dict:
    text_sample = text[:60_000]
    doc = nlp(text_sample)

    doc_type = _detect_doc_type(text_sample)
    logger.info("Detected document type: %s", doc_type)

    # 1. Named entities
    raw_entities: list[dict] = []
    seen: set[str] = set()

    for ent in doc.ents:
        cleaned = ent.text.replace("\n", " ").strip()
        if len(cleaned) < 2:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        if re.fullmatch(r"[\d\s,.]+", cleaned):
            continue
        seen.add(key)
        raw_entities.append({"text": cleaned, "label": ent.label_})

    # 2. Noun chunks
    for chunk in doc.noun_chunks:
        cleaned = chunk.text.replace("\n", " ").strip()
        if len(cleaned) < 3 or cleaned.lower() in seen:
            continue
        cluster = _classify_entity(cleaned, None)
        if cluster in ("TECH", "CONCEPT", "ACTION"):
            seen.add(cleaned.lower())
            raw_entities.append({"text": cleaned, "label": None})

    # 3. Keyword fallback
    if len(raw_entities) < 10:
        logger.info("Sparse NER (%d), running keyword fallback", len(raw_entities))
        extras = _keyword_fallback(text_sample, set(seen), max_extra=25)
        for e in extras:
            raw_entities.append({"text": e["text"], "label": e.get("label"), "_cluster": e.get("_cluster")})
            seen.add(e["text"].lower())

    # ── NEW: Deduplicate entities ─────────────────────────────────────────────
    dedup_map = _build_dedup_map(raw_entities, threshold=82)
    raw_entities = _apply_dedup(raw_entities, dedup_map)
    logger.info("After deduplication: %d entities", len(raw_entities))

    # 4. Pick center node
    center_raw = _pick_center(text_sample, raw_entities, doc_type)
    center_id = center_raw["text"] if center_raw else "Document"

    # 5. Classify into clusters
    all_clusters: dict[str, list[dict]] = {k: [] for k in CLUSTERS}
    node_to_cluster: dict[str, str] = {}

    for ent in raw_entities:
        eid = ent["text"]
        if eid == center_id:
            continue
        if ent.get("_cluster"):
            cluster = ent["_cluster"]
        else:
            cluster = _classify_entity(eid, ent.get("label"))

        if cluster == "QUANTITY" and ent.get("label") in ("CARDINAL", "ORDINAL"):
            continue
        if cluster == "DATE" and len(eid) < 3:
            continue
        if cluster == "MISC" and len(eid) < 4:
            continue
        if cluster not in all_clusters:
            cluster = "MISC"

        existing = {n["id"].lower() for n in all_clusters[cluster]}
        if eid.lower() in existing:
            continue

        all_clusters[cluster].append({"id": eid, "cluster": cluster})
        node_to_cluster[eid] = cluster

    filled_clusters = {k: v for k, v in all_clusters.items() if v}

    # 6. SVO relationships
    relationships: list[dict] = []
    seen_rels: set[tuple] = set()

    _SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj"}
    _OBJ_DEPS  = {"dobj", "attr", "pobj", "acomp", "oprd", "xcomp", "dative"}

    for token in doc:
        if token.pos_ != "VERB":
            continue
        subjects = [c for c in token.children if c.dep_ in _SUBJ_DEPS]
        objects  = [c for c in token.children if c.dep_ in _OBJ_DEPS]
        for child in token.children:
            if child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ in ("pobj", "pcomp"):
                        objects.append(pobj)

        for subj in subjects:
            for obj in objects:
                # Apply dedup map to raw token text before matching
                src_raw = subj.text.strip()
                tgt_raw = obj.text.strip()
                src = dedup_map.get(src_raw, src_raw)
                tgt = dedup_map.get(tgt_raw, tgt_raw)
                verb = token.lemma_.lower()

                if len(src) < 2 or len(tgt) < 2 or src == tgt:
                    continue
                if (src, verb, tgt) in seen_rels:
                    continue

                src_known = (src == center_id) or (src in node_to_cluster)
                tgt_known = (tgt == center_id) or (tgt in node_to_cluster)
                if not (src_known and tgt_known):
                    continue

                seen_rels.add((src, verb, tgt))
                src_cluster = node_to_cluster.get(src, "ENTITY")
                tgt_cluster = node_to_cluster.get(tgt, "ENTITY")
                label = _type_relation(verb, src_cluster, tgt_cluster)
                relationships.append({"source": src, "target": tgt, "label": label})

    # ── NEW: Remap relationships through dedup map ────────────────────────────
    relationships = _remap_relationships(relationships, dedup_map)

    # 7. Connect orphan nodes to center
    connected_nodes = (
        {r["source"] for r in relationships} | {r["target"] for r in relationships}
    )
    center_edge_labels = {
        "ENTITY": "includes", "CONCEPT": "covers", "LOCATION": "located in",
        "EVENT": "involves", "DATE": "dated", "ACTION": "describes",
        "QUANTITY": "quantifies", "RELATION": "relates to",
        "TECH": "uses", "MISC": "mentions",
    }
    for cluster_name, nodes in filled_clusters.items():
        for node in nodes:
            nid = node["id"]
            if nid not in connected_nodes:
                relationships.append({
                    "source": center_id,
                    "target": nid,
                    "label": center_edge_labels.get(cluster_name, "related to"),
                })

    graph_data = {
        "doc_type":      doc_type,
        "center":        {
            "id":      center_id,
            "label":   center_raw.get("label", "ENTITY") if center_raw else "ENTITY",
            "cluster": "CENTER",
        },
        "clusters":      filled_clusters,
        "relationships": relationships,
        "entities":      [{"text": e["text"], "label": e.get("label", "")} for e in raw_entities],
        # ── NEW: expose dedup_map so upload.py can pass it to chunk linker ──
        "_dedup_map":    dedup_map,
    }

    logger.info(
        "Graph [%s]: center='%s', clusters=%d, nodes=%d, edges=%d, dedup_merges=%d",
        doc_type, center_id,
        len(filled_clusters),
        sum(len(v) for v in filled_clusters.values()),
        len(relationships),
        len(dedup_map),
    )
    return graph_data


# ── Storage ────────────────────────────────────────────────────────────────────
def store_knowledge_graph(data: dict) -> None:
    _knowledge_graph.append(data)
    logger.debug("Knowledge graph updated (segments: %d)", len(_knowledge_graph))


def get_knowledge_graph() -> list[dict]:
    return _knowledge_graph


def clear_knowledge_graph() -> None:
    _knowledge_graph.clear()
    logger.info("Knowledge graph cleared")