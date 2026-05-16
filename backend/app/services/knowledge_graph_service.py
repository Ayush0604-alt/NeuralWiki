"""
Knowledge Graph Service — True GraphRAG Edition
------------------------------------------------
Complete rewrite. Key differences from the hub-and-spoke version:

EXTRACTION (3 layers):
  Layer 1 — spaCy SVO:       Explicit verb-subject-object triples
  Layer 2 — Co-occurrence:   Entities in the same sentence get weighted
                              edges proportional to co-occurrence count
  Layer 3 — LLM extraction:  NVIDIA API call per text window for semantic
                              relationships spaCy misses entirely
                              (comparisons, causality, hierarchy, negations)

GRAPH STRUCTURE (no hub-and-spoke):
  • Orphan nodes are NOT force-connected to the center
  • Disconnected components are bridged via their highest-degree node
    (minimum spanning approach) — max 1 edge per component pair
  • Center node = highest-degree node AFTER all edges are built,
    not an artificially chosen hub

DEDUPLICATION:
  • Fuzzy entity merging (rapidfuzz if available, else basic normalisation)

CHUNK LINKING:
  • link_entities_to_chunks() tags every chunk with its contained entities
    enabling genuine entity-filtered retrieval in vector_service.py
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from itertools import combinations

import spacy

logger = logging.getLogger(__name__)

# ── Optional fuzzy dedup ───────────────────────────────────────────────────────
try:
    from rapidfuzz import fuzz
    _FUZZY = True
except ImportError:
    try:
        from thefuzz import fuzz
        _FUZZY = True
    except ImportError:
        _FUZZY = False
        logger.warning("rapidfuzz/thefuzz not installed — basic dedup only. pip install rapidfuzz")

# ── Optional LLM extraction ────────────────────────────────────────────────────
_NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")
_LLM_EXTRACTION_ENABLED = bool(_NVIDIA_KEY)
if not _LLM_EXTRACTION_ENABLED:
    logger.warning("NVIDIA_API_KEY not set — LLM relation extraction disabled")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

_knowledge_graph: list[dict] = []

# ── Cluster config ─────────────────────────────────────────────────────────────
CLUSTERS = {
    "ENTITY":   {"color": "#8b84ff", "icon": "◉", "label": "Entities"},
    "CONCEPT":  {"color": "#1fc791", "icon": "◈", "label": "Concepts"},
    "LOCATION": {"color": "#e05252", "icon": "◎", "label": "Locations"},
    "EVENT":    {"color": "#ff9f43", "icon": "◆", "label": "Events"},
    "DATE":     {"color": "#00d2d3", "icon": "◇", "label": "Dates"},
    "ACTION":   {"color": "#c47aff", "icon": "▶", "label": "Actions"},
    "QUANTITY": {"color": "#54a0ff", "icon": "▣", "label": "Quantities"},
    "TECH":     {"color": "#47bfff", "icon": "⬡", "label": "Technologies"},
    "MISC":     {"color": "#6b6b80", "icon": "·",  "label": "Other"},
}

_SPACY_TO_CLUSTER = {
    "PERSON": "ENTITY", "ORG": "ENTITY", "PRODUCT": "ENTITY",
    "WORK_OF_ART": "ENTITY", "NORP": "ENTITY",
    "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION",
    "EVENT": "EVENT",
    "LANGUAGE": "TECH", "LAW": "CONCEPT",
    "DATE": "DATE", "TIME": "DATE",
    "MONEY": "QUANTITY", "PERCENT": "QUANTITY",
    "CARDINAL": "QUANTITY", "ORDINAL": "QUANTITY", "QUANTITY": "QUANTITY",
}

_TECH_KW = {
    "python","java","javascript","typescript","c++","c#","ruby","golang","go",
    "rust","swift","kotlin","scala","php","html","css","sql","bash","shell",
    "react","angular","vue","svelte","django","flask","fastapi","spring",
    "express","tensorflow","pytorch","keras","sklearn","pandas","numpy","scipy",
    "spark","hadoop","kafka","docker","kubernetes","k8s","aws","azure","gcp",
    "git","github","gitlab","jenkins","terraform","nginx","linux","ubuntu",
    "postgresql","mysql","mongodb","redis","cassandra","sqlite","elasticsearch",
    "rest","graphql","grpc","http","websocket","oauth","jwt","api","sdk","cli",
    "machine learning","deep learning","neural network","nlp","computer vision",
    "transformer","bert","gpt","llm","embedding","vector","rag","blockchain",
    "microservice","serverless","iot","algorithm",
}
_CONCEPT_KW = {
    "strategy","management","leadership","innovation","marketing","finance",
    "theory","hypothesis","methodology","framework","model","analysis",
    "policy","regulation","compliance","governance","ethics","culture",
    "economics","statistics","psychology","philosophy","research",
}
_ACTION_KW = {
    "process","procedure","method","technique","implementation","deployment",
    "integration","optimization","testing","evaluation","assessment","training",
    "development","design","planning","execution","collaboration",
}


# ══════════════════════════════════════════════════════════════════════════════
# Classification helpers
# ══════════════════════════════════════════════════════════════════════════════

def _classify(text: str, spacy_label: str | None) -> str:
    s = text.strip()
    if not s or len(s) < 2:
        return "MISC"
    if re.fullmatch(r"[\d,.\s%$€£¥]+", s):
        return "QUANTITY"
    if spacy_label and spacy_label in _SPACY_TO_CLUSTER:
        base = _SPACY_TO_CLUSTER[spacy_label]
        if base == "ENTITY" and s.lower() in _TECH_KW:
            return "TECH"
        return base
    lo = s.lower()
    if any(k in lo for k in _TECH_KW):    return "TECH"
    if any(k in lo for k in _CONCEPT_KW): return "CONCEPT"
    if any(k in lo for k in _ACTION_KW):  return "ACTION"
    return "MISC"


def _detect_doc_type(text: str) -> str:
    lo = text[:3000].lower()
    signals = {
        "resume":    ["experience","education","skills","curriculum vitae","cv","work history"],
        "research":  ["abstract","methodology","conclusion","hypothesis","findings","literature review"],
        "news":      ["reported","according to","announced","journalist","breaking"],
        "legal":     ["whereas","hereinafter","plaintiff","defendant","pursuant","jurisdiction"],
        "technical": ["installation","configuration","api","endpoint","function","documentation"],
        "financial": ["revenue","profit","ebitda","fiscal","quarter","balance sheet"],
    }
    scores = {dt: sum(1 for kw in kws if kw in lo) for dt, kws in signals.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


# ══════════════════════════════════════════════════════════════════════════════
# Deduplication
# ══════════════════════════════════════════════════════════════════════════════

def _build_dedup_map(entities: list[dict], threshold: int = 82) -> dict[str, str]:
    """Build {variant → canonical} using fuzzy matching + substring rules."""
    if not entities:
        return {}

    freq: dict[str, int] = defaultdict(int)
    for e in entities:
        freq[e["text"].strip()] += 1

    texts = sorted(freq.keys(), key=len, reverse=True)
    dedup: dict[str, str] = {}
    merged: set[str] = set()

    for i, a in enumerate(texts):
        if a in merged:
            continue
        al = a.lower()
        for b in texts[i + 1:]:
            if b in merged:
                continue
            bl = b.lower()
            if al == bl:
                keep, drop = (a, b) if freq[a] >= freq[b] else (b, a)
                dedup[drop] = keep
                merged.add(drop)
                continue
            if bl in al:
                dedup[b] = a
                merged.add(b)
                continue
            if al in bl:
                dedup[a] = b
                merged.add(a)
                break
            if _FUZZY:
                if fuzz.token_sort_ratio(al, bl) >= threshold:
                    keep, drop = (a, b) if freq[a] >= freq[b] else (b, a)
                    dedup[drop] = keep
                    merged.add(drop)

    logger.info("Dedup: %d raw → %d canonical (%d merged)",
                len(texts), len(texts) - len(merged), len(merged))
    return dedup


def _apply_dedup(entities: list[dict], dm: dict[str, str]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for e in entities:
        c = dm.get(e["text"], e["text"])
        if c not in seen:
            seen.add(c)
            out.append({**e, "text": c})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — spaCy SVO extraction
# ══════════════════════════════════════════════════════════════════════════════

_VERB_MAP = {
    "develop":"developed","build":"built","create":"created","design":"designed",
    "implement":"implemented","use":"uses","apply":"applies","work":"worked at",
    "found":"founded","lead":"leads","manage":"manages","join":"joined",
    "publish":"published","write":"authored","research":"researched",
    "study":"studied","analyze":"analyzed","propose":"proposed",
    "introduce":"introduced","describe":"describes","include":"includes",
    "contain":"contains","support":"supports","enable":"enables",
    "require":"requires","provide":"provides","show":"shows",
    "improve":"improves","increase":"increases","reduce":"reduces",
    "affect":"affects","impact":"impacts","cause":"causes",
    "integrate":"integrates with","collaborate":"collaborates with",
    "acquire":"acquired","merge":"merged with","invest":"invested in",
    "release":"released","launch":"launched","deploy":"deployed",
    "compare":"compared to","outperform":"outperforms","replace":"replaces",
    "depend":"depends on","base":"based on","derive":"derived from",
}

_PAIR_MAP = {
    ("ENTITY","LOCATION"):"located in",   ("ENTITY","ENTITY"):"associated with",
    ("ENTITY","CONCEPT"):"related to",    ("ENTITY","TECH"):"uses",
    ("ENTITY","EVENT"):"participated in", ("CONCEPT","CONCEPT"):"related to",
    ("CONCEPT","TECH"):"implemented via", ("TECH","TECH"):"integrates with",
    ("ACTION","ENTITY"):"involves",       ("ACTION","CONCEPT"):"applies to",
}

_SUBJ = {"nsubj", "nsubjpass", "csubj"}
_OBJ  = {"dobj", "attr", "pobj", "acomp", "oprd", "xcomp", "dative"}


def _svo_extract(doc, known: set[str], dm: dict[str, str]) -> list[dict]:
    rels: list[dict] = []
    seen: set[tuple] = set()

    for token in doc:
        if token.pos_ != "VERB":
            continue
        subjs = [c for c in token.children if c.dep_ in _SUBJ]
        objs  = [c for c in token.children if c.dep_ in _OBJ]
        for child in token.children:
            if child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ in ("pobj", "pcomp"):
                        objs.append(pobj)

        for s in subjs:
            for o in objs:
                src = dm.get(s.text.strip(), s.text.strip())
                tgt = dm.get(o.text.strip(), o.text.strip())
                verb = token.lemma_.lower()
                if len(src) < 2 or len(tgt) < 2 or src == tgt:
                    continue
                if src not in known or tgt not in known:
                    continue
                if (src, tgt) in seen:
                    continue
                seen.add((src, tgt))
                label = _VERB_MAP.get(verb)
                if not label:
                    sc, tc = _classify(src, None), _classify(tgt, None)
                    label = _PAIR_MAP.get((sc, tc), verb or "related to")
                rels.append({"source": src, "target": tgt, "label": label, "weight": 2.0})

    logger.info("SVO relationships: %d", len(rels))
    return rels


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — Co-occurrence (sentence-level)
# ══════════════════════════════════════════════════════════════════════════════

def _cooccurrence_extract(
    doc,
    known: set[str],
    dm: dict[str, str],
    min_cooccur: int = 2,
) -> list[dict]:
    """
    Entities sharing the same sentence get a weighted co-occurrence edge.
    Weight = number of sentences they share. Only emit if weight >= min_cooccur.
    """
    cooccur: dict[tuple[str, str], int] = defaultdict(int)

    for sent in doc.sents:
        sent_text = sent.text.lower()
        present = [dm.get(e, e) for e in known if e.lower() in sent_text]
        present = list(dict.fromkeys(present))  # deduplicate order
        for a, b in combinations(sorted(present), 2):
            if a != b:
                key = (min(a, b), max(a, b))
                cooccur[key] += 1

    rels = []
    for (a, b), count in cooccur.items():
        if count >= min_cooccur:
            rels.append({
                "source": a, "target": b,
                "label": "co-occurs with",
                "weight": float(count),
            })

    logger.info("Co-occurrence: %d pairs (min_cooccur=%d)", len(rels), min_cooccur)
    return rels


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3 — LLM relation extraction
# ══════════════════════════════════════════════════════════════════════════════

_LLM_REL_PROMPT = """You are a knowledge graph extractor. Given text and a list of entities, find relationships between THOSE entities only.

Return ONLY a JSON array of objects with keys: source, target, label.
- source and target must be exact strings from the entity list
- label: short verb phrase (2-5 words), e.g. "is part of", "outperforms", "depends on", "is type of", "enables", "contradicts"
- NO self-loops, NO duplicate pairs, NO vague labels like "related to"
- Focus on: causality, hierarchy, comparison, dependency, composition, contrast

Entity list: {entities}

Text:
{text}

JSON array only (no markdown, no explanation):"""


def _llm_extract_window(text_window: str, entities: list[str]) -> list[dict]:
    if not _LLM_EXTRACTION_ENABLED or len(entities) < 2:
        return []
    import json as _json
    from openai import OpenAI
    client = OpenAI(api_key=_NVIDIA_KEY, base_url="https://integrate.api.nvidia.com/v1")
    prompt = _LLM_REL_PROMPT.format(
        entities=", ".join(entities[:40]),
        text=text_window[:2000],
    )
    try:
        resp = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = _json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception as exc:
        logger.warning("LLM extraction failed: %s", exc)
    return []


def _llm_extract_all(
    text: str,
    known: set[str],
    dm: dict[str, str],
    window_size: int = 1500,
) -> list[dict]:
    if not _LLM_EXTRACTION_ENABLED:
        return []

    # Split into sentence-aware windows
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    windows: list[str] = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < window_size:
            current += s + " "
        else:
            if current:
                windows.append(current.strip())
            current = s + " "
    if current:
        windows.append(current.strip())

    entity_list = sorted(known)
    all_rels: list[dict] = []
    seen_pairs: set[tuple] = set()

    for window in windows[:12]:  # cap at 12 to control API cost
        for r in _llm_extract_window(window, entity_list):
            src_raw = str(r.get("source", "")).strip()
            tgt_raw = str(r.get("target", "")).strip()
            label   = str(r.get("label", "")).strip()
            if not src_raw or not tgt_raw or not label:
                continue
            src = dm.get(src_raw, src_raw)
            tgt = dm.get(tgt_raw, tgt_raw)
            if src not in known or tgt not in known or src == tgt:
                continue
            key = (min(src, tgt), max(src, tgt), label)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            all_rels.append({"source": src, "target": tgt, "label": label, "weight": 3.0})

    logger.info("LLM extraction: %d relationships across %d windows",
                len(all_rels), len(windows))
    return all_rels


# ══════════════════════════════════════════════════════════════════════════════
# Minimal component bridging — replaces hub-and-spoke fallback
# ══════════════════════════════════════════════════════════════════════════════

def _connect_components(
    relationships: list[dict],
    all_nodes: set[str],
    entity_freq: dict[str, int],
) -> list[dict]:
    """
    Find disconnected components and add ONE bridge edge per component pair.
    Picks the highest-importance node from each side as the bridge endpoint.
    This keeps the graph navigable without creating a star topology.
    """
    # Build adjacency
    adj: dict[str, set[str]] = defaultdict(set)
    for r in relationships:
        adj[r["source"]].add(r["target"])
        adj[r["target"]].add(r["source"])
    for n in all_nodes:
        if n not in adj:
            adj[n] = set()

    # BFS component discovery
    visited: set[str] = set()
    components: list[list[str]] = []
    for node in all_nodes:
        if node in visited:
            continue
        comp: list[str] = []
        q = [node]
        while q:
            n = q.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.append(n)
            q.extend(adj[n] - visited)
        components.append(comp)

    if len(components) <= 1:
        logger.info("Graph fully connected: %d nodes, 1 component", len(all_nodes))
        return relationships

    logger.info("Graph has %d components — adding %d bridge edges",
                len(components), len(components) - 1)

    components.sort(key=len, reverse=True)
    extra = list(relationships)
    main_comp = set(components[0])

    for comp in components[1:]:
        # Score: degree + frequency
        main_best = max(main_comp,
                        key=lambda n: len(adj[n]) * 2 + entity_freq.get(n, 0))
        comp_best = max(comp,
                        key=lambda n: len(adj[n]) * 2 + entity_freq.get(n, 0))
        extra.append({
            "source": main_best,
            "target": comp_best,
            "label":  "related to",
            "weight": 0.5,
        })
        main_comp.update(comp)

    return extra


# ══════════════════════════════════════════════════════════════════════════════
# Keyword fallback
# ══════════════════════════════════════════════════════════════════════════════

def _keyword_fallback(text: str, existing: set[str], max_extra: int = 20) -> list[dict]:
    extras: list[dict] = []
    tl = text.lower()
    for kw_set, cluster in [(_TECH_KW, "TECH"), (_CONCEPT_KW, "CONCEPT"), (_ACTION_KW, "ACTION")]:
        for kw in kw_set:
            if kw in tl and kw.lower() not in existing:
                title = kw.title() if " " not in kw else kw.title()
                extras.append({"text": title, "label": None, "_cluster": cluster})
                existing.add(kw.lower())
            if len(extras) >= max_extra:
                return extras
    return extras


# ══════════════════════════════════════════════════════════════════════════════
# Main extraction entry point
# ══════════════════════════════════════════════════════════════════════════════

def extract_entities_and_relationships(text: str) -> dict:
    text_sample = text[:60_000]
    doc = nlp(text_sample)
    doc_type = _detect_doc_type(text_sample)
    logger.info("Doc type: %s", doc_type)

    # ── Collect raw entities ───────────────────────────────────────────────────
    raw_entities: list[dict] = []
    seen: set[str] = set()

    for ent in doc.ents:
        cleaned = ent.text.replace("\n", " ").strip()
        if len(cleaned) < 2 or re.fullmatch(r"[\d\s,.]+", cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        raw_entities.append({"text": cleaned, "label": ent.label_})

    for chunk in doc.noun_chunks:
        cleaned = chunk.text.replace("\n", " ").strip()
        if len(cleaned) < 3 or cleaned.lower() in seen:
            continue
        cl = _classify(cleaned, None)
        if cl in ("TECH", "CONCEPT", "ACTION"):
            seen.add(cleaned.lower())
            raw_entities.append({"text": cleaned, "label": None})

    if len(raw_entities) < 10:
        logger.info("Sparse NER (%d), running keyword fallback", len(raw_entities))
        for e in _keyword_fallback(text_sample, set(seen), 25):
            raw_entities.append(e)
            seen.add(e["text"].lower())

    # ── Deduplicate ────────────────────────────────────────────────────────────
    dm = _build_dedup_map(raw_entities)
    raw_entities = _apply_dedup(raw_entities, dm)

    # ── Frequency map ─────────────────────────────────────────────────────────
    text_lower = text_sample.lower()
    entity_freq: dict[str, int] = {
        e["text"]: text_lower.count(e["text"].lower())
        for e in raw_entities
    }
    known_set = {e["text"] for e in raw_entities}

    # ── Layer 1: SVO ───────────────────────────────────────────────────────────
    svo_rels = _svo_extract(doc, known_set, dm)

    # ── Layer 2: Co-occurrence ─────────────────────────────────────────────────
    cooc_rels = _cooccurrence_extract(doc, known_set, dm, min_cooccur=2)

    # ── Layer 3: LLM extraction ────────────────────────────────────────────────
    llm_rels = _llm_extract_all(text_sample, known_set, dm)

    # ── Merge (LLM > SVO > co-occurrence) ────────────────────────────────────
    merged_rels: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for r in llm_rels + svo_rels + cooc_rels:
        src, tgt = r["source"], r["target"]
        pair = (min(src, tgt), max(src, tgt))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            merged_rels.append(r)

    logger.info("Merged: %d total (LLM=%d SVO=%d cooc=%d)",
                len(merged_rels), len(llm_rels), len(svo_rels), len(cooc_rels))

    # ── Connect isolated components without hub-and-spoke ─────────────────────
    merged_rels = _connect_components(merged_rels, known_set, entity_freq)

    # ── Classify nodes into clusters ──────────────────────────────────────────
    all_clusters: dict[str, list[dict]] = {k: [] for k in CLUSTERS}
    node_to_cluster: dict[str, str] = {}

    for ent in raw_entities:
        eid = ent["text"]
        cluster = ent.get("_cluster") or _classify(eid, ent.get("label"))
        if cluster not in CLUSTERS:
            cluster = "MISC"
        if cluster == "QUANTITY" and ent.get("label") in ("CARDINAL", "ORDINAL"):
            continue
        if cluster == "DATE" and len(eid) < 3:
            continue
        if cluster == "MISC" and len(eid) < 4:
            continue
        existing = {n["id"].lower() for n in all_clusters[cluster]}
        if eid.lower() in existing:
            continue
        all_clusters[cluster].append({"id": eid, "cluster": cluster})
        node_to_cluster[eid] = cluster

    filled_clusters = {k: v for k, v in all_clusters.items() if v}

    # ── Pick center = highest degree + frequency node ─────────────────────────
    degree: dict[str, int] = defaultdict(int)
    for r in merged_rels:
        degree[r["source"]] += 1
        degree[r["target"]] += 1

    score = {n: degree.get(n, 0) * 2 + entity_freq.get(n, 0) for n in known_set}
    center_id = max(score, key=score.get) if score else (
        raw_entities[0]["text"] if raw_entities else "Document"
    )

    logger.info(
        "Graph [%s]: center='%s' (degree=%d), nodes=%d, edges=%d",
        doc_type, center_id, degree.get(center_id, 0),
        len(raw_entities), len(merged_rels),
    )

    return {
        "doc_type":      doc_type,
        "center":        {"id": center_id, "label": "ENTITY", "cluster": "CENTER"},
        "clusters":      filled_clusters,
        "relationships": merged_rels,
        "entities":      [{"text": e["text"], "label": e.get("label", "")} for e in raw_entities],
        "_dedup_map":    dm,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Chunk-to-entity linking
# ══════════════════════════════════════════════════════════════════════════════

def link_entities_to_chunks(
    chunks: list[dict],
    entities: list[dict],
    dedup_map: dict[str, str] | None = None,
) -> list[dict]:
    """Tag each chunk with canonical entity IDs it contains."""
    if dedup_map is None:
        dedup_map = {}
    canonical: list[str] = []
    seen: set[str] = set()
    for e in entities:
        c = dedup_map.get(e["text"], e["text"])
        if c not in seen:
            seen.add(c)
            canonical.append(c)

    for chunk in chunks:
        cl = chunk.get("content", "").lower()
        chunk["entities"] = [e for e in canonical if e.lower() in cl]

    total = sum(len(c.get("entities", [])) for c in chunks)
    logger.info("Chunk-entity links: %d chunks → %d total", len(chunks), total)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Storage
# ══════════════════════════════════════════════════════════════════════════════

def store_knowledge_graph(data: dict) -> None:
    _knowledge_graph.append(data)

def get_knowledge_graph() -> list[dict]:
    return _knowledge_graph

def clear_knowledge_graph() -> None:
    _knowledge_graph.clear()
    logger.info("Knowledge graph cleared")