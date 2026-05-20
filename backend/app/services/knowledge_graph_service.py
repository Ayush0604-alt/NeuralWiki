"""
Knowledge Graph Service — LLM-first entity extraction
------------------------------------------------------
Replaces the previous spaCy-NER-primary approach with an LLM-first pipeline.

EXTRACTION ORDER (revised):
  Layer 0 — LLM entity extraction (PRIMARY):
      The NVIDIA API extracts domain entities the model understands natively —
      "CRISPR-Cas9", "Basel III", "load balancing", "transformer architecture".
      This runs first and its entities are added to known_set immediately so
      that spaCy co-occurrence can use them.
  Layer 1 — spaCy SVO (secondary):
      SVO triples are built from spaCy dep-parse — fast, no API cost.
      Kept because spaCy captures verb relationships reliably for common text.
  Layer 2 — Co-occurrence (cheap):
      Sentence-level co-occurrence using ALL known entities (both LLM + spaCy).
  Layer 3 — LLM relationship extraction (semantic):
      A second LLM pass finds relationships between the now-complete entity set.

This replaces the original order where spaCy NER ran first and LLM was a
last-resort supplement. The result is dramatically higher recall on
technical, legal, medical, and financial documents.

Everything else (dedup, component bridging, chunk linking, cluster
classification, center detection) is unchanged.
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
from collections import defaultdict
from itertools import combinations

from dotenv import load_dotenv
import spacy

logger = logging.getLogger(__name__)

load_dotenv()

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
        logger.warning("rapidfuzz/thefuzz not installed — basic dedup only.")

# ── LLM config ────────────────────────────────────────────────────────────────
_EXTRACT_API_KEY = os.getenv("EXTRACT_API_KEY") or os.getenv("MISTRAL_API_KEY")
_EXTRACT_API_BASE_URL = os.getenv("EXTRACT_API_BASE_URL") or "https://api.mistral.ai/v1"
_EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "mistral-small-latest")

# Fallback NVIDIA for extraction
_EXTRACT_API_KEY_FALLBACK = os.getenv("EXTRACT_API_KEY_FALLBACK")
_EXTRACT_API_BASE_URL_FALLBACK = os.getenv("EXTRACT_API_BASE_URL_FALLBACK") or "https://integrate.api.nvidia.com/v1"
_EXTRACT_MODEL_FALLBACK = os.getenv("EXTRACT_MODEL_FALLBACK", "nvidia/llama-3.1-nemotron-nano-8b-v1")

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_LLM_EXTRACTION_ENABLED = bool(_EXTRACT_API_KEY or _EXTRACT_API_KEY_FALLBACK)
if not _LLM_EXTRACTION_ENABLED:
    logger.warning("EXTRACT_API_KEY and fallback not configured — LLM entity/relation extraction disabled")
else:
    logger.info("LLM extraction enabled with Mistral primary and NVIDIA fallback")


def _extract_client():
    from openai import OpenAI
    return OpenAI(api_key=_EXTRACT_API_KEY, base_url=_EXTRACT_API_BASE_URL, max_retries=0)


def _extract_client_fallback():
    """Fallback NVIDIA client for extraction."""
    from openai import OpenAI
    return OpenAI(api_key=_EXTRACT_API_KEY_FALLBACK, base_url=_EXTRACT_API_BASE_URL_FALLBACK, max_retries=0)


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "429" in message
        or "rate limit" in message
        or "capacity exceeded" in message
        or "service_tier_capacity_exceeded" in message
    )


def _load_json_array_forgiving(raw: str) -> list | None:
    """Parse a JSON array from model output that may include fences or extra text."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    decoder = _json.JSONDecoder()
    for start_char in ("[", "{"):
        start = text.find(start_char)
        if start == -1:
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    try:
        parsed = _json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, list) else None


def _nvidia_entity_fallback(text: str) -> list[dict]:
    """NVIDIA fallback for entity extraction."""
    if not _EXTRACT_API_KEY_FALLBACK:
        return []
    try:
        client = _extract_client_fallback()
        resp = client.chat.completions.create(
            model=_EXTRACT_MODEL_FALLBACK,
            messages=[{"role": "user", "content": _LLM_ENTITY_PROMPT.format(text=text[:2500])}],
            temperature=0.0,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content or ""
        parsed = _load_json_array_forgiving(raw)
        if not isinstance(parsed, list):
            return []
        out: list[dict] = []
        seen: set[str] = set()
        for item in parsed:
            text_val = str(item.get("text", "")).strip()
            type_val = str(item.get("type", "OTHER")).upper()
            if len(text_val) < 2 or text_val.lower() in seen:
                continue
            seen.add(text_val.lower())
            cluster = _LLM_TYPE_TO_CLUSTER.get(type_val, "MISC")
            out.append({"text": text_val, "label": type_val, "_cluster": cluster, "_source": "nvidia_fallback"})
        logger.info("NVIDIA entity fallback: %d entities extracted", len(out))
        return out
    except Exception as exc:
        logger.warning("NVIDIA entity fallback failed: %s", exc)
        return []


def _gemini_entity_fallback(text: str) -> list[dict]:
    if not _GEMINI_API_KEY:
        return []
    try:
        from app.services.llm.gemini_provider import generate_gemini_response

        raw = generate_gemini_response("Extract entities", _LLM_ENTITY_PROMPT.format(text=text[:2500]), model=_GEMINI_MODEL)
        parsed = _load_json_array_forgiving(raw)
        if not isinstance(parsed, list):
            return []
        out: list[dict] = []
        seen: set[str] = set()
        for item in parsed:
            text_val = str(item.get("text", "")).strip()
            type_val = str(item.get("type", "OTHER")).upper()
            if len(text_val) < 2 or text_val.lower() in seen:
                continue
            seen.add(text_val.lower())
            out.append({"text": text_val, "label": type_val, "_cluster": _LLM_TYPE_TO_CLUSTER.get(type_val, "MISC"), "_source": "gemini"})
        return out
    except Exception as exc:
        logger.warning("Gemini entity fallback failed: %s", exc)
        return []


def _nvidia_relationship_fallback(text: str, known: set[str], dm: dict[str, str]) -> list[dict]:
    """NVIDIA fallback for relationship extraction."""
    if not _EXTRACT_API_KEY_FALLBACK or len(known) < 2:
        return []
    try:
        client = _extract_client_fallback()
        prompt = _LLM_REL_PROMPT.format(entities=", ".join(sorted(known)[:40]), text=text[:2000])
        resp = client.chat.completions.create(
            model=_EXTRACT_MODEL_FALLBACK,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=800,
        )
        raw = resp.choices[0].message.content or ""
        parsed = _load_json_array_forgiving(raw)
        if not isinstance(parsed, list):
            return []
        out: list[dict] = []
        seen_pairs: set[tuple[str, str, str]] = set()
        for r in parsed:
            src_raw = str(r.get("source", "")).strip()
            tgt_raw = str(r.get("target", "")).strip()
            label = str(r.get("label", "")).strip()
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
            out.append({"source": src, "target": tgt, "label": label, "weight": 3.0})
        logger.info("NVIDIA relationship fallback: %d relationships extracted", len(out))
        return out
    except Exception as exc:
        logger.warning("NVIDIA relationship fallback failed: %s", exc)
        return []


def _gemini_relationship_fallback(text: str, known: set[str], dm: dict[str, str]) -> list[dict]:
    if not _GEMINI_API_KEY or len(known) < 2:
        return []
    try:
        from app.services.llm.gemini_provider import generate_gemini_response

        raw = generate_gemini_response("Extract relationships", _LLM_REL_PROMPT.format(entities=", ".join(sorted(known)[:40]), text=text[:2000]), model=_GEMINI_MODEL)
        parsed = _load_json_array_forgiving(raw)
        if not isinstance(parsed, list):
            return []
        out: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for r in parsed:
            src_raw = str(r.get("source", "")).strip()
            tgt_raw = str(r.get("target", "")).strip()
            label = str(r.get("label", "")).strip()
            if not src_raw or not tgt_raw or not label:
                continue
            src = dm.get(src_raw, src_raw)
            tgt = dm.get(tgt_raw, tgt_raw)
            if src not in known or tgt not in known or src == tgt:
                continue
            key = (min(src, tgt), max(src, tgt), label)
            if key in seen:
                continue
            seen.add(key)
            out.append({"source": src, "target": tgt, "label": label, "weight": 3.0})
        return out
    except Exception as exc:
        logger.warning("Gemini relationship fallback failed: %s", exc)
        return []

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

# ── LLM entity extraction prompt ──────────────────────────────────────────────

_LLM_ENTITY_PROMPT = """You are a knowledge graph entity extractor.
Extract ALL named entities and key concepts from the text below.

Include: people, organizations, products, technologies, protocols, 
scientific terms, legal terms, financial instruments, medical terms, 
concepts, frameworks, methodologies, locations, events, dates.

Be SPECIFIC: prefer "CRISPR-Cas9" over "gene editing", "Basel III" 
over "regulation", "load balancing" over "technique".

Return ONLY a JSON array of objects: [{{"text": "...", "type": "..."}}]
Types: PERSON, ORG, PRODUCT, TECH, CONCEPT, LOCATION, EVENT, DATE, 
       QUANTITY, LAW, MEDICAL, FINANCIAL, OTHER

No markdown, no explanation, just the JSON array.

Text:
{text}"""

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


# ══════════════════════════════════════════════════════════════════════════════
# Layer 0 — LLM entity extraction (PRIMARY)
# ══════════════════════════════════════════════════════════════════════════════

_LLM_TYPE_TO_CLUSTER = {
    "PERSON": "ENTITY", "ORG": "ENTITY", "PRODUCT": "ENTITY",
    "TECH": "TECH", "CONCEPT": "CONCEPT", "LOCATION": "LOCATION",
    "EVENT": "EVENT", "DATE": "DATE", "QUANTITY": "QUANTITY",
    "LAW": "CONCEPT", "MEDICAL": "ENTITY", "FINANCIAL": "CONCEPT",
    "OTHER": "MISC",
}


def _llm_extract_entities(text: str, window_size: int = 2500) -> list[dict]:
    """
    Run the LLM over windows of the document to extract entities.
    Returns list of {text, label, _cluster} dicts.
    """
    if not _LLM_EXTRACTION_ENABLED:
        return []

    client = _extract_client() if _EXTRACT_API_KEY else None

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

    all_entities: list[dict] = []
    seen: set[str] = set()
    hit_rate_limit = False

    for window in windows[:20]:  # cap at 20 windows (~50k chars)
        prompt = _LLM_ENTITY_PROMPT.format(text=window[:2500])
        try:
            if client is not None:
                resp = client.chat.completions.create(
                    model=_EXTRACT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=600,
                )
                raw = resp.choices[0].message.content or ""
                parsed = _load_json_array_forgiving(raw)
            else:
                parsed = []
            if not isinstance(parsed, list):
                continue
            for item in parsed:
                text_val = str(item.get("text", "")).strip()
                type_val = str(item.get("type", "OTHER")).upper()
                if len(text_val) < 2:
                    continue
                key = text_val.lower()
                if key in seen:
                    continue
                seen.add(key)
                cluster = _LLM_TYPE_TO_CLUSTER.get(type_val, "MISC")
                all_entities.append({
                    "text":     text_val,
                    "label":    type_val,
                    "_cluster": cluster,
                    "_source":  "llm",
                })
        except Exception as exc:
            logger.warning("LLM entity extraction failed for window: %s", exc)
            if _is_rate_limit_error(exc):
                hit_rate_limit = True
                break

    if hit_rate_limit or not all_entities:
        # Fallback 1: Try NVIDIA
        all_entities = _nvidia_entity_fallback(text)
        # Fallback 2: Try Gemini if NVIDIA also fails
        if not all_entities:
            all_entities = _gemini_entity_fallback(text)

    logger.info("LLM entity extraction: %d entities across %d windows", len(all_entities), len(windows))
    return all_entities


# ══════════════════════════════════════════════════════════════════════════════
# Classification helpers (unchanged)
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
# Deduplication (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _build_dedup_map(entities: list[dict], threshold: int = 82) -> dict[str, str]:
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
                dedup[drop] = keep; merged.add(drop); continue
            if bl in al:
                dedup[b] = a; merged.add(b); continue
            if al in bl:
                dedup[a] = b; merged.add(a); break
            if _FUZZY and fuzz.token_sort_ratio(al, bl) >= threshold:
                keep, drop = (a, b) if freq[a] >= freq[b] else (b, a)
                dedup[drop] = keep; merged.add(drop)
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
# Layer 1 — spaCy SVO (secondary; uses enriched known_set from LLM)
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
# Layer 2 — Co-occurrence (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _cooccurrence_extract(
    doc,
    known: set[str],
    dm: dict[str, str],
    min_cooccur: int = 2,
) -> list[dict]:
    cooccur: dict[tuple[str, str], int] = defaultdict(int)
    for sent in doc.sents:
        sent_text = sent.text.lower()
        present = [dm.get(e, e) for e in known if e.lower() in sent_text]
        present = list(dict.fromkeys(present))
        for a, b in combinations(sorted(present), 2):
            if a != b:
                key = (min(a, b), max(a, b))
                cooccur[key] += 1
    rels = []
    for (a, b), count in cooccur.items():
        if count >= min_cooccur:
            rels.append({"source": a, "target": b, "label": "co-occurs with", "weight": float(count)})
    logger.info("Co-occurrence: %d pairs (min_cooccur=%d)", len(rels), min_cooccur)
    return rels


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3 — LLM relationship extraction (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _llm_extract_relationships(
    text: str,
    known: set[str],
    dm: dict[str, str],
    window_size: int = 1500,
) -> list[dict]:
    if not _LLM_EXTRACTION_ENABLED or len(known) < 2:
        return []

    client = _extract_client() if _EXTRACT_API_KEY else None

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
    hit_rate_limit = False

    for window in windows[:12]:
        prompt = _LLM_REL_PROMPT.format(entities=", ".join(entity_list[:40]), text=window[:2000])
        try:
            if client is not None:
                resp = client.chat.completions.create(
                    model=_EXTRACT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=800,
                )
                raw = resp.choices[0].message.content or ""
                parsed = _load_json_array_forgiving(raw)
            else:
                parsed = []
            if not isinstance(parsed, list):
                continue
            for r in parsed:
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
        except Exception as exc:
            logger.warning("LLM relationship extraction failed: %s", exc)
            if _is_rate_limit_error(exc):
                hit_rate_limit = True
                break

    if hit_rate_limit or not all_rels:
        # Fallback 1: Try NVIDIA
        all_rels = _nvidia_relationship_fallback(text, known, dm)
        # Fallback 2: Try Gemini if NVIDIA also fails
        if not all_rels:
            all_rels = _gemini_relationship_fallback(text, known, dm)

    logger.info("LLM relationship extraction: %d relationships across %d windows",
                len(all_rels), len(windows))
    return all_rels


# ══════════════════════════════════════════════════════════════════════════════
# Component bridging (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _connect_components(
    relationships: list[dict],
    all_nodes: set[str],
    entity_freq: dict[str, int],
) -> list[dict]:
    adj: dict[str, set[str]] = defaultdict(set)
    for r in relationships:
        adj[r["source"]].add(r["target"])
        adj[r["target"]].add(r["source"])
    for n in all_nodes:
        if n not in adj:
            adj[n] = set()
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
            visited.add(n); comp.append(n)
            q.extend(adj[n] - visited)
        components.append(comp)
    if len(components) <= 1:
        return relationships
    components.sort(key=len, reverse=True)
    extra = list(relationships)
    main_comp = set(components[0])
    for comp in components[1:]:
        main_best = max(main_comp, key=lambda n: len(adj[n]) * 2 + entity_freq.get(n, 0))
        comp_best = max(comp,      key=lambda n: len(adj[n]) * 2 + entity_freq.get(n, 0))
        extra.append({"source": main_best, "target": comp_best, "label": "related to", "weight": 0.5})
        main_comp.update(comp)
    return extra


# ══════════════════════════════════════════════════════════════════════════════
# Keyword fallback (unchanged)
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
# Main extraction entry point — REVISED PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def extract_entities_and_relationships(text: str, use_llm: bool = True) -> dict:
    """
    Revised extraction pipeline:

    1. LLM entity extraction (Layer 0) — runs FIRST for domain coverage when enabled
    2. spaCy NER supplements LLM entities (adds news-text entities cheaply)
    3. spaCy noun-chunk supplement for TECH/CONCEPT/ACTION keywords
    4. Keyword fallback if entity count still low
    5. Deduplication across all sources
    6. SVO extraction using FULL entity set (Layer 1)
    7. Co-occurrence using FULL entity set (Layer 2)
    8. LLM relationship extraction (Layer 3)
    9. Component bridging, cluster assignment, center detection
    """
    text_sample = text[:60_000]
    doc_type = _detect_doc_type(text_sample)
    logger.info("Doc type: %s", doc_type)

    # ── Layer 0: LLM entity extraction (PRIMARY) ───────────────────────────────
    llm_entities = _llm_extract_entities(text_sample) if use_llm else []
    seen: set[str] = {e["text"].lower() for e in llm_entities}
    raw_entities: list[dict] = list(llm_entities)

    # ── spaCy NER (supplement — adds named entities LLM may have missed) ──────
    doc = nlp(text_sample)

    for ent in doc.ents:
        cleaned = ent.text.replace("\n", " ").strip()
        if len(cleaned) < 2 or re.fullmatch(r"[\d\s,.]+", cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        raw_entities.append({"text": cleaned, "label": ent.label_, "_source": "spacy"})

    # spaCy noun chunks for TECH/CONCEPT/ACTION
    for chunk in doc.noun_chunks:
        cleaned = chunk.text.replace("\n", " ").strip()
        if len(cleaned) < 3 or cleaned.lower() in seen:
            continue
        cl = _classify(cleaned, None)
        if cl in ("TECH", "CONCEPT", "ACTION"):
            seen.add(cleaned.lower())
            raw_entities.append({"text": cleaned, "label": None, "_cluster": cl, "_source": "spacy_noun"})

    # Keyword fallback if still sparse
    if len(raw_entities) < 10:
        logger.info("Sparse entity set (%d), running keyword fallback", len(raw_entities))
        for e in _keyword_fallback(text_sample, set(seen), 25):
            raw_entities.append(e)
            seen.add(e["text"].lower())

    # ── Deduplication ──────────────────────────────────────────────────────────
    dm = _build_dedup_map(raw_entities)
    raw_entities = _apply_dedup(raw_entities, dm)

    # ── Frequency map ─────────────────────────────────────────────────────────
    text_lower = text_sample.lower()
    entity_freq: dict[str, int] = {
        e["text"]: text_lower.count(e["text"].lower()) for e in raw_entities
    }
    known_set = {e["text"] for e in raw_entities}

    # ── Layer 1: SVO (now benefits from LLM entities in known_set) ────────────
    svo_rels = _svo_extract(doc, known_set, dm)

    # ── Layer 2: Co-occurrence ─────────────────────────────────────────────────
    cooc_rels = _cooccurrence_extract(doc, known_set, dm, min_cooccur=2)

    # ── Layer 3: LLM relationship extraction ──────────────────────────────────
    llm_rels = _llm_extract_relationships(text_sample, known_set, dm) if use_llm else []

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

    # ── Component bridging ─────────────────────────────────────────────────────
    merged_rels = _connect_components(merged_rels, known_set, entity_freq)

    # ── Cluster assignment ─────────────────────────────────────────────────────
    all_clusters: dict[str, list[dict]] = {k: [] for k in CLUSTERS}
    node_to_cluster: dict[str, str] = {}

    for ent in raw_entities:
        eid = ent["text"]
        # Prefer cluster from LLM type, fall back to classify()
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

    # ── Center = highest degree + frequency node ───────────────────────────────
    degree: dict[str, int] = defaultdict(int)
    for r in merged_rels:
        degree[r["source"]] += 1
        degree[r["target"]] += 1

    score = {n: degree.get(n, 0) * 2 + entity_freq.get(n, 0) for n in known_set}
    center_id = max(score, key=score.get) if score else (
        raw_entities[0]["text"] if raw_entities else "Document"
    )

    # Count by source for logging
    llm_count   = sum(1 for e in raw_entities if e.get("_source") == "llm")
    spacy_count = sum(1 for e in raw_entities if e.get("_source", "").startswith("spacy"))

    logger.info(
        "Graph [%s]: center='%s' (degree=%d), nodes=%d (LLM=%d spaCy=%d), edges=%d",
        doc_type, center_id, degree.get(center_id, 0),
        len(raw_entities), llm_count, spacy_count, len(merged_rels),
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
# Chunk-to-entity linking (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def link_entities_to_chunks(
    chunks: list[dict],
    entities: list[dict],
    dedup_map: dict[str, str] | None = None,
) -> list[dict]:
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
# Storage (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def store_knowledge_graph(data: dict) -> None:
    _knowledge_graph.append(data)

def get_knowledge_graph() -> list[dict]:
    return _knowledge_graph

def clear_knowledge_graph() -> None:
    _knowledge_graph.clear()
    logger.info("Knowledge graph cleared")