import re
from typing import Dict, List, Optional


# ── CUAD: filter on exact contract identity ────────────────────────────────

# Matches the "{contract_title}: " prefix that 01_build_cuad_kb.py adds in
# front of every question (contract_title itself never contains ": ", which
# was verified — it comes from CUAD's own title field, an
# underscore/hyphen-delimited SEC-filing identifier like
# "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT").
_CUAD_TITLE_PREFIX_RE = re.compile(r"^(.*?): ")


def extract_cuad_metadata(query: str) -> Dict:
    """
    Yields {"filename": <contract_title>} when the query carries the
    "{title}: ..." prefix, otherwise {} (a safe no-op that falls back the
    way the original arda_sr/retrieval.py function does when nothing matches).
    """
    m = _CUAD_TITLE_PREFIX_RE.match(query)
    if not m:
        return {}
    return {"filename": m.group(1)}


# ── ConditionalQA: coarse keyword-based topic bucket ───────────────────────

# Compiled by skimming the 652 gov.uk page titles in kb_conditionalqa (see
# chat discussion / exploration). Intentionally small and approximate — the
# same spirit as the original's 7-item Indonesian commodity list rather than
# an exhaustive taxonomy. Order matters: the first bucket to match wins, so
# the more specific buckets appear ahead of the broader ones.
TOPIC_KEYWORD_MAP: Dict[str, List[str]] = {
    "immigration_visa": [
        "visa", "immigration", "asylum", "settle", "settlement", "indefinite leave",
        "travel document", "naturalisation", "citizenship", "passport",
    ],
    "driving_vehicle": [
        "driving", "motorcycle", "licence", "dvsa", "dvla", "mot ", "vehicle",
        "lorry", "bus driver", "goods vehicle", "speed limiter", "driving instructor",
    ],
    "family": [
        "adoption", "marriage", "divorce", "guardian", "child", "foster",
        "civil partnership", "parent", "maternity", "paternity",
    ],
    "employment": [
        "employ", "worker", "apprentice", "redundan", "dismissal",
        "employment appeal tribunal", "sick pay", "workplace",
    ],
    "benefits_pension": [
        "benefit", "pension", "allowance", "credit", "grant", "bursary",
        "maintenance", "universal credit", "disability",
    ],
    "tax": [
        "tax", "vat", "hmrc", "duty", "national insurance",
    ],
    "planning_property": [
        "planning", "hedgerow", "tree preservation", "listed building",
        "development", "home ownership", "housing",
    ],
    "legal_courts": [
        "appeal", "tribunal", "court", "magistrate", "bankrupt", "probate",
        "crime", "arrest", "charged with", "legal rights",
    ],
    "business": [
        "company", "business", "trade mark", "anti-competitive", "scam", "companies house",
    ],
    "education": [
        "exam", "learner loan", "school", "student",
    ],
}


def bucket_topic(text: str, keyword_map: Dict[str, List[str]] = TOPIC_KEYWORD_MAP) -> Optional[str]:
    """
    Shared helper invoked on BOTH sides (query-time extraction and one-off KB
    chunk enrichment), which keeps the vocabulary from ever drifting out of
    sync between them. Gives back the first bucket name that matches, or None
    when nothing does (safe no-op — HybridRetriever._matches() treats an
    absent/missing filter key as an automatic pass-through, so an unmatched
    query simply reverts to plain unfiltered hybrid scoring, as originally
    designed).
    """
    t = text.lower()
    for bucket, keywords in keyword_map.items():
        if any(kw in t for kw in keywords):
            return bucket
    return None


def extract_conditionalqa_metadata(query: str) -> Dict:
    """
    Yields {"topic": <bucket>} when some keyword matches the query text
    (the "Scenario: ...\n\nQuestion: ..." string), otherwise {}.
    """
    topic = bucket_topic(query)
    return {"topic": topic} if topic else {}
