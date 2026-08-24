import re
from typing import Dict, List, Optional


# ── CUAD: exact contract-identity filter ────────────────────────────────────

# Matches the "{contract_title}: " prefix that 01_build_cuad_kb.py prepends
# to every question (contract_title never itself contains ": ", verified —
# it's built from CUAD's own title field, an underscore/hyphen-delimited
# SEC-filing identifier such as "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR
# AGREEMENT").
_CUAD_TITLE_PREFIX_RE = re.compile(r"^(.*?): ")


def extract_cuad_metadata(query: str) -> Dict:
    """
    Returns {"filename": <contract_title>} if the query carries the
    "{title}: ..." prefix, else {} (safe no-op, same fallback behavior as
    the original arda_sr/retrieval.py function when nothing matches).
    """
    m = _CUAD_TITLE_PREFIX_RE.match(query)
    if not m:
        return {}
    return {"filename": m.group(1)}


# ── ConditionalQA: coarse topic-keyword bucket ──────────────────────────────

# Built by skimming kb_conditionalqa's 652 gov.uk page titles (see chat
# discussion / exploration). Deliberately small and approximate — same
# spirit as the original's 7-item Indonesian commodity list, not an
# exhaustive taxonomy. Order matters: first matching bucket wins, so more
# specific buckets are listed before broader ones.
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
    Shared helper used on BOTH sides (query-time extraction and one-time KB
    chunk enrichment) so the vocabulary can never drift out of sync between
    them. Returns the first matching bucket name, or None if nothing matches
    (safe no-op — HybridRetriever._matches() treats a missing/absent filter
    key as an automatic pass-through, so an unmatched query just falls back
    to plain unfiltered hybrid scoring, same as the original design).
    """
    t = text.lower()
    for bucket, keywords in keyword_map.items():
        if any(kw in t for kw in keywords):
            return bucket
    return None


def extract_conditionalqa_metadata(query: str) -> Dict:
    """
    Returns {"topic": <bucket>} if any keyword matches the query text
    (the "Scenario: ...\n\nQuestion: ..." string), else {}.
    """
    topic = bucket_topic(query)
    return {"topic": topic} if topic else {}
