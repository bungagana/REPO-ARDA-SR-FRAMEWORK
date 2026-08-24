"""
Cross-domain generalization check (UK public-policy domain): builds a
standalone knowledge base from ConditionalQA (Sun et al., ACL 2022 —
https://github.com/haitian-sun/ConditionalQA) — a PUBLIC, already-complete
corpus+QA benchmark: questions+scenarios+answers are written by human
annotators over real gov.uk policy pages, NOT generated from the same
corpus, so this check is not exposed to the reviewer's circularity concern
(same rationale as 01_build_financebench_kb.py / 01_build_pubmedqa_kb.py).

WHY THIS DATASET (see chat discussion): ConditionalQA is the closest public
structural analogue to ARDA-SR's original domain (Indonesian transmigration
government policy documents) among the candidates considered — real
government policy pages, per-question "scenario" describing the user's
situation that the correct answer is CONDITIONAL on, a natural
not_answerable label (~4% of rows), and multi-answer items where several
conditions each yield a different valid answer. This is meant to give
AQR's m4 routing / SR's scenario-comparison logic and DDA's refusal
handling more genuine signal than PubMedQA/FinanceBench did, where those
mechanisms were rarely exercised (see AFTER-REVIEW/cross-domain/AUDIT.md
and the FinanceBench/PubMedQA results discussion).

Run (from this directory):
    python 01_build_conditionalqa_kb.py [--n 150]

Output (this directory only):
    raw_conditionalqa/*.json          — cached GitHub downloads
    kb_conditionalqa/chunks.json, faiss_index/, bm25_index.pkl
    conditionalqa_test_sample.json
"""

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[2]
sys.path.insert(0, str(ROOT_DIR))

from utils.kb_builder import KnowledgeBaseBuilder  # noqa: E402

KB_DIR = THIS_DIR / "kb_conditionalqa"
RAW_DIR = THIS_DIR / "raw_conditionalqa"

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0"
FILES = ["documents.json", "train.json", "dev.json"]

_TAG_RE = re.compile(r"<[^>]+>")


def _download_if_missing() -> None:
    RAW_DIR.mkdir(exist_ok=True)
    for fname in FILES:
        dest = RAW_DIR / fname
        if dest.exists():
            continue
        url = f"{GITHUB_RAW_BASE}/{fname}"
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)


def _strip_html(html_fragments: list) -> str:
    return "\n".join(_TAG_RE.sub("", frag).strip() for frag in html_fragments if frag.strip())


def _format_answer(answers: list) -> str:
    """ConditionalQA answers: list of [answer_text, [supporting_condition_refs]].
    Multiple entries = multiple conditionally-valid answers; join them so the
    reference captures every condition-dependent answer, not just the first."""
    if not answers:
        return ""
    texts = [a[0] for a in answers if a and a[0]]
    return " | ".join(texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150,
                        help="number of test questions to sample (pool has ~2,623; default 150 "
                             "to match the FinanceBench check's scale)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    _download_if_missing()

    with open(RAW_DIR / "documents.json", encoding="utf-8") as f:
        docs_raw = json.load(f)
    with open(RAW_DIR / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(RAW_DIR / "dev.json", encoding="utf-8") as f:
        dev_raw = json.load(f)

    pool = train_raw + dev_raw
    print(f"Loaded {len(docs_raw)} policy documents, {len(train_raw)} train + "
          f"{len(dev_raw)} dev questions ({len(pool)} pooled)")

    # ── Build retrieval corpus from ALL policy documents ──────────────────
    documents = []
    for doc in docs_raw:
        text = _strip_html(doc.get("contents", []))
        if not text.strip():
            continue
        documents.append({
            "filename": doc.get("title", doc.get("url", "doc")),
            "category": "public_policy",
            "doc_type": "gov_uk_policy_page",
            "text": text,
        })
    print(f"Built {len(documents)} source documents (policy pages) for chunking")

    builder = KnowledgeBaseBuilder(kb_dir=KB_DIR)
    builder.build(documents)
    print(f"KB saved to: {KB_DIR}")

    # ── Sample n test questions (reproducible) ────────────────────────────
    rng = random.Random(args.seed)
    n = min(args.n, len(pool))
    indices = rng.sample(range(len(pool)), n)
    sample = []
    for i in indices:
        row = pool[i]
        not_answerable = bool(row.get("not_answerable", False))
        sample.append({
            "query_id": row.get("id", f"conditionalqa_{i}"),
            "question": f"Scenario: {row.get('scenario', '').strip()}\n\nQuestion: {row['question']}",
            "reference_answer": _format_answer(row.get("answers", [])),
            "source_url": row.get("url", ""),
            "not_answerable": not_answerable,
            "category": "public_policy",
            "should_be_answerable": not not_answerable,
        })

    n_unanswerable = sum(1 for s in sample if s["not_answerable"])
    print(f"Sample: {len(sample)} questions, {n_unanswerable} not_answerable "
          f"({n_unanswerable/len(sample)*100:.1f}%)")

    with open(THIS_DIR / "conditionalqa_test_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sample)} test questions -> conditionalqa_test_sample.json")


if __name__ == "__main__":
    main()
