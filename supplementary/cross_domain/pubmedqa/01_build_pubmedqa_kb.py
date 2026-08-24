"""
Run (from this directory):
    python 01_build_pubmedqa_kb.py

Output (this directory only):
    kb_pubmedqa/chunks.json, faiss_index/, bm25_index.pkl
    pubmedqa_test_sample.json   — n=100 sampled test questions (see --n)
"""

import argparse
import json
import random
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[2]
sys.path.insert(0, str(ROOT_DIR))

# Import order: KB/embedding stack (sentence_transformers/faiss) is what
# we're about to use directly here; no google.genai import in this script
# at all, so the usual Windows segfault-on-import-order issue doesn't apply.
from utils.kb_builder import KnowledgeBaseBuilder  # noqa: E402

KB_DIR = THIS_DIR / "kb_pubmedqa"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="number of test questions to sample (default 100)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    from datasets import load_dataset
    print("Loading PubMedQA (PQA-L) from HuggingFace...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    print(f"Loaded {len(ds)} rows")

    # ── Build retrieval corpus from ALL rows' context paragraphs ─────────
    documents = []
    for row in ds:
        pubid = row["pubid"]
        contexts = row["context"]["contexts"]
        labels = row["context"]["labels"]
        for i, (ctx_text, label) in enumerate(zip(contexts, labels)):
            documents.append({
                "filename": f"pubmed_{pubid}_{label}",
                "category": "health",
                "doc_type": "abstract_section",
                "text": ctx_text,
            })
    print(f"Built {len(documents)} source documents (context paragraphs) for chunking")

    builder = KnowledgeBaseBuilder(kb_dir=KB_DIR)
    builder.build(documents)
    print(f"KB saved to: {KB_DIR}")

    # ── Sample n test questions (reproducible) ────────────────────────────
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), min(args.n, len(ds)))
    sample = []
    for i in indices:
        row = ds[i]
        sample.append({
            "query_id": f"pubmedqa_{row['pubid']}",
            "question": row["question"],
            "reference_answer": row["long_answer"],
            "final_decision": row["final_decision"],  # yes/no/maybe, for optional exact-match scoring
            "category": "health",
            "should_be_answerable": True,  # every PQA-L row has grounding context in the corpus by construction
        })

    with open(THIS_DIR / "pubmedqa_test_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sample)} test questions -> pubmedqa_test_sample.json")


if __name__ == "__main__":
    main()
