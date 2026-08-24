"""
CUAD menguji generalisasi pada domain commercial-contract/legal menggunakan 510 kontrak nyata dari SEC/EDGAR dan 22.450 pertanyaan berlabel lawyer tentang 41 tipe klausul. Sekitar setengah pertanyaan bersifat not answerable, sehingga dataset ini cocok untuk menguji kemampuan sistem membedakan kapan harus menjawab vs. abstain, diukur dengan FRR (answerable) dan FAR (not-answerable).

Untuk retrieval, 510 kontrak dideduplikasi berdasarkan title dan dimasukkan sekali ke corpus. Sistem harus menemukan kontrak yang tepat dari ~500 distractors, lalu menemukan klausul yang relevan. Metadata kontrak tersedia, tetapi belum digunakan
Reuses utils.kb_builder.KnowledgeBaseBuilder AS-IS (no modification), just
pointed at a separate kb_dir — main repo's kb/ folder is never touched.

Run (from this directory):
    python 01_build_cuad_kb.py [--n 150]

Output (this directory only):
    kb_cuad/chunks.json, faiss_index/, bm25_index.pkl
    cuad_test_sample.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[2]
sys.path.insert(0, str(ROOT_DIR))

from utils.kb_builder import KnowledgeBaseBuilder  # noqa: E402

KB_DIR = THIS_DIR / "kb_cuad"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150,
                        help="number of test questions to sample (pool has 22,450; default 150 "
                             "to match the FinanceBench/ConditionalQA checks' scale)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    from datasets import load_dataset
    print("Loading CUAD (SQuAD-style clause QA) from HuggingFace...")
    # Legacy loading-script datasets require the auto-converted Parquet
    # branch under current `datasets` versions — see module docstring.
    ds = load_dataset("theatticusproject/cuad-qa", split="train", revision="refs/convert/parquet")
    print(f"Loaded {len(ds)} clause-presence QA rows")

    # ── Build retrieval corpus from unique contracts (dedup by title) ─────
    seen_titles = set()
    documents = []
    for row in ds:
        title = row["title"]
        if title in seen_titles:
            continue
        seen_titles.add(title)
        text = row.get("context", "")
        if not text.strip():
            continue
        documents.append({
            "filename": title,
            "category": "legal_contract",
            "doc_type": "sec_commercial_contract",
            "text": text,
        })
    print(f"Built {len(documents)} source documents (unique contracts) for chunking")

    builder = KnowledgeBaseBuilder(kb_dir=KB_DIR)
    builder.build(documents)
    print(f"KB saved to: {KB_DIR}")

    # ── Sample n test questions (reproducible) ────────────────────────────
    rng = random.Random(args.seed)
    n = min(args.n, len(ds))
    indices = rng.sample(range(len(ds)), n)
    sample = []
    for i in indices:
        row = ds[i]
        answer_texts = row["answers"]["text"]
        answerable = len(answer_texts) > 0
        sample.append({
            "query_id": row["id"],
            "question": f"{row['title']}: {row['question']}",
            "reference_answer": " | ".join(answer_texts) if answerable else
                                 "This clause type is not present in this contract.",
            "contract_title": row["title"],
            "category": "legal_contract",
            "should_be_answerable": answerable,
        })

    n_unanswerable = sum(1 for s in sample if not s["should_be_answerable"])
    print(f"Sample: {len(sample)} questions, {n_unanswerable} not_answerable "
          f"({n_unanswerable/len(sample)*100:.1f}%)")

    with open(THIS_DIR / "cuad_test_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sample)} test questions -> cuad_test_sample.json")


if __name__ == "__main__":
    main()
