"""
FinanceBench digunakan untuk menguji generalisasi pada domain financial-audit, sesuai Future Work ARDA-SR. Dataset publik ini berisi QA yang disusun analis keuangan berdasarkan SEC filings nyata, sehingga tidak bergantung pada corpus yang sama.

Seluruh `evidence_text` dari 150 baris dikumpulkan menjadi retrieval corpus. Sistem harus menemukan evidence yang tepat di antara cuplikan dari ~80+ perusahaan/filing. Knowledge base dibuat terpisah menggunakan `KnowledgeBaseBuilder` tanpa memodifikasi pipeline atau folder `kb/` utama.

Run (from this directory):
    python 01_build_financebench_kb.py [--n 150]

Output (this directory only):
    kb_financebench/chunks.json, faiss_index/, bm25_index.pkl
    financebench_test_sample.json
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

KB_DIR = THIS_DIR / "kb_financebench"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150,
                        help="number of test questions to sample (dataset has 150 total; "
                             "default uses all of them since it's already small)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    from datasets import load_dataset
    print("Loading FinanceBench from HuggingFace...")
    ds = load_dataset("PatronusAI/financebench", split="train")
    print(f"Loaded {len(ds)} rows")

    # ── Build retrieval corpus from ALL rows' evidence excerpts ───────────
    documents = []
    for row in ds:
        fbid = row["financebench_id"]
        for j, ev in enumerate(row["evidence"]):
            text = ev.get("evidence_text", "")
            if not text.strip():
                continue
            documents.append({
                "filename": f"{row['doc_name']}_{fbid}_{j}",
                "category": "financial",
                "doc_type": "sec_filing_excerpt",
                "text": text,
            })
    print(f"Built {len(documents)} source documents (evidence excerpts) for chunking")

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
        sample.append({
            "query_id": f"financebench_{row['financebench_id']}",
            "question": row["question"],
            "reference_answer": row["answer"],
            "company": row["company"],
            "doc_name": row["doc_name"],
            "category": "financial",
            "should_be_answerable": True,
        })

    with open(THIS_DIR / "financebench_test_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(sample)} test questions -> financebench_test_sample.json")


if __name__ == "__main__":
    main()
