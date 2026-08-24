"""
Run (from this directory):
    python 02_run_pubmedqa_test.py                      # default: standard_rag, arda_sr
    python 02_run_pubmedqa_test.py --methods arda_sr     # ARDA-SR only
    python 02_run_pubmedqa_test.py --smoke               # first 6 queries only

Output (this directory only):
    results/{method}_pubmedqa_results.json
    results/summary.csv / summary.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[2]
CROSS_DOMAIN_DIR = THIS_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(CROSS_DOMAIN_DIR))

# Import order: KB stack before google.genai (Windows segfault fix, same as
# every other AFTER-REVIEW script this session).
from utils.kb_builder import KnowledgeBase  # noqa: E402
from utils.llm_client import GeminiClient  # noqa: E402
from utils.openai_client import GPTJudgeClient  # noqa: E402
from baselines import ALL_BASELINES  # noqa: E402
from evaluation.metrics import compute_all_metrics  # noqa: E402
from generic_modules import GenericARDASRPipeline, GenericLLMJudge  # noqa: E402

KB_DIR = THIS_DIR / "kb_pubmedqa"
RESULTS_DIR = THIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# 2 baselines, deliberately spanning the range rather than picking an easy
# opponent: standard_rag = weakest baseline in the main experiment
# (FRR=0.175, Table 5), selfrag = strongest (FRR=0.115). Both verified
# domain-clean (no hardcoded "transmigration" text — AFTER-REVIEW/
# cross-domain/AUDIT.md), so no extra Generic-prompt work was needed.
DEFAULT_METHODS = ["standard_rag", "selfrag", "arda_sr"]
DOMAIN_CONTEXT = "biomedical and health research (PubMedQA: research questions over PubMed abstracts)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")] if args.methods else DEFAULT_METHODS

    with open(THIS_DIR / "pubmedqa_test_sample.json", encoding="utf-8") as f:
        qa_data = json.load(f)
    if args.smoke:
        qa_data = qa_data[:6]
    print(f"Loaded {len(qa_data)} PubMedQA test questions")

    client = GeminiClient()
    kb = KnowledgeBase(kb_dir=KB_DIR).load()
    # Judge uses the SAME Gemini client as generation (see JUDGE MODEL NOTE
    # in the module docstring) — an explicit, documented cost trade-off for
    # this cross-domain check, unlike the main experiment's independent judge.
    judge = GenericLLMJudge(GPTJudgeClient(), domain_context=DOMAIN_CONTEXT) if not args.skip_judge else None

    all_metrics = {}
    for method in methods:
        print(f"\n{'-'*40}\nRunning: {method}")
        out_path = RESULTS_DIR / f"{method}_pubmedqa_results.json"

        # Resume-friendly: if answers were already generated (e.g. an earlier
        # --skip-judge run), reuse them instead of regenerating — judging can
        # be added "later" without re-spending Gemini calls on generation.
        if out_path.exists() and not args.smoke:
            with open(out_path, encoding="utf-8") as f:
                results = json.load(f)
            print(f"  Loaded {len(results)} existing answers from {out_path.name} (not regenerating)")
        else:
            if method == "arda_sr":
                pipeline = GenericARDASRPipeline(kb, client, domain_context=DOMAIN_CONTEXT)
            else:
                # standard_rag (and the other baselines) carry no hardcoded
                # transmigration text — verified by grep, see AUDIT.md —
                # so the original baseline classes are used unmodified.
                pipeline = ALL_BASELINES[method](kb, client)

            results = []
            t0 = time.time()
            for qa in tqdm(qa_data, desc=method):
                res = pipeline.run(query=qa["question"], reference_answer=qa.get("reference_answer", ""))
                res["query_id"] = qa["query_id"]
                res["category"] = qa["category"]
                res["method"] = method
                res["should_be_answerable"] = True
                results.append(res)
            print(f"  Completed in {time.time()-t0:.1f}s")

        needs_judge = judge and any(r.get("rel") is None for r in results)
        if needs_judge:
            to_judge = [r for r in results if r.get("rel") is None]
            print(f"  LLM judging {len(to_judge)} un-judged answer(s) "
                  f"(skipping {len(results) - len(to_judge)} already-judged)...")
            new_scores = judge.judge_batch(to_judge, show_progress=True)
            for r in results:
                qid = r["query_id"]
                if qid in new_scores:
                    r.update({"rel": new_scores[qid]["rel"]/5.0,
                              "faith": new_scores[qid]["faith"]/5.0,
                              "cov": new_scores[qid]["cov"]/5.0})
        elif judge:
            print("  All answers already judged — skipping judge pass.")

        # CtxRel (Section 2.4.2: LLM-judged retrieval-quality score, 1-5) —
        # only meaningful for queries that actually retrieved evidence.
        needs_ctx_rel = judge and any(r.get("evidence") and r.get("ctx_rel") is None for r in results)
        if needs_ctx_rel:
            to_ctx_judge = [r for r in results if r.get("evidence") and r.get("ctx_rel") is None]
            print(f"  CtxRel judging {len(to_ctx_judge)} answer(s)...")
            ctx_scores = judge.judge_ctx_rel_batch(to_ctx_judge, show_progress=True)
            for r in results:
                qid = r["query_id"]
                if qid in ctx_scores:
                    r["ctx_rel"] = ctx_scores[qid]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Rebuild a {query_id: {rel,faith,cov}} dict (raw 1-5 scale) from
        # `results` for compute_all_metrics — covers BOTH answers judged in
        # this run and any judged in a previous run and loaded from disk.
        llm_scores = {
            r["query_id"]: {"rel": r["rel"] * 5, "faith": r["faith"] * 5, "cov": r["cov"] * 5}
            for r in results if r.get("rel") is not None
        } or None

        metrics = compute_all_metrics(results, llm_scores)
        all_metrics[method] = metrics
        print(f"  Rel={metrics.get('rel','--')} Faith={metrics.get('faith','--')} "
              f"FRR={metrics.get('frr','--')} Hit@5={metrics.get('hit_at_5','--')}")

    rows = [{"method": m, **{k: v for k, v in met.items() if not isinstance(v, dict)}}
            for m, met in all_metrics.items()]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved: {RESULTS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
