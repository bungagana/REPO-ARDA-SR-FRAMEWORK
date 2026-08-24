"""
LLM-as-Judge evaluation following the RAGAS methodology.
Scores answer quality on 3 dimensions: Relevance, Faithfulness, Coverage (1–5 Likert).
"""

import json
import logging
import time
from typing import Dict, List, Optional

from utils.llm_client import GeminiClient
from config import JUDGE_SCALE_MAX, REQUEST_DELAY_S

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are an independent evaluator assessing the quality of QA system responses
for an Indonesian transmigration government services domain.

Evaluate the CANDIDATE ANSWER on three dimensions (each scored 1–5):
1. Relevance (Rel): How well does the answer address the question intent?
   1=completely off-topic, 5=directly and precisely answers the question
2. Faithfulness (Faith): Is the answer factually grounded in the RETRIEVED EVIDENCE
   below, without hallucination? Judge grounding against the evidence, not just
   plausibility. If no evidence is provided (parametric/no-retrieval answer), score
   based on whether claims are appropriately hedged/uncertain rather than asserted
   as fact without support — do not reward confident unsupported claims.
   1=major factual errors or unsupported claims, 5=fully supported by the evidence
3. Coverage (Cov): Does the answer cover all important aspects?
   1=very incomplete, 5=comprehensive and thorough

Question: {query}
Reference Answer: {reference}
Retrieved Evidence (use this to judge Faithfulness; may be empty for no-retrieval answers):
{evidence}

Candidate Answer: {answer}

Return ONLY this JSON (no other text):
{{
  "rel": <1-5>,
  "faith": <1-5>,
  "cov": <1-5>,
  "rel_reason": "<one sentence>",
  "faith_reason": "<one sentence, must reference the evidence if any was provided>",
  "cov_reason": "<one sentence>"
}}"""


CTX_REL_PROMPT = """\
You are an independent evaluator assessing RETRIEVAL quality (not answer quality)
for an Indonesian transmigration government services QA system.

Score how ADEQUATELY the retrieved evidence below supports answering the question,
on a scale of 1-5:
1 = evidence is irrelevant or off-topic to the question
2 = evidence touches the general topic but lacks the specific information needed
3 = evidence is partially relevant; some needed information is present, some missing
4 = evidence is mostly relevant and sufficient, with minor gaps
5 = evidence is highly relevant and fully sufficient to answer the question

Question: {query}

Retrieved Evidence:
{evidence}

Return ONLY this JSON (no other text):
{{
  "ctx_rel": <1-5>,
  "reason": "<one sentence>"
}}"""


COMBINED_JUDGE_PROMPT = """\
You are an independent evaluator assessing the quality of QA system responses
for an Indonesian transmigration government services domain.

Evaluate the CANDIDATE ANSWER on three dimensions (each scored 1-5):
1. Relevance (Rel): How well does the answer address the question intent?
   1=completely off-topic, 5=directly and precisely answers the question
2. Faithfulness (Faith): Is the answer factually grounded in the RETRIEVED EVIDENCE
   below, without hallucination? Judge grounding against the evidence, not just
   plausibility. If no evidence is provided (parametric/no-retrieval answer), score
   based on whether claims are appropriately hedged/uncertain rather than asserted
   as fact without support -- do not reward confident unsupported claims.
   1=major factual errors or unsupported claims, 5=fully supported by the evidence
3. Coverage (Cov): Does the answer cover all important aspects?
   1=very incomplete, 5=comprehensive and thorough

Also separately evaluate the RETRIEVED EVIDENCE ITSELF (not the answer) on a fourth
dimension -- this is independent of how good the answer is:
4. Context Relevance (CtxRel): How ADEQUATELY does the retrieved evidence below
   support answering the question, on its own merits?
   1 = evidence is irrelevant or off-topic to the question
   2 = evidence touches the general topic but lacks the specific information needed
   3 = evidence is partially relevant; some needed information is present, some missing
   4 = evidence is mostly relevant and sufficient, with minor gaps
   5 = evidence is highly relevant and fully sufficient to answer the question

Question: {query}
Reference Answer: {reference}
Retrieved Evidence (use this to judge Faithfulness AND CtxRel; may be empty for no-retrieval answers):
{evidence}

Candidate Answer: {answer}

Return ONLY this JSON (no other text):
{{
  "rel": <1-5>,
  "faith": <1-5>,
  "cov": <1-5>,
  "ctx_rel": <1-5>,
  "rel_reason": "<one sentence>",
  "faith_reason": "<one sentence, must reference the evidence if any was provided>",
  "cov_reason": "<one sentence>",
  "ctx_rel_reason": "<one sentence>"
}}"""


class LLMJudge:
    """
    LLM-as-judge for automated evaluation.
    Scores are normalised to [0,1] (raw / 5) when used in metrics.
    """

    def __init__(self, client: GeminiClient | None = None):
        self.client = client or GeminiClient()

    def judge_batch(
        self,
        results: List[Dict],
        show_progress: bool = True,
    ) -> Dict[str, Dict]:
        """
        Score a list of result dicts.
        Returns: {query_id: {rel, faith, cov, rel_reason, faith_reason, cov_reason}}
        """
        from tqdm import tqdm
        scores = {}
        iterator = tqdm(results, desc="LLM judging") if show_progress else results

        for r in iterator:
            qid = r.get("query_id", r.get("query", "")[:50])
            score = self.judge_single(
                query=r.get("query", ""),
                answer=r.get("answer", ""),
                reference=r.get("reference", ""),
                evidence=r.get("evidence", []),
            )
            scores[qid] = score

        return scores

    def judge_single(
        self,
        query: str,
        answer: str,
        reference: str = "",
        evidence: Optional[List[Dict]] = None,
    ) -> Dict:
        """Score a single (query, answer, reference, evidence) tuple.

        `evidence` is the list of retrieved chunk dicts actually used by the
        pipeline for this query (same shape as elsewhere: {filename, text, ...}).
        Passing it lets Faithfulness be judged against the real evidence, per
        the paper's Faith(a) definition (Section 2.4.2) — a bare (query,
        reference, answer) judge cannot assess grounding at all.
        """
        if not answer.strip():
            return {"rel": 1, "faith": 1, "cov": 1,
                    "rel_reason": "Empty answer", "faith_reason": "Empty", "cov_reason": "Empty"}
        evidence_text = self._format_evidence(evidence or [])
        prompt = JUDGE_PROMPT.format(
            query=query,
            reference=reference or "(no reference provided)",
            evidence=evidence_text or "(no evidence retrieved — no-retrieval/parametric answer)",
            answer=answer[:1500],
        )
        # No try/except: an API failure here must abort this query's judging
        # (retryable on resume), not silently substitute a fabricated
        # rel=faith=cov=2 fallback that judge_resumable() would then treat
        # as a real, complete judgment forever. Same rationale as
        # arda_sr/pipeline.py's _simple_generate() fix (2026-08-15
        # spending-cap incident).
        raw = self.client.generate_json(prompt)
        return {
            "rel":          max(1, min(5, int(raw.get("rel", 3)))),
            "faith":        max(1, min(5, int(raw.get("faith", 3)))),
            "cov":          max(1, min(5, int(raw.get("cov", 3)))),
            "rel_reason":   str(raw.get("rel_reason", "")),
            "faith_reason": str(raw.get("faith_reason", "")),
            "cov_reason":   str(raw.get("cov_reason", "")),
        }

    def judge_combined_single(
        self,
        query: str,
        answer: str,
        reference: str = "",
        evidence: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Single-call variant of judge_single() that also returns ctx_rel in the
        same LLM call, for queries that have evidence -- cuts judging from 2
        API calls/query down to 1 for the evidence-bearing subset. Added
        2026-08-15 to speed up the not-yet-started V3 rerun; NOT used to
        retroactively touch V1/V2's already-collected (2-call) judged data,
        which remains valid as-is (compute_ctx_rel() only reads the final
        `ctx_rel` field value, not how many calls produced it). Falls back to
        the plain judge_single() when there's no evidence, since ctx_rel
        isn't applicable there anyway.
        """
        if not evidence:
            return self.judge_single(query, answer, reference, evidence)
        if not answer.strip():
            return {"rel": 1, "faith": 1, "cov": 1, "ctx_rel": 1,
                    "rel_reason": "Empty answer", "faith_reason": "Empty",
                    "cov_reason": "Empty", "ctx_rel_reason": "Empty answer"}
        evidence_text = self._format_evidence(evidence)
        prompt = COMBINED_JUDGE_PROMPT.format(
            query=query,
            reference=reference or "(no reference provided)",
            evidence=evidence_text or "(no evidence retrieved -- no-retrieval/parametric answer)",
            answer=answer[:1500],
        )
        # No try/except -- same rationale as judge_single()/judge_ctx_rel_single().
        raw = self.client.generate_json(prompt)
        return {
            "rel":            max(1, min(5, int(raw.get("rel", 3)))),
            "faith":          max(1, min(5, int(raw.get("faith", 3)))),
            "cov":            max(1, min(5, int(raw.get("cov", 3)))),
            "ctx_rel":        max(1, min(5, int(raw.get("ctx_rel", 3)))),
            "rel_reason":     str(raw.get("rel_reason", "")),
            "faith_reason":   str(raw.get("faith_reason", "")),
            "cov_reason":     str(raw.get("cov_reason", "")),
            "ctx_rel_reason": str(raw.get("ctx_rel_reason", "")),
        }

    def judge_combined_batch(
        self,
        results: List[Dict],
        show_progress: bool = True,
    ) -> Dict[str, Dict]:
        """Batch form of judge_combined_single(). Returns {query_id: {rel, faith, cov, ctx_rel, ...}}."""
        from tqdm import tqdm
        scores = {}
        iterator = tqdm(results, desc="LLM judging (combined)") if show_progress else results
        for r in iterator:
            qid = r.get("query_id", r.get("query", "")[:50])
            scores[qid] = self.judge_combined_single(
                query=r.get("query", ""),
                answer=r.get("answer", ""),
                reference=r.get("reference", ""),
                evidence=r.get("evidence", []),
            )
        return scores

    def judge_ctx_rel_batch(
        self,
        results: List[Dict],
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """
        Context Relevance (CtxRel), scored 1-5 by an LLM evaluator, per Section
        2.4.2: "CtxRel is assessed by an LLM evaluator on a scale of 1-5 based
        on the adequacy of information supporting the answer." This replaces
        the earlier lexical token-overlap heuristic, which could not capture
        semantic relevance and did not match the paper's stated methodology.
        Returns: {query_id: ctx_rel_score (1-5)}
        """
        from tqdm import tqdm
        scores = {}
        iterator = tqdm(results, desc="CtxRel judging") if show_progress else results
        for r in iterator:
            qid = r.get("query_id", r.get("query", "")[:50])
            evidence = r.get("evidence", [])
            if not evidence:
                # No retrieval performed (e.g. mode m1 / LLM-only) — CtxRel is
                # not applicable; excluded from the mean by the caller.
                continue
            scores[qid] = self.judge_ctx_rel_single(r.get("query", ""), evidence)
        return scores

    def judge_ctx_rel_single(self, query: str, evidence: List[Dict]) -> int:
        """Score how adequately `evidence` supports answering `query`, 1-5."""
        evidence_text = self._format_evidence(evidence)
        if not evidence_text:
            return 1
        prompt = CTX_REL_PROMPT.format(query=query, evidence=evidence_text)
        # No try/except — see judge_single()'s comment above; a fabricated
        # ctx_rel=3 fallback would be indistinguishable from a real score
        # and never get retried.
        raw = self.client.generate_json(prompt)
        return max(1, min(5, int(raw.get("ctx_rel", 3))))

    @staticmethod
    def _format_evidence(evidence: List[Dict]) -> str:
        if not evidence:
            return ""
        parts = []
        for i, e in enumerate(evidence[:5], 1):
            parts.append(f"[Evidence {i}] (source: {e.get('filename', '?')})\n{e.get('text', '')[:500]}")
        return "\n\n".join(parts)

    def correlation_with_human(self, judge_scores: Dict, human_scores: Dict) -> float:
        """
        Compute Pearson correlation between LLM judge and human scores.
        human_scores: same format {qid: {rel, faith, cov}}
        """
        from scipy import stats
        llm_flat, human_flat = [], []
        for qid in judge_scores:
            if qid in human_scores:
                for dim in ["rel", "faith", "cov"]:
                    llm_flat.append(judge_scores[qid].get(dim, 3))
                    human_flat.append(human_scores[qid].get(dim, 3))
        if len(llm_flat) < 3:
            return 0.0
        r, _ = stats.pearsonr(llm_flat, human_flat)
        return round(float(r), 4)
