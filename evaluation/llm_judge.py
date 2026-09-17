"""
Evaluation with an LLM acting as judge, in the spirit of the RAGAS methodology.
Answer quality is rated along 3 axes: Relevance, Faithfulness, Coverage (1–5 Likert).
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
    Automated evaluation driven by an LLM judge.
    When consumed by metrics, scores are rescaled to [0,1] (raw value / 5).
    """

    def __init__(self, client: GeminiClient | None = None):
        self.client = client or GeminiClient()

    def judge_batch(
        self,
        results: List[Dict],
        show_progress: bool = True,
    ) -> Dict[str, Dict]:
        """
        Assign scores to a list of result dicts.
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
        """Score one (query, answer, reference, evidence) tuple.

        `evidence` holds the retrieved chunk dicts the pipeline actually used
        for this query (identical shape as elsewhere: {filename, text, ...}).
        Supplying it allows Faithfulness to be judged against genuine evidence,
        following the paper's Faith(a) definition (Section 2.4.2) — a judge
        given only (query, reference, answer) cannot assess grounding at all.
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
        # Deliberately no try/except: an API failure here has to abort this
        # query's judging (so it can be retried on resume) rather than quietly
        # swap in a fabricated rel=faith=cov=2 fallback, which judge_resumable()
        # would forever take as a genuine completed judgment. Same reasoning as
        # the _simple_generate() fix in arda_sr/pipeline.py (2026-08-15
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
        One-call counterpart to judge_single() that also returns ctx_rel within
        the same LLM request, for queries carrying evidence -- this trims
        judging from 2 API calls/query to 1 on the evidence-bearing subset.
        Introduced 2026-08-15 to accelerate the not-yet-started V3 rerun; it was
        NOT applied to retroactively alter V1/V2's already-collected (2-call)
        judged data, which stays valid untouched (compute_ctx_rel() reads only
        the final `ctx_rel` field value, not how many calls produced it). When
        no evidence exists it delegates to plain judge_single(), since ctx_rel
        does not apply there anyway.
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
        # No try/except here -- same reasoning as judge_single()/judge_ctx_rel_single().
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
        """Batch version of judge_combined_single(). Returns {query_id: {rel, faith, cov, ctx_rel, ...}}."""
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
        Context Relevance (CtxRel), scored 1-5 by an LLM evaluator, as stated in
        Section 2.4.2: "CtxRel is assessed by an LLM evaluator on a scale of 1-5 based
        on the adequacy of information supporting the answer." It supersedes the
        earlier lexical token-overlap heuristic, which failed to capture semantic
        relevance and did not follow the paper's stated methodology.
        Returns: {query_id: ctx_rel_score (1-5)}
        """
        from tqdm import tqdm
        scores = {}
        iterator = tqdm(results, desc="CtxRel judging") if show_progress else results
        for r in iterator:
            qid = r.get("query_id", r.get("query", "")[:50])
            evidence = r.get("evidence", [])
            if not evidence:
                # Retrieval was skipped (e.g. mode m1 / LLM-only), so CtxRel
                # does not apply and the caller drops it from the mean.
                continue
            scores[qid] = self.judge_ctx_rel_single(r.get("query", ""), evidence)
        return scores

    def judge_ctx_rel_single(self, query: str, evidence: List[Dict]) -> int:
        """Rate on a 1-5 scale how well `evidence` backs up an answer to `query`."""
        evidence_text = self._format_evidence(evidence)
        if not evidence_text:
            return 1
        prompt = CTX_REL_PROMPT.format(query=query, evidence=evidence_text)
        # No try/except — see the comment on judge_single() above; an invented
        # ctx_rel=3 fallback would look exactly like a genuine score and would
        # never be retried.
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
        Calculate the Pearson correlation of LLM-judge scores against human scores.
        human_scores: identical format {qid: {rel, faith, cov}}
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
