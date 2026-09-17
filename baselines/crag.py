"""Baseline: CRAG (retrieval-augmented generation with correction)."""

import time
import logging
from typing import Dict, List

from baselines.base import BasePipeline
from arda_sr.retrieval import HybridRetriever
from config import TOP_K

logger = logging.getLogger(__name__)


class CRAGPipeline(BasePipeline):
    """
    CRAG: inspects how good the retrieval was before any generation happens.
    When retrieval comes back weak → it falls back on parametric knowledge.
    Reference: Yan et al. 2024 (Scopus).

    Fidelity note: what follows is a prompt-driven, same-backbone
    approximation of the retrieval-evaluation stage in CRAG. The original
    system relies on a compact trained relevance judge (a fine-tuned T5)
    together with a web-search stage for fallback and refinement; in this
    port the relevance judgment is handed to the shared LLM backbone through
    EVAL_PROMPT, and corrective retrieval stays inside the same closed
    corpus (no live web access exists in this setup). It should not be taken
    as a faithful reproduction of the original trained evaluator.
    """

    name = "crag"

    EVAL_PROMPT = """\
Evaluate the relevance of this evidence for answering the question.
Return ONLY JSON: {{"score": <0.0-1.0>, "verdict": "CORRECT"|"AMBIGUOUS"|"INCORRECT"}}

Question: {query}
Evidence:
{evidence}"""

    CORRECT_PROMPT = """\
Answer the question based on the verified evidence.
Question: {query}

Verified Evidence:
{evidence}

Answer:"""

    FALLBACK_PROMPT = """\
The retrieved evidence was insufficient. Answer using your best knowledge.
If uncertain, indicate it clearly.
Question: {query}
Answer:"""

    REFINE_PROMPT = """\
The evidence is partially relevant. Extract only the relevant parts and answer.
Question: {query}

Partial Evidence:
{evidence}

Answer (use only relevant parts):"""

    DECOMPOSE_PROMPT = """\
Decompose the question into 2-3 focused search queries that would help correct
or complete weak retrieval results. Return ONLY JSON:
{{"queries": ["...", "..."]}}

Question: {query}
Current evidence:
{evidence}"""

    RECOMPOSE_PROMPT = """\
Recompose the answer from the filtered evidence. Ignore irrelevant passages and
answer only if the evidence supports the claim.

Question: {query}
Filtered Evidence:
{evidence}

Answer:"""

    def __init__(self, kb, client=None):
        super().__init__(kb, client)
        self.retriever = HybridRetriever(kb)

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        t = time.time()
        result = self._base_result(query, reference_answer)
        try:
            evidence = self.retriever.retrieve(query, k=k)
            result["evidence"] = evidence

            # First, assess how good the retrieval is
            ev_text  = self._format_evidence(evidence)
            eval_out = self.client.generate_json(
                self.EVAL_PROMPT.format(query=query, evidence=ev_text[:1500])
            )
            verdict = str(eval_out.get("verdict", "AMBIGUOUS")).upper()
            score   = float(eval_out.get("score", 0.5))
            result["crag_verdict"] = verdict
            result["crag_score"]   = score

            # Second, generate according to the verdict
            if verdict == "CORRECT" and score >= 0.6:
                answer = self.client.generate(
                    self.CORRECT_PROMPT.format(query=query, evidence=ev_text), max_tokens=512
                )
            elif verdict == "INCORRECT" or score < 0.3:
                extra_evidence = self._corrective_retrieve(query, ev_text)
                if extra_evidence:
                    evidence.extend(extra_evidence)
                    corrected_text = self._format_evidence(evidence)
                    answer = self.client.generate(
                        self.RECOMPOSE_PROMPT.format(query=query, evidence=corrected_text),
                        max_tokens=512,
                    )
                else:
                    answer = self.client.generate(
                        self.FALLBACK_PROMPT.format(query=query), max_tokens=512
                    )
            else:
                extra_evidence = self._corrective_retrieve(query, ev_text)
                if extra_evidence:
                    evidence.extend(extra_evidence)
                    ev_text = self._format_evidence(evidence)
                answer = self.client.generate(
                    self.REFINE_PROMPT.format(query=query, evidence=ev_text), max_tokens=512
                )

            result["answer"]     = answer
            result["evidence"]   = evidence[:k]
            result["is_refusal"] = self._is_refusal(answer)
        except Exception as exc:
            logger.error(f"CRAG failed: {exc}")
        result["latency_s"] = round(time.time() - t, 3)
        return result

    def _corrective_retrieve(self, query: str, evidence_text: str) -> List[Dict]:
        """Carry out CRAG-style corrective retrieval within the same corpus."""
        try:
            out = self.client.generate_json(
                self.DECOMPOSE_PROMPT.format(query=query, evidence=evidence_text[:1000])
            )
            sub_queries = out.get("queries", []) if isinstance(out, dict) else []
        except Exception:
            sub_queries = []

        collected: List[Dict] = []
        seen = set()
        for sub_query in sub_queries[:3]:
            for chunk in self.retriever.retrieve(str(sub_query), k=2):
                key = (chunk.get("filename"), chunk.get("text", "")[:80])
                if key in seen:
                    continue
                seen.add(key)
                collected.append(chunk)
        return collected
