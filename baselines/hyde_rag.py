"""Baseline: HyDE-RAG (Hypothetical Document Embeddings)."""

import time
import logging
from typing import Dict

from baselines.base import BasePipeline
from config import TOP_K

logger = logging.getLogger(__name__)


class HyDERAGPipeline(BasePipeline):
    """
    HyDE-RAG: draft a hypothetical answer → embed that draft → pull back
    similar genuine documents. It tightens retrieval alignment on complex
    questions.
    Reference: IEEE 11080443.
    """

    name = "hyde_rag"

    HYDE_PROMPT = """\
Write a short, plausible answer to the following question about Indonesian transmigration.
This is a HYPOTHETICAL answer used for document retrieval purposes only.
Be specific and use domain terminology.

Question: {query}

Hypothetical Answer (2–3 sentences):"""

    ANSWER_PROMPT = """\
Answer the question based on the retrieved evidence below.
Question: {query}

Evidence:
{evidence}

Answer:"""

    def __init__(self, kb, client=None):
        super().__init__(kb, client)

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        t = time.time()
        result = self._base_result(query, reference_answer)
        try:
            # First, draft the hypothetical document
            hyp_answer = self.client.generate(
                self.HYDE_PROMPT.format(query=query), max_tokens=256
            )
            # Second, run dense search against the hypothetical document's embedding.
            hyp_vec = self.kb.embed_query(hyp_answer)
            indices = self.kb.faiss_search(hyp_vec, k)
            evidence = [self.kb.chunks[i] for i in indices if i >= 0]
            result["evidence"] = evidence

            # Third, produce the final answer from the retrieved evidence
            ev_text = self._format_evidence(evidence)
            answer  = self.client.generate(
                self.ANSWER_PROMPT.format(query=query, evidence=ev_text), max_tokens=512
            )
            result["answer"]     = answer
            result["is_refusal"] = self._is_refusal(answer)
        except Exception as exc:
            logger.error(f"HyDE-RAG failed: {exc}")
        result["latency_s"] = round(time.time() - t, 3)
        return result
