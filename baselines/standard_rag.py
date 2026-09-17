"""Baseline: Standard RAG (retrieve-then-generate, with no variation)."""

import time
import logging
from typing import Dict

import numpy as np

from baselines.base import BasePipeline
from config import TOP_K

logger = logging.getLogger(__name__)


class StandardRAGPipeline(BasePipeline):
    """
    Standard RAG: retrieval of the Top-K always happens first, generation
    second. Retrieval is dense (cosine) and nothing else.
    Reference: Zhao et al. (2026) RAG Survey.
    """

    name = "standard_rag"

    PROMPT = """\
Answer the question based ONLY on the retrieved evidence below.
If the evidence does not contain the answer, say "I don't have this information."

Question: {query}

Evidence:
{evidence}

Answer:"""

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        t = time.time()
        result = self._base_result(query, reference_answer)
        try:
            # Dense retrieval alone (neither BM25 nor metadata filtering)
            query_vec = self.kb.embed_query(query)
            indices = self.kb.faiss_search(query_vec, k)
            evidence = [self.kb.chunks[i] for i in indices if i >= 0]
            result["evidence"] = evidence

            ev_text = self._format_evidence(evidence)
            prompt  = self.PROMPT.format(query=query, evidence=ev_text)
            answer  = self.client.generate(prompt, max_tokens=512)
            result["answer"]     = answer
            result["is_refusal"] = self._is_refusal(answer)
        except Exception as exc:
            logger.error(f"Standard RAG failed: {exc}")
        result["latency_s"] = round(time.time() - t, 3)
        return result
