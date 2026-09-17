"""
Baseline: FLARE — Forward-Looking Active REtrieval Augmented Generation.

Text is produced step by step; spans with low confidence are spotted,
retrieval is triggered to close the knowledge gaps, and generation resumes.
Reference: Jiang et al. 2023, EMNLP (Scopus indexed).
"""

import time
import re
import logging
from typing import Dict, List

from baselines.base import BasePipeline
from arda_sr.retrieval import HybridRetriever
from config import TOP_K

logger = logging.getLogger(__name__)

MAX_FLARE_ITERS = 3


class FLAREPipeline(BasePipeline):
    """
    FLARE: chooses on its own when retrieval is warranted by anticipating
    upcoming tokens and gauging confidence. Low confidence → retrieve →
    regenerate.

    Fidelity note: in the original FLARE (Jiang et al. 2023), low confidence
    is spotted from the generating LM's own per-token output probabilities
    (a white-box signal) over the forward-looking sentence. The Gemini API
    available here does not surface per-token log-probabilities for this
    call pattern, so this port falls back on a separate LLM self-judgment
    (CONFIDENCE_PROMPT) that asks whether the predicted sentence "contains
    uncertainty" — a black-box stand-in for FLARE's central triggering
    mechanism, not an exact copy. That mechanism gap is significant and
    ought to be stated plainly whenever results are compared with FLARE's
    originally reported behavior.
    """

    name = "flare"

    INITIAL_PROMPT = """\
Write the first sentence of an answer to the question about Indonesian transmigration.
Be concise.

Question: {query}

First sentence:"""

    LOOKAHEAD_PROMPT = """\
Predict the next sentence that should follow this partial answer. The sentence
should state the next factual claim needed to answer the question.

Question: {query}
Partial answer: {partial}

Next sentence:"""

    CONTINUE_PROMPT = """\
Regenerate or continue the next part of the answer using the retrieved evidence.

Question: {query}
Previous partial answer: {partial}
Retrieved evidence:
{evidence}

Next supported sentence or final answer continuation:"""

    CONFIDENCE_PROMPT = """\
Does this predicted next sentence contain factual uncertainty or need external evidence?
Return ONLY JSON: {{"uncertain": true/false, "query": "<retrieval query or empty>"}}

Question: {query}
Partial Answer: {partial}
Predicted Next Sentence: {prediction}"""

    def __init__(self, kb, client=None):
        super().__init__(kb, client)
        self.retriever = HybridRetriever(kb)

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        t = time.time()
        result = self._base_result(query, reference_answer)
        partial = ""
        evidence_all: List[Dict] = []

        try:
            # Produce the opening partial answer
            partial = self.client.generate(
                self.INITIAL_PROMPT.format(query=query), max_tokens=384
            )

            for i in range(MAX_FLARE_ITERS):
                prediction = self.client.generate(
                    self.LOOKAHEAD_PROMPT.format(query=query, partial=partial[:600]),
                    max_tokens=160,
                )
                conf = self.client.generate_json(
                    self.CONFIDENCE_PROMPT.format(
                        query=query,
                        partial=partial[:500],
                        prediction=prediction[:300],
                    )
                )
                if not conf.get("uncertain", False):
                    partial = f"{partial.strip()} {prediction.strip()}".strip()
                    break
                topic = conf.get("query") or prediction or query

                # Fetch evidence from the forward-looking predicted text.
                evidence = self.retriever.retrieve(topic or query, k=3)
                evidence_all.extend(evidence)
                ev_text = self._format_evidence(evidence)

                # Resume generation using the evidence just retrieved
                continuation = self.client.generate(
                    self.CONTINUE_PROMPT.format(
                        query=query, partial=partial[:400], evidence=ev_text[:1000]
                    ),
                    max_tokens=384,
                )
                partial = f"{partial.strip()} {continuation.strip()}".strip()

            # Wrap up and tidy the output
            answer = re.sub(r"\[UNCERTAIN:.*?\]", "", partial).strip()
            result["answer"]       = answer
            result["evidence"]     = evidence_all[:k]
            result["is_refusal"]   = self._is_refusal(answer)
            result["flare_iters"]  = i + 1
        except Exception as exc:
            logger.error(f"FLARE failed: {exc}")
        result["latency_s"] = round(time.time() - t, 3)
        return result
