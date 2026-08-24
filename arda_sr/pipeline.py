"""
ARDA-SR end-to-end inference pipeline.
Implements Algorithm 1 from the paper.

(q, D) → a*  via:
  1. AQR  — mode routing
  2. Retrieval — evidence construction (when required)
  3. DDA  — dual-draft arbitration (non-policy queries)
  4. SR   — policy-scenario reasoning (m4 queries)
"""

import logging
import time
from typing import Dict, List, Optional

from utils.llm_client import GeminiClient
from utils.kb_builder import KnowledgeBase
from arda_sr.aqr import AQR
from arda_sr.dda import DDA
from arda_sr.sr import SR
from arda_sr.retrieval import HybridRetriever
from config import TOP_K

logger = logging.getLogger(__name__)

RETRIEVAL_MODES = {"m2", "m3", "m4"}

# Same phrase list as DDA._is_refusal() (arda_sr/dda.py) — reused here because
# the no-DDA ablation fallback path (_simple_generate) used to hardcode
# is_refusal=False unconditionally, which silently forced FRR to 0.0 for any
# ablation variant with use_dda=False (V0/V1/V2). Kept as a free function
# (not imported from DDA) so this module has no import-time dependency on a
# DDA instance existing.
_REFUSAL_PHRASES = [
    "i don't have", "i do not have", "tidak memiliki informasi",
    "tidak dapat menjawab", "saya tidak tahu", "tidak ada informasi",
    "please refer to", "silakan merujuk", "cannot answer",
    "unable to answer", "no information available",
    "tidak tersedia", "belum tersedia", "tidak ditemukan",
    "tidak diketahui", "tidak dapat ditemukan", "tidak dapat dipastikan",
    "not available", "not found", "cannot be determined",
]


def _looks_like_refusal(text: str) -> bool:
    if not text or not text.strip():
        return True
    t = text.lower()
    return any(p in t for p in _REFUSAL_PHRASES)


class ARDASRPipeline:
    """
    Full ARDA-SR inference pipeline.

    Usage:
        pipeline = ARDASRPipeline(kb)
        result   = pipeline.run(query, reference_answer="...")
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        client: GeminiClient | None = None,
        betas: Dict | None = None,
        # ablation flags
        use_aqr: bool = True,
        use_dda: bool = True,
        use_sr:  bool = True,
        use_hybrid_retrieval: bool = True,
    ):
        self.client    = client or GeminiClient()
        self.kb        = kb
        self.retriever = HybridRetriever(kb)
        self.aqr      = AQR(self.client) if use_aqr else None
        self.dda      = DDA(self.client, betas=betas) if use_dda else None
        self.sr       = SR(self.client)  if use_sr  else None
        self.use_hybrid = use_hybrid_retrieval

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        """
        Run the full ARDA-SR pipeline.
        Returns a result dict ready for evaluation.
        """
        t_start = time.time()
        result = {
            "query":      query,
            "reference":  reference_answer,
            "method":     "arda_sr",
            "mode":       None,
            "answer":     "",
            "evidence":   [],
            "routing":    {},
            "dda_info":  {},
            "sr_info":   {},
            "is_refusal": False,
            "latency_s":  0.0,
        }

        # ── Step 1: AQR routing ──────────────────────────────────────────
        if self.aqr is not None:
            routing = self.aqr.classify(query)
        else:
            # Ablation: no AQR → default to m2 (always retrieve)
            routing = {"mode": "m2", "mode_probs": {}, "entropy": 0.0,
                       "features": {}, "hybrid_path": False, "reasoning": "AQR disabled"}

        mode   = routing["mode"]
        result["mode"]    = mode
        result["routing"] = routing

        # ── Step 2: Retrieval (when required) ────────────────────────────
        evidence: List[Dict] = []
        if mode in RETRIEVAL_MODES or routing.get("hybrid_path"):
            meta_filter = HybridRetriever.extract_metadata_from_query(query)
            evidence = self.retriever.retrieve(query, k=k, metadata_filter=meta_filter or None)
            result["evidence"] = evidence

        result["hit_at_k"] = len(evidence) > 0

        # ── Step 3a: SR for policy-scenario queries ──────────────────────
        if mode == "m4" and self.sr is not None:
            sr_out = self.sr.reason(query, evidence)
            result["answer"]     = sr_out["answer"]
            result["sr_info"]   = sr_out
            result["is_refusal"] = not bool(sr_out["answer"].strip())

        # ── Step 3b: DDA for all other queries ───────────────────────────
        else:
            if self.dda is not None:
                dda_out = self.dda.arbitrate(query, evidence, reference_answer)
                result["answer"]     = dda_out["answer"]
                result["dda_info"]  = dda_out
                result["is_refusal"] = dda_out["is_refusal"]
            else:
                # Ablation: no DDA → simple retrieval-grounded generation
                result["answer"]     = self._simple_generate(query, evidence)
                result["is_refusal"] = _looks_like_refusal(result["answer"])

        result["latency_s"] = round(time.time() - t_start, 3)
        return result

    def _simple_generate(self, query: str, evidence: List[Dict]) -> str:
        """Fallback when DDA is disabled (ablation V0/V1/V2).

        Deliberately does NOT catch generate()'s exceptions (API errors,
        spending-cap 429s, etc.) — letting them propagate up means the
        query is never appended to results/checkpointed as "done" (see
        generate_resumable() in run_ablation_full.py), so a resumed run
        correctly retries it instead of silently recording an empty answer
        as a completed, judged query. See chat discussion 2026-08-15: the
        old swallow-and-return-"" behavior let 320/1000 V2 queries get
        marked complete during a spending-cap outage.
        """
        if evidence:
            ev_text = "\n\n".join(
                f"[{i+1}] {e.get('text','')[:400]}" for i, e in enumerate(evidence)
            )
            prompt = f"Answer the question based on the evidence.\n\nQuestion: {query}\n\nEvidence:\n{ev_text}\n\nAnswer:"
        else:
            prompt = f"Answer the following question:\n\nQuestion: {query}\n\nAnswer:"
        return self.client.generate(prompt, max_tokens=512)
