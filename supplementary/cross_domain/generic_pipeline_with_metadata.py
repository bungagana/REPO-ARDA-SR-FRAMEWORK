import time
from typing import Callable, Dict, List, Optional

from arda_sr.retrieval import HybridRetriever
from config import TOP_K
from generic_modules import GenericARDASRPipeline, RETRIEVAL_MODES


class MetadataAwareARDASRPipeline(GenericARDASRPipeline):
    def __init__(
        self,
        kb,
        client=None,
        betas: Optional[Dict] = None,
        domain_context: str = "",
        metadata_extractor: Optional[Callable[[str], Dict]] = None,
    ):
        init_kwargs = {"betas": betas}
        if domain_context:
            init_kwargs["domain_context"] = domain_context
        super().__init__(kb, client, **init_kwargs)
        self.metadata_extractor = metadata_extractor

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        # Verbatim copy of GenericARDASRPipeline.run() (generic_modules.py)
        # with exactly one change: the metadata-extraction call is
        # pluggable instead of hardcoded to the Indonesia-specific
        # arda_sr.retrieval.HybridRetriever.extract_metadata_from_query.
        t_start = time.time()
        result = {
            "query": query, "reference": reference_answer, "method": "arda_sr",
            "mode": None, "answer": "", "evidence": [], "routing": {},
            "dda_info": {}, "sr_info": {}, "is_refusal": False, "latency_s": 0.0,
        }

        routing = self.aqr.classify(query)
        mode = routing["mode"]
        result["mode"] = mode
        result["routing"] = routing

        evidence: List[Dict] = []
        if mode in RETRIEVAL_MODES or routing.get("hybrid_path"):
            if self.metadata_extractor is not None:
                meta_filter = self.metadata_extractor(query)
            else:
                meta_filter = HybridRetriever.extract_metadata_from_query(query)
            evidence = self.retriever.retrieve(query, k=k, metadata_filter=meta_filter or None)
            result["evidence"] = evidence
            result["metadata_filter_used"] = meta_filter or {}
        result["hit_at_k"] = len(evidence) > 0

        if mode == "m4":
            sr_out = self.sr.reason(query, evidence)
            result["answer"]     = sr_out["answer"]
            result["sr_info"]    = sr_out
            result["is_refusal"] = not bool(sr_out["answer"].strip())
        else:
            dda_out = self.dda.arbitrate(query, evidence, reference_answer)
            result["answer"]     = dda_out["answer"]
            result["dda_info"]   = dda_out
            result["is_refusal"] = dda_out["is_refusal"]

        result["latency_s"] = round(time.time() - t_start, 3)
        return result
