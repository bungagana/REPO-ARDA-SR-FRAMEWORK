"""Base class (abstract) shared by every QA pipeline in this codebase."""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from utils.llm_client import GeminiClient
from utils.kb_builder import KnowledgeBase
from config import TOP_K

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Shared interface that both baseline and proposed methods implement."""

    name: str = "base"

    def __init__(self, kb: KnowledgeBase, client: GeminiClient | None = None):
        self.kb     = kb
        self.client = client or GeminiClient()

    @abstractmethod
    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
        """
        Execute this pipeline once for one query.

        The returned dict contains the keys:
          query, reference, method, answer, evidence,
          is_refusal, latency_s
          (extra keys may be supplied by subclasses)
        """

    def _base_result(self, query: str, reference: str) -> Dict:
        return {
            "query":      query,
            "reference":  reference,
            "method":     self.name,
            "answer":     "",
            "evidence":   [],
            "is_refusal": False,
            "latency_s":  0.0,
        }

    @staticmethod
    def _is_refusal(text: str) -> bool:
        phrases = [
            "i don't have", "i do not have", "tidak memiliki informasi",
            "tidak dapat menjawab", "saya tidak tahu", "tidak ada informasi",
            "please refer to", "silakan merujuk", "cannot answer",
            "unable to answer", "no information available", "maaf", "I cannot",
            "tidak tersedia", "belum tersedia", "tidak ditemukan",
            "tidak diketahui", "tidak dapat ditemukan", "tidak dapat dipastikan",
            "not available", "not found", "cannot be determined",
        ]
        t = text.lower()
        return any(p in t for p in phrases)

    @staticmethod
    def _format_evidence(evidence: List[Dict]) -> str:
        parts = []
        for i, e in enumerate(evidence, 1):
            parts.append(f"[Evidence {i}] (source: {e.get('filename','?')})\n{e.get('text','')[:600]}")
        return "\n\n".join(parts)
