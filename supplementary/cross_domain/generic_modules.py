
import math
import logging
from typing import Dict, List, Optional

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import time  # noqa: E402
from utils.llm_client import GeminiClient  # noqa: E402
from arda_sr.retrieval import HybridRetriever  # noqa: E402 (no domain-hardcoded prompts in here — reused as-is)
from config import (  # noqa: E402
    ENTROPY_THRESHOLD, DDA_DECISION_MARGIN, SR_LAMBDA, SR_NUM_SCENARIOS, TOP_K,
)

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN_CONTEXT = "Indonesian transmigration and regional governance"

MODES = ["m1", "m2", "m3", "m4"]


# ═════════════════════════════════════════════════════════════════════════
# GenericAQR — verbatim copy of arda_sr/aqr.py, prompt domain-parameterized
# ═════════════════════════════════════════════════════════════════════════

GENERIC_AQR_PROMPT = """\
You are an AQR (Adaptive Query Router) for a QA system in the following domain: {domain_context}.

Query: "{query}"

Classify this query into one of 4 response modes and provide feature scores.
Modes: m1=Direct Knowledge (no retrieval), m2=Factual Retrieval, m3=Hybrid/Ambiguous, m4=Policy Scenario
Probabilities must sum to 1.0. All scores are floats between 0.0 and 1.0.

Respond with ONLY valid JSON (no markdown, no explanation):
{{"features":{{"entity_signal":0.5,"domain_specificity":0.5,"temporal_ref":0.0,"multihop_signal":0.2,"context_dep":0.1}},"mode_probs":{{"m1":0.1,"m2":0.6,"m3":0.2,"m4":0.1}},"reasoning":"brief reason"}}

Now classify:
Query: "{query}"
JSON:"""


class GenericAQR:
    """Domain-parameterized copy of arda_sr.aqr.AQR. Logic identical."""

    def __init__(self, client: GeminiClient | None = None, tau_h: float = ENTROPY_THRESHOLD,
                 domain_context: str = DEFAULT_DOMAIN_CONTEXT):
        self.client = client or GeminiClient()
        self.tau_h = tau_h
        self.domain_context = domain_context

    def classify(self, query: str) -> Dict:
        prompt = GENERIC_AQR_PROMPT.format(query=query, domain_context=self.domain_context)
        try:
            result = self.client.generate_json(prompt)
        except Exception as exc:
            logger.warning(f"AQR classification failed: {exc}. Defaulting to m3.")
            return self._default_result()

        features   = result.get("features", {})
        mode_probs = result.get("mode_probs", {"m1": 0.25, "m2": 0.25, "m3": 0.25, "m4": 0.25})
        reasoning  = result.get("reasoning", "")

        total = sum(mode_probs.get(m, 0.0) for m in MODES)
        if total < 1e-9:
            total = 1.0
        mode_probs = {m: mode_probs.get(m, 0.0) / total for m in MODES}

        entropy  = self._entropy(mode_probs)
        hybrid   = entropy > self.tau_h
        dominant = max(mode_probs, key=mode_probs.get)
        mode     = "m3" if hybrid else dominant

        return {
            "mode": mode, "mode_probs": mode_probs, "entropy": round(entropy, 4),
            "features": features, "hybrid_path": hybrid, "reasoning": reasoning,
        }

    @staticmethod
    def _entropy(probs: Dict[str, float]) -> float:
        h = 0.0
        for p in probs.values():
            if p > 1e-12:
                h -= p * math.log2(p)
        return h

    def _default_result(self) -> Dict:
        uniform = {m: 0.25 for m in MODES}
        return {
            "mode": "m3", "mode_probs": uniform,
            "entropy": round(self._entropy(uniform), 4),
            "features": {}, "hybrid_path": True,
            "reasoning": "Default fallback due to classifier error.",
        }


# ═════════════════════════════════════════════════════════════════════════
# GenericDDA — verbatim copy of arda_sr/dda.py, prompts domain-parameterized
# ═════════════════════════════════════════════════════════════════════════

DEFAULT_BETAS = {"b1": 0.30, "b2": 0.25, "b3": 0.20, "b4": 0.25}
MIN_QUALITY_THRESHOLD = 0.3  # identical to arda_sr/dda.py — Algorithm 1 line 31

GENERIC_DIRECT_PROMPT = """\
You are a knowledgeable assistant for the following domain: {domain_context}.
Answer the following question using your general knowledge. Be concise and accurate.

Question: {query}

Answer:"""

GENERIC_RETRIEVAL_PROMPT = """\
You are a precise assistant for the following domain: {domain_context}.
Answer the question strictly based on the provided evidence. Do not add information not in the evidence.
If the evidence is insufficient, say so briefly.

Question: {query}

Evidence:
{evidence}

Answer:"""

# Utility-scoring and merge prompts contain no domain-specific text in the
# original — copied unchanged.
UTILITY_PROMPT = """\
Evaluate the quality of this answer to the given question.
Return ONLY JSON with scores in [0.0, 1.0]:

Question: {query}
Reference Answer: {reference}
Evidence Available to the Answer (may be empty if this was a parametric-only draft):
{evidence}
Candidate Answer: {answer}

{{
  "relevance": <0.0–1.0>,     // how relevant is the answer to the question
  "faithfulness": <0.0–1.0>,  // how well grounded in the Evidence above / factually accurate.
                               // If Evidence is empty, judge only against general factual plausibility.
  "coverage": <0.0–1.0>,      // completeness of information provided
  "risk": <0.0–1.0>           // risk of hallucination or incorrect info (LOWER is better answer)
}}"""

COMBINE_PROMPT = """\
You have two candidate answers. Combine the best elements of both into one comprehensive answer.
Prioritize factual accuracy over length.

Question: {query}
Answer A (direct knowledge): {a_dir}
Answer B (retrieval-grounded): {a_ret}

Combined Answer:"""


class GenericDDA:
    """Domain-parameterized copy of arda_sr.dda.DDA. Logic identical."""

    def __init__(self, client: GeminiClient | None = None, betas: Dict[str, float] | None = None,
                 domain_context: str = DEFAULT_DOMAIN_CONTEXT):
        self.client = client or GeminiClient()
        self.betas = betas or DEFAULT_BETAS
        self.domain_context = domain_context

    def arbitrate(self, query: str, evidence: List[Dict], reference: str = "") -> Dict:
        draft_dir = self._generate_direct(query)
        evidence_text = self._format_evidence(evidence)
        draft_ret = self._generate_retrieval(query, evidence_text) if evidence else ""

        u_dir = self._score_utility(query, draft_dir, reference, evidence_text="")
        u_ret = (
            self._score_utility(query, draft_ret, reference, evidence_text=evidence_text)
            if draft_ret else {"relevance": 0, "faithfulness": 0, "coverage": 0, "risk": 1}
        )

        score_dir = self._weighted_utility(u_dir)
        score_ret = self._weighted_utility(u_ret) if draft_ret else -1.0

        min_quality_passed = True
        if not draft_ret:
            selected = "direct"
            answer = draft_dir
            min_quality_passed = score_dir >= MIN_QUALITY_THRESHOLD
        elif score_dir - score_ret > DDA_DECISION_MARGIN:
            selected = "direct"
            answer = draft_dir
        elif score_ret - score_dir > DDA_DECISION_MARGIN:
            selected = "retrieval"
            answer = draft_ret
        else:
            selected = "combined"
            answer = self._combine(query, draft_dir, draft_ret)

        is_refusal = self._is_refusal(answer) or not min_quality_passed

        return {
            "answer": answer, "draft_dir": draft_dir, "draft_ret": draft_ret,
            "utility_dir": u_dir, "utility_ret": u_ret,
            "score_dir": round(score_dir, 4), "score_ret": round(score_ret, 4),
            "selected": selected, "min_quality_passed": min_quality_passed,
            "is_refusal": is_refusal, "betas": self.betas,
        }

    def _generate_direct(self, query: str) -> str:
        # No try/except: an API failure must propagate and abort this
        # query's arbitrate() call (retryable), not be swallowed into an
        # empty draft that would silently get checkpointed as complete.
        # Same fix as arda_sr/dda.py, 2026-08-15 spending-cap incident.
        prompt = GENERIC_DIRECT_PROMPT.format(query=query, domain_context=self.domain_context)
        return self.client.generate(prompt, max_tokens=512)

    def _generate_retrieval(self, query: str, evidence_text: str) -> str:
        prompt = GENERIC_RETRIEVAL_PROMPT.format(query=query, evidence=evidence_text,
                                                   domain_context=self.domain_context)
        return self.client.generate(prompt, max_tokens=768)

    def _score_utility(self, query: str, answer: str, reference: str, evidence_text: str = "") -> Dict:
        if not answer.strip():
            return {"relevance": 0.0, "faithfulness": 0.0, "coverage": 0.0, "risk": 1.0}
        prompt = UTILITY_PROMPT.format(
            query=query, reference=reference or answer, answer=answer,
            evidence=evidence_text or "(none — parametric-only draft)",
        )
        # No try/except -- a scoring failure must abort/retry the query, not
        # silently substitute a fabricated neutral 0.5 vector.
        scores = self.client.generate_json(prompt)
        return {
            "relevance":    float(scores.get("relevance", 0.5)),
            "faithfulness": float(scores.get("faithfulness", 0.5)),
            "coverage":     float(scores.get("coverage", 0.5)),
            "risk":         float(scores.get("risk", 0.5)),
        }

    def _weighted_utility(self, scores: Dict) -> float:
        b = self.betas
        return (b["b1"] * scores["relevance"] + b["b2"] * scores["faithfulness"]
                + b["b3"] * scores["coverage"] - b["b4"] * scores["risk"])

    def _combine(self, query: str, a_dir: str, a_ret: str) -> str:
        prompt = COMBINE_PROMPT.format(query=query, a_dir=a_dir, a_ret=a_ret)
        try:
            return self.client.generate(prompt, max_tokens=768)
        except Exception:
            return a_ret or a_dir

    @staticmethod
    def _format_evidence(evidence: List[Dict]) -> str:
        parts = []
        for i, e in enumerate(evidence, 1):
            parts.append(f"[Evidence {i}] (source: {e.get('filename','?')})\n{e.get('text','')[:600]}")
        return "\n\n".join(parts)

    @staticmethod
    def _is_refusal(text: str) -> bool:
        refusal_phrases = [
            "i don't have", "i do not have", "tidak memiliki informasi",
            "tidak dapat menjawab", "saya tidak tahu", "tidak ada informasi",
            "please refer to", "silakan merujuk", "cannot answer",
            "unable to answer", "no information available",
            "tidak tersedia", "belum tersedia", "tidak ditemukan",
            "tidak diketahui", "tidak dapat ditemukan", "tidak dapat dipastikan",
            "not available", "not found", "cannot be determined",
        ]
        t = text.lower()
        return any(p in t for p in refusal_phrases)


# ═════════════════════════════════════════════════════════════════════════
# GenericSR — verbatim copy of arda_sr/sr.py, prompts domain-parameterized
# ═════════════════════════════════════════════════════════════════════════

GENERIC_SCENARIO_GENERATION_PROMPT = """\
You are an analyst for the following domain: {domain_context}.
Generate exactly {n_scenarios} alternative intervention/decision scenarios for the following query.

Query: "{query}"

Relevant Evidence:
{evidence}

For each scenario, provide a structured analysis. Return ONLY this JSON array:
[
  {{
    "name": "<short scenario name>",
    "description": "<2-3 sentence description>",
    "p_success": <0.0–1.0>,
    "utility": <0.0–1.0>,
    "risk": <0.0–1.0>,
    "loss": <0.0–1.0>,
    "rationale": "<why this scenario, its key trade-offs>",
    "assumptions": "<key assumptions>",
    "timeline": "<estimated timeline>"
  }},
  ...
]

Scoring guidelines:
- p_success: probability of successful implementation given constraints
- utility: alignment with objectives and stakeholder needs
- risk: probability of adverse outcomes
- loss: magnitude of worst-case impact (0=negligible, 1=catastrophic)"""

GENERIC_POLICY_ANSWER_PROMPT = """\
You are an advisor for the following domain: {domain_context}.
Provide a structured recommendation based on the scenario analysis below.

Query: "{query}"

Optimal Scenario Selected: {optimal_name}
EU Score: {eu_score:.3f}

All Scenarios Evaluated:
{scenarios_text}

Evidence Used:
{evidence}

Provide a comprehensive answer that:
1. States the recommended intervention/scenario and why
2. Compares it explicitly against the alternatives
3. Identifies key risks and mitigation strategies
4. Gives actionable implementation steps
5. Notes any assumptions or limitations

Recommendation:"""


class GenericSR:
    """Domain-parameterized copy of arda_sr.sr.SR. Logic identical."""

    def __init__(self, client: GeminiClient | None = None, lam: float = SR_LAMBDA,
                 n_scenarios: int = SR_NUM_SCENARIOS, domain_context: str = DEFAULT_DOMAIN_CONTEXT):
        self.client = client or GeminiClient()
        self.lam = lam
        self.n_scenarios = n_scenarios
        self.domain_context = domain_context

    def reason(self, query: str, evidence: List[Dict]) -> Dict:
        evidence_text = self._format_evidence(evidence)
        scenarios = self._generate_scenarios(query, evidence_text)
        if not scenarios:
            return self._fallback(query)

        eu_scores = {}
        for s in scenarios:
            eu = self._expected_utility(s)
            eu_scores[s["name"]] = round(eu, 4)
            s["eu"] = round(eu, 4)

        optimal = max(scenarios, key=lambda s: s["eu"])
        scenarios_text = self._format_scenarios(scenarios)
        answer = self._generate_answer(query, optimal, scenarios_text, evidence_text)
        compliant = self._check_compliance(answer)

        return {
            "answer": answer, "scenarios": scenarios, "optimal_scenario": optimal,
            "eu_scores": eu_scores, "sr_compliant": compliant,
        }

    def _generate_scenarios(self, query: str, evidence_text: str) -> List[Dict]:
        prompt = GENERIC_SCENARIO_GENERATION_PROMPT.format(
            n_scenarios=self.n_scenarios, query=query, evidence=evidence_text[:2000],
            domain_context=self.domain_context,
        )
        try:
            raw = self.client.generate_json(prompt)
            if isinstance(raw, list):
                return [self._validate_scenario(s) for s in raw]
            return []
        except Exception as exc:
            logger.warning(f"SR scenario generation failed: {exc}")
            return []

    def _expected_utility(self, s: Dict) -> float:
        p = float(s.get("p_success", 0.5))
        u = float(s.get("utility",   0.5))
        r = float(s.get("risk",      0.5))
        lo = float(s.get("loss",     0.5))
        return p * u - self.lam * r * lo

    def _generate_answer(self, query: str, optimal: Dict, scenarios_text: str, evidence_text: str) -> str:
        prompt = GENERIC_POLICY_ANSWER_PROMPT.format(
            query=query, optimal_name=optimal.get("name", ""), eu_score=optimal.get("eu", 0.0),
            scenarios_text=scenarios_text, evidence=evidence_text[:1500],
            domain_context=self.domain_context,
        )
        try:
            return self.client.generate(prompt, max_tokens=1024)
        except Exception as exc:
            logger.warning(f"SR answer generation failed: {exc}")
            return f"Recommended scenario: {optimal.get('name','')}. EU={optimal.get('eu',0):.3f}."

    @staticmethod
    def _validate_scenario(s: dict) -> dict:
        defaults = {
            "name": "Scenario", "description": "", "p_success": 0.5,
            "utility": 0.5, "risk": 0.5, "loss": 0.5,
            "rationale": "", "assumptions": "", "timeline": "",
        }
        for k, v in defaults.items():
            if k not in s:
                s[k] = v
        for num_key in ["p_success", "utility", "risk", "loss"]:
            try:
                s[num_key] = max(0.0, min(1.0, float(s[num_key])))
            except (ValueError, TypeError):
                s[num_key] = 0.5
        return s

    @staticmethod
    def _format_evidence(evidence: List[Dict]) -> str:
        parts = []
        for i, e in enumerate(evidence, 1):
            parts.append(f"[{i}] {e.get('filename','?')}: {e.get('text','')[:400]}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_scenarios(scenarios: List[Dict]) -> str:
        lines = []
        for s in scenarios:
            lines.append(
                f"• {s['name']} — p_success={s['p_success']:.2f}, "
                f"utility={s['utility']:.2f}, risk={s['risk']:.2f}, "
                f"EU={s.get('eu',0):.3f}\n  {s.get('rationale','')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _check_compliance(answer: str) -> bool:
        required = ["scenario", "risk", "recommend"]
        t = answer.lower()
        return sum(1 for r in required if r in t) >= 2

    @staticmethod
    def _fallback(query: str) -> Dict:
        return {
            "answer": f"Unable to generate scenarios for: {query}",
            "scenarios": [], "optimal_scenario": {}, "eu_scores": {}, "sr_compliant": False,
        }


# ═════════════════════════════════════════════════════════════════════════
# GenericLLMJudge — verbatim copy of evaluation/llm_judge.py, domain-parameterized
# ═════════════════════════════════════════════════════════════════════════

GENERIC_JUDGE_PROMPT = """\
You are an independent evaluator assessing the quality of QA system responses
for the following domain: {domain_context}.

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

GENERIC_COMBINED_JUDGE_PROMPT = """\
You are an independent evaluator assessing the quality of QA system responses
for the following domain: {domain_context}.

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

GENERIC_CTX_REL_PROMPT = """\
You are an independent evaluator assessing RETRIEVAL quality (not answer quality)
for a QA system in the following domain: {domain_context}.

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


class GenericLLMJudge:
    """Domain-parameterized copy of evaluation.llm_judge.LLMJudge. Logic identical."""

    def __init__(self, client: GeminiClient | None = None, domain_context: str = DEFAULT_DOMAIN_CONTEXT):
        self.client = client or GeminiClient()
        self.domain_context = domain_context

    def judge_batch(self, results: List[Dict], show_progress: bool = True) -> Dict[str, Dict]:
        from tqdm import tqdm
        scores = {}
        iterator = tqdm(results, desc="LLM judging") if show_progress else results
        for r in iterator:
            qid = r.get("query_id", r.get("query", "")[:50])
            scores[qid] = self.judge_single(
                query=r.get("query", ""), answer=r.get("answer", ""),
                reference=r.get("reference", ""), evidence=r.get("evidence", []),
            )
        return scores

    def judge_single(self, query: str, answer: str, reference: str = "",
                      evidence: Optional[List[Dict]] = None) -> Dict:
        if not answer.strip():
            return {"rel": 1, "faith": 1, "cov": 1,
                    "rel_reason": "Empty answer", "faith_reason": "Empty", "cov_reason": "Empty"}
        evidence_text = self._format_evidence(evidence or [])
        prompt = GENERIC_JUDGE_PROMPT.format(
            query=query, reference=reference or "(no reference provided)",
            evidence=evidence_text or "(no evidence retrieved — no-retrieval/parametric answer)",
            answer=answer[:1500], domain_context=self.domain_context,
        )
        # No try/except: an API failure must abort this query's judging
        # (retryable on resume via the caller's checkpoint), not silently
        # substitute a fabricated rel=faith=cov=2 fallback that would be
        # indistinguishable from a real judgment and never get retried.
        # Same fix/rationale applied to evaluation/llm_judge.py and
        # arda_sr/pipeline.py on 2026-08-15 (spending-cap incident).
        raw = self.client.generate_json(prompt)
        return {
            "rel":          max(1, min(5, int(raw.get("rel", 3)))),
            "faith":        max(1, min(5, int(raw.get("faith", 3)))),
            "cov":          max(1, min(5, int(raw.get("cov", 3)))),
            "rel_reason":   str(raw.get("rel_reason", "")),
            "faith_reason": str(raw.get("faith_reason", "")),
            "cov_reason":   str(raw.get("cov_reason", "")),
        }

    def judge_combined_single(self, query: str, answer: str, reference: str = "",
                               evidence: Optional[List[Dict]] = None) -> Dict:
        """1-call variant: rel/faith/cov + ctx_rel together when evidence exists.
        Falls back to judge_single() when there's no evidence (ctx_rel N/A)."""
        if not evidence:
            return self.judge_single(query, answer, reference, evidence)
        if not answer.strip():
            return {"rel": 1, "faith": 1, "cov": 1, "ctx_rel": 1,
                    "rel_reason": "Empty answer", "faith_reason": "Empty",
                    "cov_reason": "Empty", "ctx_rel_reason": "Empty answer"}
        evidence_text = self._format_evidence(evidence)
        prompt = GENERIC_COMBINED_JUDGE_PROMPT.format(
            query=query, reference=reference or "(no reference provided)",
            evidence=evidence_text or "(no evidence retrieved -- no-retrieval/parametric answer)",
            answer=answer[:1500], domain_context=self.domain_context,
        )
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

    def judge_combined_batch(self, results: List[Dict], show_progress: bool = True) -> Dict[str, Dict]:
        from tqdm import tqdm
        scores = {}
        iterator = tqdm(results, desc="LLM judging (combined)") if show_progress else results
        for r in iterator:
            qid = r.get("query_id", r.get("query", "")[:50])
            scores[qid] = self.judge_combined_single(
                query=r.get("query", ""), answer=r.get("answer", ""),
                reference=r.get("reference", ""), evidence=r.get("evidence", []),
            )
        return scores

    def judge_ctx_rel_batch(self, results: List[Dict], show_progress: bool = True) -> Dict[str, int]:
        from tqdm import tqdm
        scores = {}
        iterator = tqdm(results, desc="CtxRel judging") if show_progress else results
        for r in iterator:
            qid = r.get("query_id", r.get("query", "")[:50])
            evidence = r.get("evidence", [])
            if not evidence:
                continue
            scores[qid] = self.judge_ctx_rel_single(r.get("query", ""), evidence)
        return scores

    def judge_ctx_rel_single(self, query: str, evidence: List[Dict]) -> int:
        evidence_text = self._format_evidence(evidence)
        if not evidence_text:
            return 1
        prompt = GENERIC_CTX_REL_PROMPT.format(query=query, evidence=evidence_text,
                                                  domain_context=self.domain_context)
        # No try/except -- see judge_single()'s comment above.
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


# ═════════════════════════════════════════════════════════════════════════
# GenericARDASRPipeline — verbatim copy of arda_sr/pipeline.py::ARDASRPipeline
# .run(), wired to the Generic* modules above. arda_sr/pipeline.py itself is
# NOT imported or modified — this is a fully independent orchestrator so the
# main repo stays untouched.
# ═════════════════════════════════════════════════════════════════════════

RETRIEVAL_MODES = {"m2", "m3", "m4"}


class GenericARDASRPipeline:
    """
    Domain-parameterized copy of arda_sr.pipeline.ARDASRPipeline.
    Control flow identical to the original .run(): AQR routes -> retrieval
    (if needed) -> SR (mode m4) or DDA (otherwise) produces the answer.

    Usage:
        pipeline = GenericARDASRPipeline(kb, client, domain_context="...")
        result   = pipeline.run(query, reference_answer="...")
    """

    def __init__(self, kb, client: GeminiClient | None = None, betas: Dict | None = None,
                 domain_context: str = DEFAULT_DOMAIN_CONTEXT):
        self.client    = client or GeminiClient()
        self.kb        = kb
        self.retriever = HybridRetriever(kb)
        self.aqr = GenericAQR(self.client, domain_context=domain_context)
        self.dda = GenericDDA(self.client, betas=betas, domain_context=domain_context)
        self.sr  = GenericSR(self.client, domain_context=domain_context)
        self.domain_context = domain_context

    def run(self, query: str, reference_answer: str = "", k: int = TOP_K) -> Dict:
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
            # NOTE: extract_metadata_from_query is still the Indonesia-specific
            # regex from arda_sr/retrieval.py (provinces/commodities/regulation
            # type) — for non-Indonesian-government domains it will simply
            # return an empty filter (no crash), so retrieval degrades
            # gracefully to plain hybrid scoring without pre-filtering. This
            # is a known, documented limitation of this cross-domain check,
            # not a bug — see AFTER-REVIEW/cross-domain/AUDIT.md.
            meta_filter = HybridRetriever.extract_metadata_from_query(query)
            evidence = self.retriever.retrieve(query, k=k, metadata_filter=meta_filter or None)
            result["evidence"] = evidence
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
