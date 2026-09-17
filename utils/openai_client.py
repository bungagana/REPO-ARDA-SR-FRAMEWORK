"""
Wrapper for the OpenAI API (official `openai` SDK) that serves LLM-as-judge evaluation.

It is used ONLY as an alternative judge model (a drop-in for evaluation.llm_judge.LLMJudge),
so judging is not carried out by the very model family (Gemini) that also
generates the answers under judgment — this answers the "judge is not
independent of the generator" critique. Generation/retrieval/DDA/SR/AQR all
remain on Gemini; QA dataset generation remains on Claude Haiku
(utils/claude_client.py). This client is limited to judging by design.

Uses the Responses API (`client.responses.create`), the endpoint currently
recommended for GPT-5.x models, together with `reasoning: {"effort": ...}` —
judging is a short structured-JSON scoring task, so the default reasoning
effort is low (see config.GPT_JUDGE_REASONING_EFFORT).

JSON output: produced through prompting (the same convention as
GeminiClient.generate_json() / ClaudeClient.generate_json()) so that LLMJudge
and QAValidator run untouched against this client too — no SDK-specific
structured-output schema is employed, keeping behavior identical (and thus
comparable) across judge backends.
"""

import time
import json
import re
import logging
from typing import Optional

import openai

from config import OPENAI_API_KEY, GPT_JUDGE_MODEL, GPT_JUDGE_REASONING_EFFORT, REQUEST_DELAY_S, MAX_RETRIES

logger = logging.getLogger(__name__)


class GPTJudgeClient:
    """
    A slim wrapper around openai.OpenAI exposing the same generate()/
    generate_json() interface as utils.llm_client.GeminiClient and
    utils.claude_client.ClaudeClient, letting it stand in for
    evaluation.llm_judge.LLMJudge(client=...).
    """

    def __init__(self, model: str = GPT_JUDGE_MODEL, reasoning_effort: str = GPT_JUDGE_REASONING_EFFORT):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Check your .env file.")
        self._client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model
        self.reasoning_effort = reasoning_effort
        self._last_call = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < REQUEST_DELAY_S:
            time.sleep(REQUEST_DELAY_S - elapsed)
        self._last_call = time.time()

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """Produce text for a prompt without streaming. Gives back the raw string."""
        self._throttle()
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.responses.create(
                    model=self.model_name,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": self.reasoning_effort},
                )
                text = getattr(response, "output_text", None)
                if text is None:
                    # Fallback: scan the output items until the first text part turns up.
                    text = ""
                    for item in getattr(response, "output", []) or []:
                        for part in getattr(item, "content", []) or []:
                            if getattr(part, "type", "") == "output_text":
                                text = part.text
                                break
                        if text:
                            break
                return (text or "").strip()
            except openai.RateLimitError as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"GPT judge rate limited (attempt {attempt+1}/{MAX_RETRIES}): {exc}. "
                    f"Waiting {wait}s"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    raise
            except openai.APIStatusError as exc:
                if exc.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning(
                        f"GPT judge server error (attempt {attempt+1}/{MAX_RETRIES}): "
                        f"{exc}. Waiting {wait}s"
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(wait)
                        continue
                raise
            except openai.APIConnectionError as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"GPT judge connection error (attempt {attempt+1}/{MAX_RETRIES}): "
                    f"{exc}. Waiting {wait}s"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    raise

    def generate_json(self, prompt: str, max_tokens: int = 2048) -> dict | list:
        """Produce a response and parse it as JSON, dropping markdown fences when found."""
        raw = self.generate(prompt, max_tokens)
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
            if match:
                return json.loads(match.group(1))
            logger.error(f"Failed to parse JSON from: {raw[:200]}")
            raise


_client: Optional[GPTJudgeClient] = None


def get_client() -> GPTJudgeClient:
    global _client
    if _client is None:
        _client = GPTJudgeClient()
    return _client
