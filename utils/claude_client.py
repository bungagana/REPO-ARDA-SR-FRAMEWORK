"""
Wrapper for the Claude API (official `anthropic` SDK) that handles QA generation.

It serves QA dataset generation, per Section 2.4.1 ("The QAs were generated
using claude-haiku-4-5"). Every other piece (ARDA-SR generation, judging, and
the baselines) still runs on Gemini via config.py — this client covers QA
generation alone by design.

Non-streaming: since QA generation emits brief, structured JSON (batches of
~15 QA pairs, comfortably below the ~16000-token non-streaming safety ceiling),
the streaming/get_final_message() pattern reserved for long generations is not
needed here.

JSON output: obtained through prompting (the current DK_PROMPT/FR_PROMPT/etc.
templates already say to "Return ONLY a JSON array"), rather than
output_config.format structured outputs — this keeps it aligned with
GeminiClient.generate_json()'s regex-fallback parsing, so QAGenerator
(02_generate_qa.py) runs untouched against either backend.
"""

import time
import json
import re
import logging
from typing import Optional

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_QA_MODEL, REQUEST_DELAY_S, MAX_RETRIES

logger = logging.getLogger(__name__)


class ClaudeClient:
    """
    A slim wrapper over anthropic.Anthropic exposing the same generate()/
    generate_json() interface as utils.llm_client.GeminiClient, letting it
    stand in for QA-generation call sites without changes.
    """

    def __init__(self, model: str = CLAUDE_QA_MODEL):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set. Check your .env file.")
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = model
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
                response = self._client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                if response.stop_reason == "refusal":
                    raise RuntimeError(
                        f"Claude declined the request (stop_reason=refusal): "
                        f"{getattr(response, 'stop_details', None)}"
                    )
                text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                return text.strip()
            except anthropic.RateLimitError as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"Claude rate limited (attempt {attempt+1}/{MAX_RETRIES}): {exc}. "
                    f"Waiting {wait}s"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    raise
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Claude server error (attempt {attempt+1}/{MAX_RETRIES}): "
                        f"{exc}. Waiting {wait}s"
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(wait)
                        continue
                raise
            except anthropic.APIConnectionError as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"Claude connection error (attempt {attempt+1}/{MAX_RETRIES}): "
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


_client: Optional[ClaudeClient] = None


def get_client() -> ClaudeClient:
    global _client
    if _client is None:
        _client = ClaudeClient()
    return _client
