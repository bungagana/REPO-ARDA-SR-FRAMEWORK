"""Wrapper for the Gemini API built on the google-genai SDK, with retries and rate limiting."""

import time
import json
import re
import logging
from typing import Optional

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY, GEMINI_MODEL, TEMPERATURE,
    MAX_OUTPUT_TOKENS, REQUEST_DELAY_S, MAX_RETRIES,
)

logger = logging.getLogger(__name__)


class GeminiClient:
    """A thread-safe Gemini client featuring exponential backoff and rate limiting."""

    def __init__(self, model: str = GEMINI_MODEL, temperature: float = TEMPERATURE):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set. Check your .env file.")
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name  = model
        self.temperature = temperature
        self._last_call  = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < REQUEST_DELAY_S:
            time.sleep(REQUEST_DELAY_S - elapsed)
        self._last_call = time.time()

    def generate(self, prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
        """Produce text for a prompt. Gives back the raw string."""
        self._throttle()
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=max_tokens,
        )
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text.strip()
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"Gemini error (attempt {attempt+1}/{MAX_RETRIES}): {exc}. "
                    f"Waiting {wait}s"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    raise

    def generate_json(self, prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> dict | list:
        """Produce a response and parse it as JSON, stripping markdown fences if any."""
        raw = self.generate(prompt, max_tokens)
        # Remove ```json ... ``` fence markers
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to pull out the first JSON object or array
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
            if match:
                return json.loads(match.group(1))
            logger.error(f"Failed to parse JSON from: {raw[:200]}")
            raise


# Convenience singleton held at module level
_client: Optional[GeminiClient] = None


def get_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
