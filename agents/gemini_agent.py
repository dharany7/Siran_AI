"""
gemini_agent.py — Concrete LLM agent backed by Google Gemini.

Uses the NEW ``google-genai`` package exclusively:
    from google import genai
    client = genai.Client(api_key=...)
    client.models.generate_content(model=..., contents=...)

The deprecated ``google-generativeai`` / ``google.generativeai``
package is NOT used anywhere in this codebase.

Usage (standalone):
    import asyncio
    from agents.gemini_agent import GeminiAgent
    agent = GeminiAgent()
    result = asyncio.run(agent.run({"prompt": "Summarise this siren event."}))
    print(result["text"])
"""
from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from backend.config import get_gemini_client, get_settings

logger = logging.getLogger(__name__)


class GeminiAgent(BaseAgent):
    """
    LLM reasoning agent powered by Google Gemini (google-genai SDK).

    Payload keys
    ------------
    prompt : str
        The user/system prompt to send to the model.
    model : str, optional
        Override the default model (``settings.gemini_model``).

    Returns
    -------
    dict with keys:
        text  : str   — the generated text response
        model : str   — model name used
        error : str   — present only if the call failed
    """

    def __init__(self) -> None:
        super().__init__(name="gemini")

    async def run(self, payload: dict) -> dict:
        prompt: str = payload.get("prompt", "")
        model: str = payload.get("model", get_settings().gemini_model)

        if not prompt:
            return {"error": "No prompt provided", "text": "", "model": model}

        self.logger.info("Calling Gemini model=%s prompt_len=%d", model, len(prompt))

        try:
            # ── New google-genai client pattern ─────────────────────────────
            client = get_gemini_client()          # cached singleton
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            # ───────────────────────────────────────────────────────────────
            return {"text": response.text, "model": model}

        except ValueError as exc:
            # API key not configured
            self.logger.error("Gemini config error: %s", exc)
            return {"error": str(exc), "text": "", "model": model}

        except Exception as exc:  # noqa: BLE001
            self.logger.error("Gemini call failed: %s", exc)
            return {"error": str(exc), "text": "", "model": model}
