"""
prompt_guard.py — Custom prompt-injection guard.

Defends LLM prompts against common injection patterns:
    - Role-override attempts ("Ignore previous instructions…")
    - System-prompt leakage requests ("Print your system prompt")
    - Jailbreak token sequences

Usage:
    from security.prompt_guard import PromptGuard
    guard = PromptGuard()
    safe, reason = guard.check(user_input)
    if not safe:
        raise ValueError(f"Prompt injection detected: {reason}")
"""
from __future__ import annotations

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Injection pattern library ──────────────────────────────────────────────────
_PATTERNS: list[Tuple[re.Pattern, str]] = [
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
     "role-override attempt"),
    (re.compile(r"(print|reveal|show|output|leak)\s+(your\s+)?(system\s+)?prompt", re.I),
     "system-prompt leakage request"),
    (re.compile(r"you\s+are\s+now\s+(a\s+)?(?!siren)", re.I),
     "persona-hijack attempt"),
    (re.compile(r"(jailbreak|DAN|do\s+anything\s+now)", re.I),
     "known jailbreak token"),
    (re.compile(r"(\[INST\]|<\|system\|>|<\|user\|>|<\|assistant\|>)", re.I),
     "raw model-template injection"),
    (re.compile(r"(act\s+as|pretend\s+(to\s+be|you\s+are))\s+(?!siren)", re.I),
     "role-play injection"),
]

_MAX_INPUT_LENGTH = 4_096  # hard limit to prevent oversized inputs


class PromptGuard:
    """
    Stateless guard that scans a user input string for injection attempts.
    Thread-safe; create one instance and reuse it.
    """

    def check(self, text: str) -> Tuple[bool, str]:
        """
        Scan *text* for injection patterns.

        Returns:
            (True, "ok") if safe.
            (False, reason) if a pattern matched.
        """
        if len(text) > _MAX_INPUT_LENGTH:
            return False, "input exceeds maximum allowed length"

        for pattern, reason in _PATTERNS:
            if pattern.search(text):
                logger.warning("Prompt injection detected (%s): %.80r", reason, text)
                return False, reason

        return True, "ok"

    def sanitize(self, text: str) -> str:
        """
        Strip dangerous sub-strings from *text* (lossy; prefer check() and reject).
        Only use this if you must pass partial input downstream.
        """
        sanitized = text
        for pattern, _ in _PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized[: _MAX_INPUT_LENGTH]
