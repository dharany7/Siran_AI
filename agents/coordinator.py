"""
coordinator.py — Orchestrates multiple agents for a single inference request.

The Coordinator receives a raw payload, fans out to the appropriate
sub-agents (audio, ANPR, LLM, etc.), and returns a merged result.
"""
import asyncio
import logging
from typing import List

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class Coordinator:
    """Fan-out coordinator for the Multi-Agent System."""

    def __init__(self, agents: List[BaseAgent]) -> None:
        self.agents = agents

    async def dispatch(self, payload: dict) -> dict:
        """
        Run all registered agents concurrently and merge their results.

        Args:
            payload: Shared input data passed to every agent.

        Returns:
            Merged dict of all agent outputs, keyed by agent name.
        """
        tasks = {agent.name: agent.run(payload) for agent in self.agents}
        results = {}
        for name, coro in tasks.items():
            try:
                results[name] = await coro
            except Exception as exc:  # noqa: BLE001
                logger.error("Agent %s failed: %s", name, exc)
                results[name] = {"error": str(exc)}
        return results
