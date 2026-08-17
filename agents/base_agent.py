"""
base_agent.py — Abstract base class for all MAS agents.

Every agent in the system should inherit from BaseAgent and implement
the `run()` method.
"""
import abc
import logging

logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """Abstract base for all Siren-AI agents."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")

    @abc.abstractmethod
    async def run(self, payload: dict) -> dict:
        """
        Execute the agent's core logic.

        Args:
            payload: Input data dict for this agent.

        Returns:
            Result data dict.
        """

    def __repr__(self) -> str:
        return f"<Agent name={self.name!r}>"
