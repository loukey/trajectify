"""No-operation agent for testing the pipeline without an LLM."""

from __future__ import annotations

from trajectify.agents.base import BaseAgent
from trajectify.agents.registry import register_agent
from trajectify.environments.base import BaseEnvironment
from trajectify.models.result import AgentResult
from trajectify.models.trajectory import TrajectoryRecorder


@register_agent
class NopAgent(BaseAgent):
    """Does nothing — useful for verifying Docker + Verifier plumbing."""

    @staticmethod
    def agent_name() -> str:
        return "nop"

    async def setup(self, env: BaseEnvironment) -> None:
        self.logger.info("NopAgent.setup (no-op)")

    async def run(
        self,
        instruction: str,
        env: BaseEnvironment,
        recorder: TrajectoryRecorder,
    ) -> AgentResult:
        self.logger.info("NopAgent.run (no-op)")
        return AgentResult()
