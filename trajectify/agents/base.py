"""Abstract base class for all agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from trajectify.environments.base import BaseEnvironment
from trajectify.log import logger as global_logger
from trajectify.models.result import AgentResult
from trajectify.models.trajectory import TrajectoryRecorder


class BaseAgent(ABC):
    """Pluggable agent interface.

    Every agent must implement :meth:`setup` and :meth:`run`.
    The ``run`` method receives a ``TrajectoryRecorder`` (injected by the Runner)
    and returns an ``AgentResult``.
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        self.logs_dir = logs_dir
        self.model_name = model_name
        self.logger = (logger or global_logger).getChild(self.__class__.__name__)

    @staticmethod
    @abstractmethod
    def agent_name() -> str:
        """Unique identifier used in configuration files."""

    @abstractmethod
    async def setup(self, env: BaseEnvironment) -> None:
        """Install / initialise the agent inside the running environment."""

    @abstractmethod
    async def run(
        self,
        instruction: str,
        env: BaseEnvironment,
        recorder: TrajectoryRecorder,
    ) -> AgentResult:
        """Execute the task and return the result.

        Use *recorder* to log trajectory steps. The Runner will pass the
        recorder to exporters after the run completes.
        """
