"""Abstract base class for execution environments."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from trajectify.log import logger as global_logger


class ExecResult(BaseModel):
    """Result of executing a command inside the environment."""

    stdout: str | None = None
    stderr: str | None = None
    return_code: int = 0


class BaseEnvironment(ABC):
    """Abstract interface for a containerised execution environment."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        host_agent_dir: Path,
        host_verifier_dir: Path,
        *,
        cpus: int = 1,
        memory_mb: int = 1024,
        allow_internet: bool = True,
        logger: logging.Logger | None = None,
    ):
        self.environment_dir = environment_dir
        self.environment_name = environment_name
        self.session_id = session_id
        self.host_agent_dir = host_agent_dir
        self.host_verifier_dir = host_verifier_dir
        self.cpus = cpus
        self.memory_mb = memory_mb
        self.allow_internet = allow_internet
        self.logger = (logger or global_logger).getChild(self.__class__.__name__)

    @abstractmethod
    async def start(self, force_build: bool = False) -> None:
        """Build (if needed) and start the environment."""

    @abstractmethod
    async def stop(self, delete: bool = True) -> None:
        """Stop the environment. If *delete*, also remove images/volumes."""

    @abstractmethod
    async def upload_file(self, source: Path | str, target: str) -> None:
        """Copy a local file into the environment."""

    @abstractmethod
    async def upload_dir(self, source: Path | str, target: str) -> None:
        """Copy a local directory into the environment."""

    @abstractmethod
    async def download_file(self, source: str, target: Path | str) -> None:
        """Copy a file from the environment to the local machine."""

    @abstractmethod
    async def download_dir(self, source: str, target: Path | str) -> None:
        """Copy a directory from the environment to the local machine."""

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        """Execute a shell command inside the environment."""
