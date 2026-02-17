"""Task and TaskConfig models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TaskPaths(BaseModel):
    """Resolved paths inside a task directory.

    Each loader fills in the concrete paths for its format.
    """

    model_config = {"arbitrary_types_allowed": True}

    task_dir: Path
    config_path: Path
    instruction_path: Path | None = None
    environment_dir: Path
    tests_dir: Path
    test_script_path: Path
    solution_dir: Path | None = None


class EnvironmentSpec(BaseModel):
    """Environment requirements declared inside task.toml."""

    build_timeout_sec: float = 600
    cpus: int = 1
    memory_mb: int = 1024
    allow_internet: bool = True
    docker_image: str | None = None
    memory: str | None = None
    storage: str | None = None


class AgentConstraints(BaseModel):
    """Agent constraints declared inside task.toml."""

    timeout_sec: float = 300


class VerifierConstraints(BaseModel):
    """Verifier constraints declared inside task.toml."""

    timeout_sec: float = 120
    env: dict[str, str] | None = None


class TaskConfig(BaseModel):
    """The content of a task.toml file."""

    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    agent: AgentConstraints = Field(default_factory=AgentConstraints)
    verifier: VerifierConstraints = Field(default_factory=VerifierConstraints)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """A loaded task ready for execution."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    instruction: str
    config: TaskConfig
    paths: TaskPaths
