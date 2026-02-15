"""Runtime configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """User-provided agent configuration."""

    name: str = "terminus"
    model: str = "anthropic/claude-sonnet-4-20250514"
    temperature: float = 0.7
    collect_rollout_details: bool = False
    enable_summarize: bool = True
    max_turns: int = 1_000_000
    kwargs: dict[str, Any] = Field(default_factory=dict)
    override_timeout_sec: float | None = None


class EnvironmentRunConfig(BaseModel):
    """User-provided environment configuration."""

    type: str = "docker"
    force_build: bool = False
    delete_after: bool = True


class VerifierRunConfig(BaseModel):
    """User-provided verifier configuration."""

    disable: bool = False
    override_timeout_sec: float | None = None


class ExporterConfig(BaseModel):
    """Configuration for trajectory exporters."""

    formats: list[str] = Field(default_factory=lambda: ["atif"])
    output_dir: Path | None = None


class RunConfig(BaseModel):
    """Full configuration for a single run (was TrialConfig)."""

    model_config = {"arbitrary_types_allowed": True}

    run_name: str
    task_path: Path
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentRunConfig = Field(default_factory=EnvironmentRunConfig)
    verifier: VerifierRunConfig = Field(default_factory=VerifierRunConfig)
    exporter: ExporterConfig = Field(default_factory=ExporterConfig)
    output_dir: Path = Path("output")


class JobConfig(BaseModel):
    """Top-level configuration loaded from a YAML file."""

    model_config = {"arbitrary_types_allowed": True}

    tasks: list[Path]
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentRunConfig = Field(default_factory=EnvironmentRunConfig)
    verifier: VerifierRunConfig = Field(default_factory=VerifierRunConfig)
    exporter: ExporterConfig = Field(default_factory=ExporterConfig)
    n_concurrent: int = 1
    output_dir: Path = Path("output")
