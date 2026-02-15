"""YAML configuration loading with environment variable support."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

from trajectify.config.models import JobConfig


class LLMConfig(BaseModel):
    """LLM provider settings loaded from llm_config.yaml."""

    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    api_base: str | None = None

    def apply_to_env(self) -> None:
        """Export api_key into environment variables so litellm can find it."""
        if self.api_key:
            os.environ.setdefault("OPENAI_API_KEY", self.api_key)
            os.environ.setdefault("ANTHROPIC_API_KEY", self.api_key)


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def load_llm_config() -> LLMConfig:
    """Load LLM config from config/llm_config.yaml.

    Returns a default LLMConfig if the file does not exist.
    """
    root = _find_project_root()
    config_path = root / "configs" / "llm_config.yaml"

    if not config_path.exists():
        return LLMConfig()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return LLMConfig.model_validate(raw)


def load_job_config(path: Path) -> JobConfig:
    """Load a job configuration from a YAML file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return JobConfig.model_validate(raw)
