"""Agent registry — instantiate agents by name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trajectify.agents.base import BaseAgent

_AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}


def register_agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    """Class decorator that adds an agent to the registry."""
    _AGENT_REGISTRY[cls.agent_name()] = cls
    return cls


def create_agent(
    name: str,
    logs_dir: Path,
    model_name: str | None = None,
    **kwargs: Any,
) -> BaseAgent:
    """Create an agent instance by its registered name."""
    _ensure_builtins_loaded()

    cls = _AGENT_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_AGENT_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown agent '{name}'. Available: {available}")

    return cls(logs_dir=logs_dir, model_name=model_name, **kwargs)


def _ensure_builtins_loaded() -> None:
    """Import built-in agent modules so their @register_agent decorators fire."""
    if _AGENT_REGISTRY:
        return
    import trajectify.agents.nop  # noqa: F401

    try:
        import trajectify.agents.terminus  # noqa: F401
    except ImportError:
        pass
