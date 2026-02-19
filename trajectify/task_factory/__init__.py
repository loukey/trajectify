"""Task factory registry — instantiate factories by name."""

from __future__ import annotations

from trajectify.task_factory.base import GeneratedTask, TaskFactory

_FACTORY_REGISTRY: dict[str, type[TaskFactory]] = {}


def register_factory(cls: type[TaskFactory]) -> type[TaskFactory]:
    """Class decorator that adds a factory to the registry."""
    _FACTORY_REGISTRY[cls.factory_name()] = cls
    return cls


def create_factory(name: str) -> TaskFactory:
    """Create a factory instance by its registered name."""
    _ensure_builtins_loaded()
    cls = _FACTORY_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_FACTORY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown factory '{name}'. Available: {available}")
    return cls()


def list_factories() -> list[str]:
    """Return names of all registered factories."""
    _ensure_builtins_loaded()
    return sorted(_FACTORY_REGISTRY.keys())


def _ensure_builtins_loaded() -> None:
    """Import built-in factory modules so their @register_factory decorators fire."""
    if _FACTORY_REGISTRY:
        return
    import trajectify.task_factory.log_analysis  # noqa: F401
    import trajectify.task_factory.bug_fix  # noqa: F401
    import trajectify.task_factory.code_removal  # noqa: F401


__all__ = [
    "GeneratedTask",
    "TaskFactory",
    "create_factory",
    "list_factories",
    "register_factory",
]
