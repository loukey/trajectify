"""Abstract base class for task loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from trajectify.models.task import Task


class BaseTaskLoader(ABC):
    """Pluggable task loader interface."""

    @staticmethod
    @abstractmethod
    def format_name() -> str:
        """Short identifier for this task format (e.g. 'terminal_bench')."""

    @abstractmethod
    def load(self, task_dir: Path) -> Task:
        """Load a task from the given directory."""
