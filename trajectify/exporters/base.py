"""Abstract base class for trajectory exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from trajectify.models.result import AgentResult, VerifierResult
from trajectify.models.trajectory import TrajectoryRecorder


class BaseExporter(ABC):
    """Pluggable exporter interface.

    Each exporter receives the recorder (with full trajectory data) and
    writes output in its own format.
    """

    @staticmethod
    @abstractmethod
    def format_name() -> str:
        """Short identifier for this export format (e.g. 'atif', 'sft')."""

    @abstractmethod
    def export(
        self,
        recorder: TrajectoryRecorder,
        output_dir: Path,
        *,
        run_name: str = "",
        agent_result: AgentResult | None = None,
        verifier_result: VerifierResult | None = None,
    ) -> Path:
        """Write the exported file(s) and return the primary output path."""
