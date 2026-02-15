"""ATIF v1.2 trajectory exporter."""

from __future__ import annotations

import json
from pathlib import Path

from trajectify.exporters.base import BaseExporter
from trajectify.models.result import AgentResult, VerifierResult
from trajectify.models.trajectory import TrajectoryRecorder


class AtifExporter(BaseExporter):
    """Exports trajectories in ATIF (Agent Trajectory Interchange Format) v1.2."""

    @staticmethod
    def format_name() -> str:
        return "atif"

    def export(
        self,
        recorder: TrajectoryRecorder,
        output_dir: Path,
        *,
        run_name: str = "",
        agent_result: AgentResult | None = None,
        verifier_result: VerifierResult | None = None,
    ) -> Path:
        trajectory = recorder.build_trajectory(session_id=run_name)

        # Attach verifier rewards to final_metrics.extra if available
        if verifier_result and trajectory.final_metrics:
            extra = trajectory.final_metrics.extra or {}
            extra["rewards"] = verifier_result.rewards
            trajectory.final_metrics.extra = extra

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "trajectory.json"
        path.write_text(
            json.dumps(trajectory.to_json_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
