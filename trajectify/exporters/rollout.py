"""RL rollout format exporter."""

from __future__ import annotations

import json
from pathlib import Path

from trajectify.exporters.base import BaseExporter
from trajectify.models.result import AgentResult, VerifierResult
from trajectify.models.trajectory import TrajectoryRecorder


class RolloutExporter(BaseExporter):
    """Exports rollout details (token IDs + logprobs) for RL training.

    Output format (JSON):
    {
        "run_name": "...",
        "reward": <float or null>,
        "rollout_details": [<RolloutDetail>, ...],
        "metadata": {...}
    }
    """

    @staticmethod
    def format_name() -> str:
        return "rollout"

    def export(
        self,
        recorder: TrajectoryRecorder,
        output_dir: Path,
        *,
        run_name: str = "",
        agent_result: AgentResult | None = None,
        verifier_result: VerifierResult | None = None,
    ) -> Path:
        reward: float | None = None
        if verifier_result and verifier_result.rewards:
            values = list(verifier_result.rewards.values())
            reward = float(values[0]) if values else None

        rollout_details = []
        if agent_result and agent_result.rollout_details:
            rollout_details = [dict(rd) for rd in agent_result.rollout_details]

        data = {
            "run_name": run_name,
            "reward": reward,
            "rollout_details": rollout_details,
            "metadata": {
                "agent": recorder.agent_info.name,
                "model": recorder.agent_info.model_name,
                "n_input_tokens": agent_result.n_input_tokens if agent_result else 0,
                "n_output_tokens": agent_result.n_output_tokens if agent_result else 0,
                "cost_usd": agent_result.cost_usd if agent_result else None,
            },
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "rollout.json"
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
