"""SFT (Supervised Fine-Tuning) format exporter."""

from __future__ import annotations

import json
from pathlib import Path

from trajectify.exporters.base import BaseExporter
from trajectify.models.result import AgentResult, VerifierResult
from trajectify.models.trajectory import TrajectoryRecorder


class SftExporter(BaseExporter):
    """Exports trajectories as instruction-trajectory pairs for SFT training.

    Output format (JSON):
    {
        "instruction": "<first user message>",
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ],
        "reward": <float or null>,
        "metadata": {...}
    }
    """

    @staticmethod
    def format_name() -> str:
        return "sft"

    def export(
        self,
        recorder: TrajectoryRecorder,
        output_dir: Path,
        *,
        run_name: str = "",
        agent_result: AgentResult | None = None,
        verifier_result: VerifierResult | None = None,
    ) -> Path:
        steps = recorder.steps
        conversations: list[dict[str, str]] = []
        instruction = ""

        for step in steps:
            if step.source == "user":
                if not instruction:
                    instruction = step.message
                conversations.append({"role": "user", "content": step.message})
            elif step.source == "agent":
                conversations.append({"role": "assistant", "content": step.message})

        # Extract primary reward
        reward: float | None = None
        if verifier_result and verifier_result.rewards:
            values = list(verifier_result.rewards.values())
            reward = float(values[0]) if values else None

        data = {
            "instruction": instruction,
            "conversations": conversations,
            "reward": reward,
            "metadata": {
                "run_name": run_name,
                "agent": recorder.agent_info.name,
                "model": recorder.agent_info.model_name,
            },
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "sft.json"
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
