"""Tests for trajectory exporters."""

import json
import tempfile
from pathlib import Path

from trajectify.exporters.atif import AtifExporter
from trajectify.exporters.rollout import RolloutExporter
from trajectify.exporters.sft import SftExporter
from trajectify.models.result import AgentResult, VerifierResult
from trajectify.models.trajectory import AgentInfo, TrajectoryRecorder
from trajectify.models.usage import UsageInfo


def _make_recorder() -> TrajectoryRecorder:
    recorder = TrajectoryRecorder(agent_info=AgentInfo(name="test", model_name="m1"))
    recorder.add_user_step("Do the task")
    recorder.add_agent_step(
        message="Analysis: looks good\nPlan: run command",
        model_name="m1",
        usage=UsageInfo(prompt_tokens=100, completion_tokens=50, cost_usd=0.01),
    )
    recorder.add_user_step("Terminal output:\n```\nfoo\n```")
    recorder.add_agent_step(
        message="Analysis: done\nPlan: finish",
        model_name="m1",
        usage=UsageInfo(prompt_tokens=150, completion_tokens=60, cost_usd=0.02),
    )
    return recorder


def test_atif_exporter():
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = _make_recorder()
        verifier = VerifierResult(rewards={"reward": 1.0})

        exporter = AtifExporter()
        path = exporter.export(
            recorder, Path(tmpdir),
            run_name="test-run",
            verifier_result=verifier,
        )

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["schema_version"] == "ATIF-v1.2"
        assert len(data["steps"]) == 4
        assert data["final_metrics"]["extra"]["rewards"] == {"reward": 1.0}


def test_sft_exporter():
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = _make_recorder()
        verifier = VerifierResult(rewards={"reward": 0.5})

        exporter = SftExporter()
        path = exporter.export(
            recorder, Path(tmpdir),
            run_name="test-run",
            verifier_result=verifier,
        )

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["instruction"] == "Do the task"
        assert len(data["conversations"]) == 4
        assert data["reward"] == 0.5
        assert data["metadata"]["agent"] == "test"


def test_rollout_exporter():
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = _make_recorder()
        agent_result = AgentResult(
            n_input_tokens=250,
            n_output_tokens=110,
            cost_usd=0.03,
            rollout_details=[{
                "prompt_token_ids": [[1, 2, 3]],
                "completion_token_ids": [[4, 5]],
                "logprobs": [[-0.1, -0.2]],
            }],
        )
        verifier = VerifierResult(rewards={"reward": 1.0})

        exporter = RolloutExporter()
        path = exporter.export(
            recorder, Path(tmpdir),
            run_name="test-run",
            agent_result=agent_result,
            verifier_result=verifier,
        )

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["reward"] == 1.0
        assert len(data["rollout_details"]) == 1
        assert data["metadata"]["n_input_tokens"] == 250


def test_exporter_format_names():
    assert AtifExporter.format_name() == "atif"
    assert SftExporter.format_name() == "sft"
    assert RolloutExporter.format_name() == "rollout"
