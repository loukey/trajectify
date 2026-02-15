"""Tests for TrajectoryRecorder."""

from trajectify.models.trajectory import AgentInfo, TrajectoryRecorder
from trajectify.models.usage import UsageInfo


def test_recorder_add_steps():
    recorder = TrajectoryRecorder(agent_info=AgentInfo(name="test", model_name="m1"))

    recorder.add_user_step("hello")
    recorder.add_agent_step(
        message="response",
        model_name="m1",
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5),
    )
    recorder.add_system_step("summary")

    assert len(recorder.steps) == 3
    assert recorder.steps[0].source == "user"
    assert recorder.steps[0].step_id == 1
    assert recorder.steps[1].source == "agent"
    assert recorder.steps[1].step_id == 2
    assert recorder.steps[2].source == "system"


def test_recorder_build_trajectory():
    recorder = TrajectoryRecorder(agent_info=AgentInfo(name="test"))

    recorder.add_agent_step(
        message="step1",
        usage=UsageInfo(prompt_tokens=100, completion_tokens=50, cost_usd=0.01),
    )
    recorder.add_agent_step(
        message="step2",
        usage=UsageInfo(prompt_tokens=200, completion_tokens=100, cost_usd=0.02),
    )

    traj = recorder.build_trajectory(session_id="sess-1")
    assert traj.session_id == "sess-1"
    assert traj.schema_version == "ATIF-v1.2"
    assert traj.agent.name == "test"
    assert len(traj.steps) == 2
    assert traj.final_metrics.total_prompt_tokens == 300
    assert traj.final_metrics.total_completion_tokens == 150
    assert traj.final_metrics.total_cost_usd == 0.03
    assert traj.final_metrics.total_steps == 2


def test_recorder_to_json_dict():
    recorder = TrajectoryRecorder(agent_info=AgentInfo(name="x"))
    recorder.add_user_step("hi")

    traj = recorder.build_trajectory()
    data = traj.to_json_dict()

    assert data["schema_version"] == "ATIF-v1.2"
    assert data["agent"]["name"] == "x"
    assert len(data["steps"]) == 1
    # exclude_none should remove None fields
    assert "reasoning_content" not in data["steps"][0]
