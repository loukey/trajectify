"""ATIF trajectory data models and TrajectoryRecorder."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from trajectify.models.usage import UsageInfo


class ToolCall(BaseModel):
    """A single tool/function call made by the agent."""

    tool_call_id: str = ""
    function_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)


class ObservationResult(BaseModel):
    """Result of a single tool call execution."""

    source_call_id: str = ""
    content: str | None = None
    subagent_trajectory_ref: str | None = None


class Observation(BaseModel):
    """Observation after executing tool calls."""

    results: list[ObservationResult] = Field(default_factory=list)


class Step(BaseModel):
    """A single step in the agent trajectory."""

    step_id: int
    timestamp: str | None = None
    source: str = "agent"  # "agent" | "user" | "system"
    message: str = ""
    reasoning_content: str | None = None
    model_name: str | None = None
    tool_calls: list[ToolCall] | None = None
    observation: Observation | None = None
    usage: UsageInfo | None = None
    extra: dict[str, Any] | None = None


class AgentInfo(BaseModel):
    """Metadata about the agent that produced the trajectory."""

    name: str
    version: str = "unknown"
    model_name: str | None = None
    extra: dict[str, Any] | None = None


class FinalMetrics(BaseModel):
    """Aggregated metrics for an entire trajectory."""

    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cache_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int = 0
    extra: dict[str, Any] | None = None


class Trajectory(BaseModel):
    """A complete ATIF trajectory."""

    schema_version: str = "ATIF-v1.2"
    session_id: str = ""
    agent: AgentInfo
    steps: list[Step] = Field(default_factory=list)
    final_metrics: FinalMetrics | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class TrajectoryRecorder:
    """Records trajectory steps during an agent run.

    Created by the Runner and injected into agents. Agents call methods like
    ``add_user_step()``, ``add_agent_step()``, etc. After the run, the Runner
    passes the recorder to exporters.
    """

    def __init__(self, agent_info: AgentInfo) -> None:
        self._agent_info = agent_info
        self._steps: list[Step] = []
        self._step_id: int = 0

    @property
    def steps(self) -> list[Step]:
        return self._steps

    @property
    def agent_info(self) -> AgentInfo:
        return self._agent_info

    def _next_step_id(self) -> int:
        self._step_id += 1
        return self._step_id

    def add_user_step(self, message: str) -> None:
        self._steps.append(Step(
            step_id=self._next_step_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="user",
            message=message,
        ))

    def add_system_step(self, message: str) -> None:
        self._steps.append(Step(
            step_id=self._next_step_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="system",
            message=message,
        ))

    def add_agent_step(
        self,
        message: str,
        *,
        model_name: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        observation: Observation | None = None,
        usage: UsageInfo | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._steps.append(Step(
            step_id=self._next_step_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="agent",
            message=message,
            reasoning_content=reasoning_content,
            model_name=model_name,
            tool_calls=tool_calls,
            observation=observation,
            usage=usage,
            extra=extra,
        ))

    def build_trajectory(self, session_id: str = "") -> Trajectory:
        """Assemble a complete Trajectory from recorded steps."""
        prompt_total = sum(
            s.usage.prompt_tokens for s in self._steps
            if s.usage and s.usage.prompt_tokens
        )
        comp_total = sum(
            s.usage.completion_tokens for s in self._steps
            if s.usage and s.usage.completion_tokens
        )
        cache_total = sum(
            s.usage.cache_tokens for s in self._steps
            if s.usage and s.usage.cache_tokens
        )
        cost_total = sum(
            s.usage.cost_usd for s in self._steps
            if s.usage and s.usage.cost_usd
        )

        return Trajectory(
            session_id=session_id,
            agent=self._agent_info,
            steps=self._steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=prompt_total or None,
                total_completion_tokens=comp_total or None,
                total_cache_tokens=cache_total or None,
                total_cost_usd=cost_total or None,
                total_steps=len(self._steps),
            ),
        )
