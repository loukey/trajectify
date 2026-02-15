"""Run result and related models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from trajectify.models.rollout import RolloutDetail


class AgentResult(BaseModel):
    """Result returned by an agent after execution."""

    n_input_tokens: int = 0
    n_output_tokens: int = 0
    n_cache_tokens: int = 0
    cost_usd: float | None = None
    rollout_details: list[RolloutDetail] | None = None
    metadata: dict[str, Any] | None = None


class VerifierResult(BaseModel):
    """Result returned by the verifier."""

    rewards: dict[str, float | int]


class TimingInfo(BaseModel):
    """Start/end timestamps for a phase."""

    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    @property
    def duration_sec(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


class ExceptionInfo(BaseModel):
    """Serialisable exception information."""

    exception_type: str
    message: str

    @classmethod
    def from_exception(cls, exc: BaseException) -> ExceptionInfo:
        return cls(
            exception_type=type(exc).__name__,
            message=str(exc),
        )


class RunResult(BaseModel):
    """Complete result of a single run (was TrialResult)."""

    run_name: str
    task_name: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    environment_setup: TimingInfo | None = None
    agent_setup: TimingInfo | None = None
    agent_execution: TimingInfo | None = None
    verifier_timing: TimingInfo | None = None

    agent_result: AgentResult | None = None
    verifier_result: VerifierResult | None = None
    exception_info: ExceptionInfo | None = None
