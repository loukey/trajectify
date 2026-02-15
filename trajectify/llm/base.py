"""Abstract LLM interface and response types."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from trajectify.models.usage import UsageInfo


class ContextLengthExceededError(Exception):
    """Raised when the prompt + history exceeds the model's context window."""


class OutputLengthExceededError(Exception):
    """Raised when the model's output is truncated due to max-output-tokens."""

    def __init__(self, message: str, truncated_response: str = ""):
        super().__init__(message)
        self.truncated_response = truncated_response


class LLMResponse(BaseModel):
    """Value returned by :meth:`BaseLLM.call`."""

    content: str = ""
    reasoning_content: str | None = None
    usage: UsageInfo | None = None

    # rollout-detail fields (only populated when collect_rollout_details=True)
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None


class BaseLLM(ABC):
    """Minimal interface that LLM backends must implement."""

    @abstractmethod
    async def call(
        self,
        prompt: str,
        message_history: list[dict] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a prompt (with optional history) and return a response."""

    @abstractmethod
    def get_context_limit(self) -> int:
        """Maximum input tokens the model supports."""

    @abstractmethod
    def get_output_limit(self) -> int | None:
        """Maximum output tokens (``None`` if unlimited / unknown)."""
