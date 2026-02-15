"""LiteLLM-based LLM provider — supports all major LLM APIs."""

from __future__ import annotations

import logging

import litellm
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trajectify.llm.base import (
    BaseLLM,
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from trajectify.log import logger as global_logger
from trajectify.models.usage import UsageInfo


def _is_context_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keywords = ["context length", "context_length", "token limit", "max.*token"]
    return any(k in msg for k in keywords)


class LiteLLMProvider(BaseLLM):
    """Wraps :pypi:`litellm` to provide a unified LLM calling interface."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        api_base: str | None = None,
        collect_rollout_details: bool = False,
        logger: logging.Logger | None = None,
    ):
        self._model = model_name
        self._temperature = temperature
        self._api_base = api_base
        self._collect_rollout = collect_rollout_details
        self._logger = (logger or global_logger).getChild("LiteLLM")

        litellm.drop_params = True

        try:
            info = litellm.get_model_info(self._model)
            self._context_limit: int = info.get("max_input_tokens", 128_000)
            self._output_limit: int | None = info.get("max_output_tokens")
        except Exception:
            self._context_limit = 128_000
            self._output_limit = None

    def get_context_limit(self) -> int:
        return self._context_limit

    def get_output_limit(self) -> int | None:
        return self._output_limit

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(litellm.RateLimitError),
    )
    async def call(
        self,
        prompt: str,
        message_history: list[dict] | None = None,
        **kwargs,
    ) -> LLMResponse:
        messages = list(message_history or [])
        messages.append({"role": "user", "content": prompt})

        completion_kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._api_base:
            completion_kwargs["api_base"] = self._api_base

        if self._collect_rollout:
            completion_kwargs["logprobs"] = True
            completion_kwargs.setdefault("extra_body", {})
            completion_kwargs["extra_body"]["return_token_ids"] = True

        try:
            response = await litellm.acompletion(**completion_kwargs)
        except Exception as exc:
            if _is_context_error(exc):
                raise ContextLengthExceededError(str(exc)) from exc
            raise

        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None)

        usage = self._extract_usage(response)

        if choice.finish_reason == "length":
            raise OutputLengthExceededError(
                "Output truncated (finish_reason=length)",
                truncated_response=content,
            )

        prompt_ids, completion_ids, lp = None, None, None
        if self._collect_rollout:
            prompt_ids, completion_ids = self._extract_token_ids(response)
            lp = self._extract_logprobs(response)

        return LLMResponse(
            content=content,
            reasoning_content=reasoning,
            usage=usage,
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
            logprobs=lp,
        )

    @staticmethod
    def _extract_usage(response) -> UsageInfo:
        u = response.usage
        if u is None:
            return UsageInfo()
        try:
            cost = litellm.completion_cost(response) or 0.0
        except Exception:
            cost = 0.0
        return UsageInfo(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            cache_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cost_usd=cost,
        )

    @staticmethod
    def _extract_token_ids(response) -> tuple[list[int] | None, list[int] | None]:
        choice = response.choices[0]
        prompt_ids = None
        completion_ids = None

        if hasattr(response, "prompt_token_ids"):
            prompt_ids = response.prompt_token_ids
        if hasattr(choice, "token_ids"):
            completion_ids = choice.token_ids

        return prompt_ids, completion_ids

    @staticmethod
    def _extract_logprobs(response) -> list[float] | None:
        choice = response.choices[0]
        logprobs_obj = getattr(choice, "logprobs", None)
        if logprobs_obj is None:
            return None

        content_lp = getattr(logprobs_obj, "content", None)
        if not content_lp:
            return None

        return [tok.logprob for tok in content_lp if hasattr(tok, "logprob")]
