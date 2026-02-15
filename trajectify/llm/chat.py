"""Chat — manages multi-turn conversation state and rollout accumulation."""

from __future__ import annotations

from trajectify.llm.base import BaseLLM, LLMResponse
from trajectify.models.rollout import RolloutDetail


class Chat:
    """Stateful wrapper around a :class:`BaseLLM`.

    Maintains the message history, accumulates token counts, and collects
    rollout details (token IDs + logprobs) across turns.
    """

    def __init__(self, model: BaseLLM, system_prompt: str | None = None):
        self._model = model

        self._messages: list[dict] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cache_tokens: int = 0
        self._total_cost: float = 0.0

        self._prompt_ids_list: list[list[int]] = []
        self._completion_ids_list: list[list[int]] = []
        self._logprobs_list: list[list[float]] = []

    @property
    def llm(self) -> BaseLLM:
        """The underlying LLM instance (read-only)."""
        return self._model

    @property
    def messages(self) -> list[dict]:
        return self._messages

    @messages.setter
    def messages(self, value: list[dict]) -> None:
        self._messages = value

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_cache_tokens(self) -> int:
        return self._total_cache_tokens

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def rollout_details(self) -> list[RolloutDetail]:
        if (
            not self._prompt_ids_list
            and not self._completion_ids_list
            and not self._logprobs_list
        ):
            return []

        detail: RolloutDetail = {}
        if self._prompt_ids_list:
            detail["prompt_token_ids"] = self._prompt_ids_list
        if self._completion_ids_list:
            detail["completion_token_ids"] = self._completion_ids_list
        if self._logprobs_list:
            detail["logprobs"] = self._logprobs_list
        return [detail]

    async def chat(self, prompt: str) -> LLMResponse:
        """Send *prompt* as the next user message and return the response."""
        response = await self._model.call(
            prompt=prompt,
            message_history=self._messages,
        )

        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens
            self._total_output_tokens += response.usage.completion_tokens
            self._total_cache_tokens += response.usage.cache_tokens
            self._total_cost += response.usage.cost_usd

        self._accumulate_rollout(response)

        self._messages.append({"role": "user", "content": prompt})
        self._messages.append({"role": "assistant", "content": response.content})

        return response

    def _accumulate_rollout(self, resp: LLMResponse) -> None:
        if resp.prompt_token_ids is not None:
            self._prompt_ids_list.append(resp.prompt_token_ids)
        if resp.completion_token_ids is not None:
            self._completion_ids_list.append(resp.completion_token_ids)
        if resp.logprobs is not None:
            self._logprobs_list.append(resp.logprobs)
