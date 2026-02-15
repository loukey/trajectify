"""Tests for the Chat class."""

import pytest

from trajectify.llm.base import BaseLLM, LLMResponse
from trajectify.llm.chat import Chat
from trajectify.models.usage import UsageInfo


class MockLLM(BaseLLM):
    """A mock LLM that returns a fixed response."""

    def __init__(self, content: str = "hello", usage: UsageInfo | None = None):
        self._content = content
        self._usage = usage or UsageInfo(prompt_tokens=10, completion_tokens=5, cost_usd=0.001)

    async def call(self, prompt, message_history=None, **kwargs):
        return LLMResponse(content=self._content, usage=self._usage)

    def get_context_limit(self) -> int:
        return 128_000

    def get_output_limit(self) -> int | None:
        return 4096


@pytest.mark.asyncio
async def test_chat_accumulates_tokens():
    llm = MockLLM(usage=UsageInfo(prompt_tokens=100, completion_tokens=50, cost_usd=0.01))
    chat = Chat(model=llm, system_prompt="You are helpful.")

    await chat.chat("Hello")
    assert chat.total_input_tokens == 100
    assert chat.total_output_tokens == 50
    assert chat.total_cost == pytest.approx(0.01)

    await chat.chat("World")
    assert chat.total_input_tokens == 200
    assert chat.total_output_tokens == 100
    assert chat.total_cost == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_chat_maintains_history():
    chat = Chat(model=MockLLM(), system_prompt="sys")

    await chat.chat("first")
    assert len(chat.messages) == 3  # system + user + assistant
    assert chat.messages[0] == {"role": "system", "content": "sys"}
    assert chat.messages[1] == {"role": "user", "content": "first"}
    assert chat.messages[2] == {"role": "assistant", "content": "hello"}


@pytest.mark.asyncio
async def test_chat_llm_property():
    llm = MockLLM()
    chat = Chat(model=llm)
    assert chat.llm is llm


@pytest.mark.asyncio
async def test_chat_rollout_empty_by_default():
    chat = Chat(model=MockLLM())
    await chat.chat("test")
    assert chat.rollout_details == []
