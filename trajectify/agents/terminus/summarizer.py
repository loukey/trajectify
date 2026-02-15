"""Conversation summarizer — extracted from TerminusAgent."""

from __future__ import annotations

import logging

from trajectify.llm.base import BaseLLM
from trajectify.llm.chat import Chat
from trajectify.log import logger as global_logger
from trajectify.models.trajectory import TrajectoryRecorder


class Summarizer:
    """Handles conversation summarization when context gets full.

    Compresses older messages via a hybrid summarize-and-buffer approach:
    keeps the system prompt, summarizes older messages, and retains the
    most recent turns in full fidelity.
    """

    def __init__(
        self,
        chat: Chat,
        recorder: TrajectoryRecorder,
        logger: logging.Logger | None = None,
    ):
        self._chat = chat
        self._recorder = recorder
        self._logger = (logger or global_logger).getChild("Summarizer")
        self._count: int = 0

    @property
    def count(self) -> int:
        return self._count

    async def maybe_summarize(self, current_prompt: str) -> str:
        """Check if context is getting full; if so, summarize."""
        llm = self._chat.llm
        ctx_limit = llm.get_context_limit()
        current_tokens = self._chat.total_input_tokens + self._chat.total_output_tokens
        estimated_used = current_tokens * 0.6
        free = ctx_limit - estimated_used

        threshold = 8000
        if free > threshold:
            return current_prompt

        return await self.force_summarize(current_prompt)

    async def force_summarize(self, current_prompt: str) -> str:
        """Compress the conversation using a hybrid summarize-and-buffer approach."""
        self._count += 1
        self._logger.info("Summarizing conversation (count=%d)", self._count)

        messages = self._chat.messages

        system_msg = messages[0] if messages and messages[0]["role"] == "system" else None

        conversation_msgs = [m for m in messages if m["role"] != "system"]
        keep_recent = 6
        if len(conversation_msgs) > keep_recent:
            older_msgs = conversation_msgs[:-keep_recent]
            recent_msgs = conversation_msgs[-keep_recent:]
        else:
            older_msgs = conversation_msgs
            recent_msgs = []

        history_text = "\n".join(
            f"[{m['role']}]: {m['content'][:200]}" for m in older_msgs[-20:]
        )
        summary_prompt = (
            "Summarize the conversation so far in 2-3 paragraphs. "
            "Focus on: what has been accomplished, what the current state is, "
            "and what remains to be done.\n\n"
            f"Recent history:\n{history_text}"
        )
        llm = self._chat.llm
        summary_response = await llm.call(summary_prompt)

        new_messages = []
        if system_msg:
            new_messages.append(system_msg)
        new_messages.append({
            "role": "user",
            "content": (
                "[CONVERSATION SUMMARY]\n"
                f"{summary_response.content}\n"
                "[END SUMMARY]"
            ),
        })
        new_messages.append({
            "role": "assistant",
            "content": (
                "Understood. I have the context from the summary. "
                "Continuing with the task."
            ),
        })

        if recent_msgs and recent_msgs[0]["role"] == "assistant":
            recent_msgs = recent_msgs[1:]
        new_messages.extend(recent_msgs)

        self._chat.messages = new_messages

        self._recorder.add_system_step(f"[Summarization #{self._count}]")

        return current_prompt
