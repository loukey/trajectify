"""Terminus agent — LLM-driven terminal interaction loop."""

from __future__ import annotations

import time
from pathlib import Path

from trajectify.agents.base import BaseAgent
from trajectify.agents.registry import register_agent
from trajectify.agents.terminus.parser import ParseError, parse_response
from trajectify.agents.terminus.summarizer import Summarizer
from trajectify.agents.terminus.tmux import TmuxSession
from trajectify.environments.base import BaseEnvironment
from trajectify.llm.base import ContextLengthExceededError, LLMResponse, OutputLengthExceededError
from trajectify.llm.chat import Chat
from trajectify.llm.litellm_provider import LiteLLMProvider
from trajectify.models.result import AgentResult
from trajectify.models.trajectory import (
    Observation,
    ObservationResult,
    ToolCall,
    TrajectoryRecorder,
)
from trajectify.models.usage import UsageInfo

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@register_agent
class TerminusAgent(BaseAgent):
    """An agent that uses an LLM to drive a tmux terminal session."""

    def __init__(
        self,
        *,
        temperature: float = 0.7,
        collect_rollout_details: bool = False,
        enable_summarize: bool = True,
        max_turns: int = 1_000_000,
        api_base: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._temperature = temperature
        self._collect_rollout = collect_rollout_details
        self._enable_summarize = enable_summarize
        self._max_turns = max_turns
        self._api_base = api_base

        self._system_template = (_PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")
        self._timeout_prompt = (_PROMPTS_DIR / "timeout.txt").read_text(encoding="utf-8")

        self._chat: Chat | None = None
        self._tmux: TmuxSession | None = None
        self._task_complete_pending: bool = False
        self._api_times_ms: list[float] = []

    @staticmethod
    def agent_name() -> str:
        return "terminus"

    async def setup(self, env: BaseEnvironment) -> None:
        self._tmux = TmuxSession(env=env, logger=self.logger)
        await self._tmux.start()

    async def run(
        self,
        instruction: str,
        env: BaseEnvironment,
        recorder: TrajectoryRecorder,
    ) -> AgentResult:
        llm = LiteLLMProvider(
            model_name=self.model_name or "anthropic/claude-sonnet-4-20250514",
            temperature=self._temperature,
            api_base=self._api_base,
            collect_rollout_details=self._collect_rollout,
            logger=self.logger,
        )
        system_prompt = self._system_template.format(instruction=instruction)
        self._chat = Chat(model=llm, system_prompt=system_prompt)

        summarizer = Summarizer(
            chat=self._chat,
            recorder=recorder,
            logger=self.logger,
        ) if self._enable_summarize else None

        initial_screen = await self._tmux.capture_pane()
        user_prompt = f"Terminal output:\n```\n{initial_screen}\n```"

        recorder.add_user_step(user_prompt)

        await self._agent_loop(user_prompt, recorder, summarizer)

        return self._build_result(recorder, summarizer)

    async def _agent_loop(
        self,
        initial_prompt: str,
        recorder: TrajectoryRecorder,
        summarizer: Summarizer | None,
    ) -> None:
        prompt = initial_prompt

        for episode in range(self._max_turns):
            if summarizer:
                prompt = await summarizer.maybe_summarize(prompt)

            try:
                t0 = time.monotonic()
                response = await self._chat.chat(prompt)
                self._api_times_ms.append((time.monotonic() - t0) * 1000)
            except ContextLengthExceededError:
                self.logger.warning("Context length exceeded; attempting summarization")
                if summarizer:
                    prompt = await summarizer.force_summarize(prompt)
                    continue
                else:
                    self.logger.error("Context exceeded and summarization disabled")
                    break
            except OutputLengthExceededError as exc:
                self.logger.warning("Output truncated; using partial response")
                response = LLMResponse(content=exc.truncated_response)

            try:
                parsed = parse_response(response.content)
            except ParseError as exc:
                self.logger.warning("Parse error: %s", exc)
                prompt = (
                    f"Your previous response was not valid JSON. Error: {exc}\n"
                    "Please respond with a valid JSON object."
                )
                self._record_error_step(recorder, response, str(exc))
                continue

            combined_output = await self._execute_commands(parsed.commands)

            self._record_parsed_step(recorder, parsed, response, combined_output)

            if parsed.is_task_complete:
                if self._task_complete_pending:
                    self.logger.info("Task marked complete (confirmed)")
                    break
                else:
                    self._task_complete_pending = True
                    prompt = (
                        f"Terminal output:\n```\n{combined_output}\n```\n\n"
                        "You indicated the task is complete. Please confirm by "
                        "setting is_task_complete to true again, or continue working."
                    )
                    continue
            else:
                self._task_complete_pending = False

            prompt = f"Terminal output:\n```\n{combined_output}\n```"

    async def _execute_commands(self, commands) -> str:
        last_output = ""
        for cmd in commands:
            last_output = await self._tmux.send_keys(
                keys=cmd.keystrokes,
                wait_sec=cmd.duration,
            )
        if not commands:
            last_output = await self._tmux.capture_pane()
        return last_output

    def _record_error_step(
        self,
        recorder: TrajectoryRecorder,
        response: LLMResponse,
        error: str,
    ) -> None:
        usage = response.usage if response else None
        recorder.add_agent_step(
            message=response.content[:500] if response.content else "",
            model_name=self.model_name,
            usage=usage,
            extra={"error": error},
        )

    def _record_parsed_step(
        self,
        recorder: TrajectoryRecorder,
        parsed,
        response: LLMResponse,
        output: str,
    ) -> None:
        step_id = len(recorder.steps) + 1
        tool_calls = []
        for i, cmd in enumerate(parsed.commands):
            tool_calls.append(ToolCall(
                tool_call_id=f"cmd-{step_id}-{i}",
                function_name="bash_command",
                arguments={"keystrokes": cmd.keystrokes, "duration": cmd.duration},
            ))

        observation = Observation(results=[
            ObservationResult(
                source_call_id=tc.tool_call_id,
                content=output if i == len(tool_calls) - 1 else None,
            )
            for i, tc in enumerate(tool_calls)
        ]) if tool_calls else None

        recorder.add_agent_step(
            message=f"Analysis: {parsed.analysis}\nPlan: {parsed.plan}",
            model_name=self.model_name,
            tool_calls=tool_calls or None,
            observation=observation,
            usage=response.usage,
            extra={"is_task_complete": parsed.is_task_complete},
        )

    def _build_result(
        self,
        recorder: TrajectoryRecorder,
        summarizer: Summarizer | None,
    ) -> AgentResult:
        result = AgentResult()
        if self._chat:
            result.n_input_tokens = self._chat.total_input_tokens
            result.n_output_tokens = self._chat.total_output_tokens
            result.n_cache_tokens = self._chat.total_cache_tokens
            result.cost_usd = self._chat.total_cost

            rollout = self._chat.rollout_details
            if rollout:
                result.rollout_details = rollout

        result.metadata = {
            "n_episodes": len(recorder.steps),
            "summarization_count": summarizer.count if summarizer else 0,
            "api_request_times_msec": self._api_times_ms,
        }
        return result
