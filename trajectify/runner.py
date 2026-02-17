"""Runner — the core execution unit (was Trial in agent-hub)."""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timezone
from pathlib import Path

from trajectify.agents.registry import create_agent
from trajectify.config.models import RunConfig
from trajectify.environments.docker import DockerEnvironment
from trajectify.exporters.atif import AtifExporter
from trajectify.exporters.base import BaseExporter
from trajectify.exporters.rollout import RolloutExporter
from trajectify.exporters.sft import SftExporter
from trajectify.log import logger
from trajectify.models.result import (
    AgentResult,
    ExceptionInfo,
    RunResult,
    TimingInfo,
)
from trajectify.models.trajectory import AgentInfo, TrajectoryRecorder
from trajectify.task_loaders.base import BaseTaskLoader
from trajectify.task_loaders.terminal_bench import TerminalBenchLoader
from trajectify.task_loaders.terminal_bench_v1 import TerminalBenchV1Loader
from trajectify.verifier import Verifier


class AgentTimeoutError(asyncio.TimeoutError):
    pass


class VerifierTimeoutError(asyncio.TimeoutError):
    pass


_EXPORTER_MAP: dict[str, type[BaseExporter]] = {
    "atif": AtifExporter,
    "sft": SftExporter,
    "rollout": RolloutExporter,
}


class Runner:
    """Runs a single agent attempt on a single task.

    Integrates the TrajectoryRecorder and exporter pipeline:
    1. Load task, create agent and environment.
    2. Create a TrajectoryRecorder and inject it into the agent.
    3. After the agent finishes, pass the recorder to each configured exporter.
    """

    def __init__(self, config: RunConfig):
        self.config = config

        self._run_dir = Path(config.output_dir) / config.run_name
        self._agent_dir = self._run_dir / "agent"
        self._verifier_dir = self._run_dir / "verifier"
        self._agent_dir.mkdir(parents=True, exist_ok=True)
        self._verifier_dir.mkdir(parents=True, exist_ok=True)

        # Load the task
        loader = self._detect_loader(config.task_path)
        self._task = loader.load(config.task_path)

        # Create the agent
        agent_kwargs = {
            "temperature": config.agent.temperature,
            "collect_rollout_details": config.agent.collect_rollout_details,
            "enable_summarize": config.agent.enable_summarize,
            "max_turns": config.agent.max_turns,
            **config.agent.kwargs,
        }
        self._agent = create_agent(
            name=config.agent.name,
            logs_dir=self._agent_dir,
            model_name=config.agent.model,
            **agent_kwargs,
        )

        # Create the environment
        self._env = DockerEnvironment(
            environment_dir=self._task.paths.environment_dir,
            environment_name=self._task.name,
            session_id=config.run_name,
            host_agent_dir=self._agent_dir,
            host_verifier_dir=self._verifier_dir,
            cpus=self._task.config.environment.cpus,
            memory_mb=self._task.config.environment.memory_mb,
            allow_internet=self._task.config.environment.allow_internet,
        )

        # Timeouts
        self._agent_timeout = (
            config.agent.override_timeout_sec
            or self._task.config.agent.timeout_sec
        )
        self._verifier_timeout = (
            config.verifier.override_timeout_sec
            or self._task.config.verifier.timeout_sec
        )
        self._build_timeout = self._task.config.environment.build_timeout_sec

        # Build exporters
        self._exporters: list[BaseExporter] = []
        for fmt in config.exporter.formats:
            cls = _EXPORTER_MAP.get(fmt)
            if cls:
                self._exporters.append(cls())

        self._logger = logger.getChild(f"Runner.{config.run_name}")
        self._result: RunResult | None = None

    @staticmethod
    def _detect_loader(task_path: Path) -> BaseTaskLoader:
        task_dir = Path(task_path)
        if (task_dir / "task.toml").exists():
            return TerminalBenchLoader()
        elif (task_dir / "task.yaml").exists():
            return TerminalBenchV1Loader()
        else:
            raise FileNotFoundError(
                f"No task.toml or task.yaml found in {task_dir}"
            )

    async def run(self) -> RunResult:
        self._result = RunResult(
            run_name=self.config.run_name,
            task_name=self._task.name,
        )

        # Persist config
        config_path = self._run_dir / "config.json"
        config_path.write_text(
            self.config.model_dump_json(indent=2), encoding="utf-8",
        )

        # Create recorder
        recorder = TrajectoryRecorder(
            agent_info=AgentInfo(
                name=self.config.agent.name,
                model_name=self.config.agent.model,
            ),
        )

        agent_result: AgentResult | None = None

        try:
            await self._phase_start_environment()
            await self._phase_setup_agent()

            try:
                agent_result = await self._phase_execute_agent(recorder)
            except AgentTimeoutError as exc:
                self._result.exception_info = ExceptionInfo.from_exception(exc)
                self._logger.warning("Agent timed out after %ss", self._agent_timeout)

            if not self.config.verifier.disable:
                await self._phase_verify()

        except Exception as exc:
            self._logger.error("Run failed: %s", exc)
            if self._result.exception_info is None:
                self._result.exception_info = ExceptionInfo.from_exception(exc)
                (self._run_dir / "exception.txt").write_text(
                    traceback.format_exc(), encoding="utf-8",
                )
        finally:
            # Run exporters before cleanup
            self._run_exporters(recorder, agent_result)
            await self._cleanup()

        return self._result

    def _run_exporters(
        self,
        recorder: TrajectoryRecorder,
        agent_result: AgentResult | None,
    ) -> None:
        export_dir = self.config.exporter.output_dir or self._agent_dir
        for exporter in self._exporters:
            try:
                path = exporter.export(
                    recorder,
                    export_dir,
                    run_name=self.config.run_name,
                    agent_result=agent_result,
                    verifier_result=self._result.verifier_result if self._result else None,
                )
                self._logger.info(
                    "Exported %s to %s", exporter.format_name(), path,
                )
            except Exception as exc:
                self._logger.warning(
                    "Exporter %s failed: %s", exporter.format_name(), exc,
                )

    async def _phase_start_environment(self) -> None:
        self._result.environment_setup = TimingInfo()
        try:
            await asyncio.wait_for(
                self._env.start(force_build=self.config.environment.force_build),
                timeout=self._build_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Environment build timed out after {self._build_timeout}s"
            ) from exc
        finally:
            self._result.environment_setup.finished_at = datetime.now(timezone.utc)

    async def _phase_setup_agent(self) -> None:
        self._result.agent_setup = TimingInfo()
        try:
            await asyncio.wait_for(self._agent.setup(self._env), timeout=360)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Agent setup timed out after 360s") from exc
        finally:
            self._result.agent_setup.finished_at = datetime.now(timezone.utc)

    async def _phase_execute_agent(self, recorder: TrajectoryRecorder) -> AgentResult:
        self._result.agent_execution = TimingInfo()
        try:
            agent_result = await asyncio.wait_for(
                self._agent.run(
                    instruction=self._task.instruction,
                    env=self._env,
                    recorder=recorder,
                ),
                timeout=self._agent_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise AgentTimeoutError(
                f"Agent execution timed out after {self._agent_timeout}s"
            ) from exc
        finally:
            self._result.agent_execution.finished_at = datetime.now(timezone.utc)

        self._result.agent_result = agent_result
        return agent_result

    async def _phase_verify(self) -> None:
        self._result.verifier_timing = TimingInfo()
        try:
            verifier = Verifier(
                task=self._task,
                env=self._env,
                host_verifier_dir=self._verifier_dir,
            )
            self._result.verifier_result = await asyncio.wait_for(
                verifier.verify(),
                timeout=self._verifier_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise VerifierTimeoutError(
                f"Verifier timed out after {self._verifier_timeout}s"
            ) from exc
        finally:
            self._result.verifier_timing.finished_at = datetime.now(timezone.utc)

    async def _cleanup(self) -> None:
        try:
            await self._env.stop(delete=self.config.environment.delete_after)
        except Exception as exc:
            self._logger.warning("Environment cleanup failed: %s", exc)

        self._result.finished_at = datetime.now(timezone.utc)

        result_path = self._run_dir / "result.json"
        result_path.write_text(
            self._result.model_dump_json(indent=2), encoding="utf-8",
        )
