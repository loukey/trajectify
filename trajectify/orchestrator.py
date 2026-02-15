"""Orchestrator — run multiple tasks in parallel with asyncio."""

from __future__ import annotations

import asyncio

from trajectify.config.models import RunConfig
from trajectify.log import logger
from trajectify.models.result import RunResult
from trajectify.runner import Runner


class Orchestrator:
    """Execute a batch of :class:`Runner` instances with bounded concurrency."""

    def __init__(
        self,
        configs: list[RunConfig],
        n_concurrent: int = 1,
    ):
        self._configs = configs
        self._n_concurrent = max(1, n_concurrent)
        self._results: list[RunResult] = []
        self._logger = logger.getChild("Orchestrator")

    async def run(self) -> list[RunResult]:
        sem = asyncio.Semaphore(self._n_concurrent)
        total = len(self._configs)

        self._logger.info(
            "Starting %d run(s) with concurrency=%d", total, self._n_concurrent,
        )

        async def _run_one(cfg: RunConfig, idx: int) -> RunResult:
            async with sem:
                self._logger.info(
                    "[%d/%d] Running %s", idx + 1, total, cfg.run_name,
                )
                runner = Runner(cfg)
                result = await runner.run()

                reward_str = "N/A"
                if result.verifier_result and result.verifier_result.rewards:
                    reward_str = str(result.verifier_result.rewards)

                status = "OK" if result.exception_info is None else result.exception_info.exception_type
                self._logger.info(
                    "[%d/%d] %s finished — status=%s, reward=%s",
                    idx + 1, total, cfg.run_name, status, reward_str,
                )
                return result

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(_run_one(cfg, i))
                for i, cfg in enumerate(self._configs)
            ]

        self._results = [t.result() for t in tasks]
        return self._results
