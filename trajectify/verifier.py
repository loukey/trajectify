"""Verifier — runs test.sh inside the container and reads the reward."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from trajectify.environments.base import BaseEnvironment
from trajectify.log import logger as global_logger
from trajectify.models.result import VerifierResult
from trajectify.models.task import Task

_CONTAINER_TESTS = "/tests"
_CONTAINER_VERIFIER_DIR = "/logs/verifier"


class Verifier:
    """Upload tests into the container, execute them, and collect the reward."""

    def __init__(
        self,
        task: Task,
        env: BaseEnvironment,
        host_verifier_dir: Path,
        logger: logging.Logger | None = None,
    ):
        self._task = task
        self._env = env
        self._host_verifier_dir = host_verifier_dir
        self._logger = (logger or global_logger).getChild("Verifier")

    async def verify(self) -> VerifierResult:
        await self._env.upload_dir(
            source=self._task.paths.tests_dir,
            target=_CONTAINER_TESTS,
        )

        # If the test script lives outside tests_dir, upload it separately
        test_script = self._task.paths.test_script_path
        try:
            test_script_rel = test_script.relative_to(self._task.paths.tests_dir)
        except ValueError:
            # Script is outside tests_dir (e.g. TB 1.0 run-tests.sh at task root)
            await self._env.upload_file(
                source=test_script,
                target=f"{_CONTAINER_TESTS}/{test_script.name}",
            )
            test_script_rel = Path(test_script.name)

        # Fix Windows line endings
        await self._env.exec(
            command=f"find {_CONTAINER_TESTS} -type f -exec sed -i 's/\\r$//' {{}} +"
        )
        stdout_container = f"{_CONTAINER_VERIFIER_DIR}/test-stdout.txt"

        result = await self._env.exec(
            command=(
                f"bash {_CONTAINER_TESTS}/{test_script_rel} 2>&1 "
                f"| tee {stdout_container}"
            ),
            env={"TEST_DIR": _CONTAINER_TESTS},
        )

        stdout_host = self._host_verifier_dir / "test-stdout.txt"
        if result.stdout and (
            not stdout_host.exists() or stdout_host.stat().st_size == 0
        ):
            stdout_host.write_text(result.stdout, encoding="utf-8")

        return self._read_reward(result.return_code)

    def _read_reward(self, return_code: int = 0) -> VerifierResult:
        reward_txt = self._host_verifier_dir / "reward.txt"
        reward_json = self._host_verifier_dir / "reward.json"

        if reward_txt.exists() and reward_txt.stat().st_size > 0:
            try:
                value = float(reward_txt.read_text(encoding="utf-8").strip())
                return VerifierResult(rewards={"reward": value})
            except ValueError as exc:
                raise RuntimeError(
                    f"Cannot parse reward.txt as float: {reward_txt}"
                ) from exc

        if reward_json.exists() and reward_json.stat().st_size > 0:
            try:
                data = json.loads(reward_json.read_text(encoding="utf-8"))
                return VerifierResult(rewards=data)
            except (ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Cannot parse reward.json: {reward_json}"
                ) from exc

        # No reward file — fall back to test script exit code
        # (TB 1.0 scripts don't write reward files)
        self._logger.info(
            "No reward file found; using exit code %d", return_code,
        )
        reward = 1.0 if return_code == 0 else 0.0
        return VerifierResult(rewards={"reward": reward})
