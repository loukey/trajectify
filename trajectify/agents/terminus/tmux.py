"""TmuxSession — manages a tmux session inside the container."""

from __future__ import annotations

import asyncio
import logging

from trajectify.environments.base import BaseEnvironment
from trajectify.log import logger as global_logger

_DEFAULT_WAIT = 0.5
_MAX_OUTPUT_BYTES = 10_000


class TmuxSession:
    """Create and interact with a tmux session inside the environment."""

    def __init__(
        self,
        env: BaseEnvironment,
        session_name: str = "agent",
        width: int = 160,
        height: int = 40,
        logger: logging.Logger | None = None,
    ):
        self._env = env
        self._session = session_name
        self._width = width
        self._height = height
        self._logger = (logger or global_logger).getChild("TmuxSession")
        self._last_capture_lines: int = 0

    async def start(self) -> None:
        check = await self._env.exec("which tmux")
        if check.return_code != 0:
            self._logger.info("tmux not found; installing...")
            await self._env.exec(
                "apt-get update -qq && apt-get install -y -qq tmux"
                " || yum install -y tmux"
                " || apk add tmux"
            )

        await self._env.exec(
            f"tmux new-session -d -s {self._session} "
            f"-x {self._width} -y {self._height}"
        )
        await self._env.exec(
            f"tmux set-option -t {self._session} history-limit 100000"
        )
        self._logger.info(
            "tmux session '%s' started (%dx%d)",
            self._session, self._width, self._height,
        )

    async def stop(self) -> None:
        await self._env.exec(f"tmux kill-session -t {self._session}")

    async def send_keys(
        self,
        keys: str,
        wait_sec: float = _DEFAULT_WAIT,
    ) -> str:
        wait_sec = max(0.1, min(wait_sec, 60.0))

        escaped = keys.replace("'", "'\\''")
        await self._env.exec(
            f"tmux send-keys -t {self._session} -l '{escaped}'"
        )
        await self._env.exec(
            f"tmux send-keys -t {self._session} Enter"
        )

        await asyncio.sleep(wait_sec)

        return await self.capture_pane()

    async def capture_pane(self, full_history: bool = False) -> str:
        if full_history:
            cmd = f"tmux capture-pane -t {self._session} -p -S - -E -"
        else:
            cmd = f"tmux capture-pane -t {self._session} -p"

        result = await self._env.exec(cmd)
        output = (result.stdout or "").rstrip()

        if len(output) > _MAX_OUTPUT_BYTES:
            output = (
                output[:_MAX_OUTPUT_BYTES // 2]
                + "\n\n... [output truncated] ...\n\n"
                + output[-_MAX_OUTPUT_BYTES // 2:]
            )

        return output

    async def get_incremental_output(self) -> str:
        full = await self.capture_pane(full_history=True)
        lines = full.split("\n")
        new_lines = lines[self._last_capture_lines:]
        self._last_capture_lines = len(lines)
        return "\n".join(new_lines).rstrip()
